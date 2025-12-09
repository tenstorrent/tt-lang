# TT-Lang

# Language Specification

## Introduction

TT-Lang is a Python based *domain specific language (DSL)* designed to express kernel programs for TT hardware. While based on Python the language maintains a number of constraints to what parts of Python can be used in what context, hence the DSL nature of it. TT-Lang is tightly integrated with [TT-NN](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html) to provide seamless experience of mixing existing TT-NN operations and user-defined kernel programs.

The programming model of TT-Lang is centered around explicit specification of data movement and compute threads and explicit synchronization between them. This allows the user to have fine grained control of the execution schedule and its performance implications. TT-Lang offers abstractions familiar to TT-Metalium users such as *circular buffers* and *semaphores*. TT-Lang also offers new, higher level abstractions, such as *tensor slices*, *blocks* and *pipes* that wrap the complexity of dealing with tensor memory layout, compute API and core-to-core communication correspondingly.

## Kernel program

*Kernel function* is a Python function with an optional ttl.kernel decorator. This function constructs a Program object by passing up to three *thread functions*. The Program object is callable with input and output [*TT-NN tensors*](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/tensor.html). A thread function is a Python function with no arguments annotated by ttl.compute or ttl.datamovement decorators. Typically thread functions are defined in the scope for the kernel function so that they can capture objects shared by all thread functions.

## Example

```py
@ttl.kernel()
def foo(
    x: ttnn.Tensor,
    y: ttnn.Tensor,
) -> None:
    @ttl.compute()
    def some_compute():
        # ...

    @ttl.datamovement()
    def some_dm0():
        # ...

    @ttl.datamovement()
    def some_dm1():
        # ...

    return ttl.Program(some_compute, some_dm0, some_dm1)(x, y)

shape = ttnn.Shape([128, 128])

x = ttnn.rand(shape, layout=ttnn.TILE_LAYOUT)
y = ttnn.zeros(shape, layout=ttnn.TILE_LAYOUT)

foo(x, y)
```

## Grid

A *grid* defines a space of Tensix cores to which the kernel is submitted for execution. In a single-chip case it is two dimensional. In a multi-chip case it has three or more dimensions representing different levels of connectivity (same card, same host, same rack etc).

## Grid size function

The ttl.grid\_size function returns the size of the grid. The function takes an argument that specifies how many dimensions to return. If requested dimensions are smaller than grid dimensions, the highest rank dimension is flattened. If requested dimensions are greater than grid dimensions, highest rank dimensions are padded with one. The ttl.grid\_size can be used inside a kernel function as well as inside thread functions.

| Function | Description |
| :---- | :---- |
| ttl.grid\_size(dims: int) \-\> int | Tuple\[int, ...\] | Return grid size in specified dimensionality. |

## Example

```py
# for (8, 8) single chip grid gets x_size = 64
x_size = ttl.grid_size(dims = 1)

# for (8, 8, 8) multi-chip grid gets x_size = 8, y_size = 64
x_size, y_size = ttl.grid_size(dims = 2)

# for (8, 8) single-chip grid gets x_size = 8, y_size = 8, z_size = 1
x_size, y_size, z_size = ttl.grid_size(dims = 3)
```

## Core function

The ttl.core function returns *core coordinates* of the current Tensix core. The function takes an argument that specifies how many dimensions to return. If requested dimensions are smaller than grid dimensions, the highest rank dimension is flattened. If requested dimensions are greater than grid dimensions, highest rank dimensions are padded with zero. The ttl.core can be used inside a kernel function as well as inside thread functions.

| Function | Description |
| :---- | :---- |
| ttl.core(dims: int) \-\> int | Tuple\[int, ...\] | Return core coordinates in specified dimensionality. |

## Example

```py
# for (8, 8) single chip grid gets x = [0, 64)
x = ttl.core(dims = 1)

# for (8, 8, 8) multi-chip grid gets x = [0, 8), y = [0, 64)
x, y = ttl.core(dims = 2)

# for (8, 8) single-chip gets x = [0, 8), y = [0, 8), z = 0
x, y, z = ttl.core(dims = 3)
```

## Circular buffer

A *circular buffer* is a communication primitive for synchronizing passing data between thread functions. A circular buffer is created with the ttl.make\_circular\_buffer\_like function by passing TT-NN tensor, *shape* and *buffer factor*. The TT-NN tensor determines basic properties (likeness) such as data type and *shape unit*. The shape unit is a whole tile if the tensor has a tiled layout and is a scalar if the tensor has a row-major layout.) Shape determines the shape of a *block* returned by one of the *acquisition functions* and is expressed in corresponding shape units. Buffer factor determines the total size of L1 memory allocated as a product of block size and buffer factor. For the most common double buffering case buffer factor defaults to 2\.

There are two acquisition functions on a circular buffer object: wait and reserve. A circular buffer is constructed in the scope of the kernel function but its object functions can only be used inside of thread functions. Acquisition functions can be used with Python with statement, which will automatically release acquired blocks at the end of the with scope. Alternatively, if acquisition functions are used without the with the user must explicitly call a corresponding release function: pop for wait and push for reserve.

## Example

```py
    x_cb = ttl.make_circular_buffer_like(x,
        shape = (2, 2),
        buffer_factor = 2)

    @ttl.datamovement()
    def some_read():
        with x_cb.reserve() as x_blk:

            # produce data into x_blk ...

            # implicit x_cb.push() at the end of the scope

    @ttl.compute()
    def some_compute():
        x_blk = x_cb.wait()

        # consume data in x_blk ...

        x_cb.pop() # explicit
```

##

| Type alias/Function | Description |
| :---- | :---- |
| ttl.Shape \= Tuple\[int, ...\] | Shape of a circular buffer. |
| ttl.make\_circular\_buffer\_like(   ttnn.Tensor: likeness\_tensor,   shape: ttl.Shape,   buffer\_factor: int) \-\> ttl.CircularBuffer | Create a circular buffer by inheriting basic properties from likeness\_tensor. |
| ttl.CircularBuffer.reserve(self) \-\> ttl.Block | Reserve and return a block from a circular buffer. **This function is blocking** and will wait until a *free* block is available. A free block is typically used by a producer to write the data into. |
| ttl.CircularBuffer.push(self) | Push a block to a circular buffer. This function is called by the producer to signal the consumer that a block *filled* with data is available. **This function is non-blocking.** |
| ttl.CircularBuffer.wait(self) \-\> ttl.Block | Wait for and return a block from a circular buffer. **This function is blocking** and will wait until a block filled with data is available. A filled block is typically used by a consumer to read data from. |
| ttl.CircularBuffer.pop(self) | Pop a block from a circular buffer. This function is called by the consumer to signal the producer that block is free and available. **This function is non-blocking.** |

## Block

A *block* represents memory acquired from a circular buffer. Block size is determined by the shape of a circular buffer and its memory is allocated when a circular buffer is created. Inside of a compute thread a block can participate in an expression as an input and as a storage for a result by using store function. Inside of data movement threads a block can also participate in ttl.copy as source or destination.

## Example

```py
    @ttl.datamovement()
    def some_read():
        # acquire a_blk and b_blk ...

        # source is a tensor slice, destination is a block
        a_xf = ttl.copy(a[0], a_blk)
        b_xf = ttl.copy(b[0], b_blk)
        a_xf.wait()
        b_xf.wait()

    @ttl.compute()
    def some_compute():
        # acquire a_blk, b_blk and y_blk ...

        a_squared = a_blk ** 2
        b_squared = b_blk ** 2
        y = ttl.math.sqrt(a_squared + b_squared)
        y_blk.store(y)

    @ttl.datamovement()
    def some_write():
        # acquire y_blk ...

        # source is a block, destination is a tensor slice
        y_xf = ttl.copy(y_blk, y[0])
        y_xf.wait()
```

##

| Function | Description |
| :---- | :---- |
| ttl.Block.store(self, expr: ttl.BlockExpr) | This function materializes the result of a *block expression* and stores it in the block. Block expression uses Python builtin math operators and ttl.math.xxx functions on block expression. **This function is blocking** so that block is safe to use immediately after the call. |
| ttl.math.sqrt(expr: ttl.BlockExpr) \-\> ttl.BlockExpr | Example of math function. |
| ttl.BlockExpr.\_\_add\_\_(   self,   other: ttl.BlockExpr) \-\> ttl.BlockExpr | Example of math operator. |

##

## Pipe

A *pipe* is an object representing data movement of blocks between Tensix cores and is used as source and destination in the ttl.copy. The pipe is constructed with source core coordinate (src\_core) and destination as either a single core coordinate for unicast (dst\_core) or *core range* for multicast (dst\_core\_range). The core range uses a combination of dimension slices and values to describe a contiguous hypercube. The core range dimensions’ aspects will match the corresponding aspects returned by the grid\_size function for the same number of dimensions.

| Type alias/Function | Description |
| :---- | :---- |
| ttl.CoreAddr \= Tuple\[int, ...\] | Core coordinates. |
| ttl.CoreRange \= Tuple\[int | slice, ...\] | Core range. |
| ttl.Pipe(   src\_core: ttl.CoreAddr,   dst\_core: CoreAddr \= None,   dst\_core\_range: CoreRange \= None) \-\> ttl.Pipe | Constructs pipe description to be used in to construct pipe net. Either dst\_core or dst\_core\_range but not both must be specified. |

## Pipe net

A *pipe net* is an object representing a network of pipes. A pipe net is constructed from a list of pipes and encapsulates all necessary information to determine if a given core is source, destination or both and where and from which core or cores the corresponding transfers will occur. Pipe net object has two functions: if\_src and if\_dst. Both functions have a single argument: *condition body function*.

Condition body function is invoked for each pipe in case of if\_src if the current core is a source, and in case of if\_dst if the current core is a destination. The condition body function has a single argument: the pipe that satisfies the condition. Condition body function can identify the destination pipe and source pipe by its src\_core, and  dst\_core and dst\_core\_range read-only properties correspondingly.

A pipe net is constructed in the scope of the kernel function but can only be used with its if\_src and if\_dst functions inside of the data movement thread function. The corresponding  ttl.copy where a pipe is a source or destination can be called only inside of condition body function. Calls into if\_src and if\_dst can be nested within condition functions for different pipe nets.

| Function | Description |
| :---- | :---- |
| ttl.PipeNet(pipes: List\[ttl.Pipe\]) \-\> ttl.PipeNet | Constructs pipe net. |
| ttl.PipeNet.if\_src(   Self,   cond\_fun: Callable\[\[ttl.SrcPipeIdentity\], None\]) | Call condition function for each pipe in the pipe net that is a source. |
| ttl.PipeNet.if\_dst(   self,   cond\_fun: Callable\[\[ttl.DstPipeIdentity\], None\]) | Call condition function for each pipe in the pipe net that is a destination. |
| @property ttl.SrcPipeIdentity.dst\_core(self) \-\> ttl.CoreAddr | Get destination core for pipe in if\_src. |
| @property ttl.SrcPipeIdentity.dst\_core\_range(self) \-\> ttl.CoreRange | Get destination core range for pipe in if\_src. |
| @property ttl.DstPipeIdentity.src\_core(self) \-\> ttl.CoreAddr | Get source core for pipe in if\_dst. |

##

## Gather example

```py
# Grid:
#
# column
# x == 0
#   |
#   V
# (0, 0) (1, 0) (2, 0) (3, 0) <-- row y == 0
# (0, 1) (1, 1) (2, 1) (3, 1)
# (0, 2) (1, 2) (2, 2) (3, 2)
# (0, 3) (1, 3) (2, 3) (3, 3)

# ---------------------
# gather from row y to (0, y) with unicast

grid_x, grid_y = ttl.grid_size()

net = ttl.PipeNet([ttl.Pipe(
    src_core = (x, y),
    dst_core = (0, y)) for x in range(1, grid_x) for y in range(grid_y)])

# (1, 0) -> (0, 0) |             |
# (2, 0) -> (0, 0) | sequential  |
# (3, 0) -> (0, 0) |             |
# ...              |             | concurrent
#                                |
# (1, 1) -> (0, 1)               |
# ...                            |

    @ttl.datamovement()
    def dm():
        # reserve blk
        # ...
        blk = ...

        def pipe_src(pipe):
            # write data into blk
            # ...

            # then copy blk to pipe
            xf = ttl.copy(blk, pipe)
            xf.wait()

        def pipe_dst(pipe):
            # copy blk from pipe
            xf = ttl.copy(pipe, blk)
            xf.wait()

            # then read data from blk
            # ...

        net.if_src(pipe_src)
        net.if_dst(pipe_dst)
```

## Scatter example

```py
# ---------------------
# scatter from (x, 0) to column x with multicast

grid_x, grid_y = ttl.grid_size()

net = ttl.PipeNet([ttl.Pipe(
    src_core = (x, 0),
    dst_core_range = (x, slice(1, grid_y))) for x in range(grid_x)])

# (0, 0) => (0, 1) (0, 2) (0, 3) ... |
# (1, 0) => (1, 1) (1, 2) (1, 3) ... | concurrent
# ...                                |

    @ttl.datamovement()
    def dm():
        # reserve blk
        # ...
        blk = ...

        def pipe_src(pipe):
            # write data into blk
            # ...

            # then copy blk to pipe
            xf = ttl.copy(blk, pipe)
            xf.wait()

        def pipe_dst(pipe):
            # copy blk from pipe
            xf = ttl.copy(pipe, blk)
            xf.wait()

            # then read data from blk
            # ...

        net.if_src(pipe_src)
        net.if_dst(pipe_dst)
```

## Scatter-gather example

```py
# ---------------------
# scatter-gather column x with multicast/loopback

grid_x, grid_y = ttl.grid_size()

net = ttl.PipeNet([ttl.Pipe(
    src_core = (x, y),
    dst_core_range = (x, slice(0, grid_y))) for x in range(grid_x) for y in range(grid_y)])

# (0, 0) => (0, 0) (0, 1) (0, 2) ... |            |
# (0, 1) => (0, 0) (0, 1) (0, 2) ... | sequential |
# (0, 2) => (0, 0) (0, 1) (0, 2) ... |            |
# ...                                |            | concurrent
#                                                 |
# (1, 0) => (1, 0) (1, 1) (1, 2) ...              |
# ...                                             |

    @ttl.datamovement()
    def dm():
        # reserve blk
        # ...
        blk = ...

        def pipe_src(pipe):
            # write data into blk
            # ...

            # then copy blk to pipe
            xf = ttl.copy(blk, pipe)
            xf.wait()

        def pipe_dst(pipe):
            # copy blk from pipe
            xf = ttl.copy(pipe, blk)
            xf.wait()

            # then read data from blk
            # ...

        net.if_src(pipe_src)
        net.if_dst(pipe_dst)
```

## Forward to a \+1 neighbor example

```py
# ---------------------
# forward to a +1 neighbor in a column x

grid_x, grid_y = ttl.grid_size()

net = ttl.PipeNet([ttl.Pipe(
    src_core = (x, y),
    dst_core = (x, (y + 1) % grid_y)) for x in range(grid_x) for y in range(grid_y)])

# (0, 0) => (0, 1)  |
# (0, 1) => (0, 2)  |
# ...               |
# (0, 7)* => (0, 0) |
# ...               | concurrent
#                   |
# (1, 0) => (1, 1)  |
# ...               |
#
# * - assuming (8, 8) grid

    @ttl.datamovement()
    def dm():

        # reserve blk_to_send and blk_received
        # ...
        blk_to_send = ...
        blk_received = ...

        def pipe_src(pipe):
            # write data into blk_to_send
            # ...

            # then copy blk to blk_to_send
            xf = ttl.copy(blk_to_send, pipe)
            xf.wait()

        def pipe_dst(pipe):
            # copy blk_received from pipe
            xf = ttl.copy(pipe, blk_received)
            xf.wait()

            # then read data from blk_received
            # ...

        net.if_src(pipe_src)
        net.if_dst(pipe_dst)
```

## Tensor slice

A *tensor slice* is a view into a TT-NN tensor defined in terms of a dimension slice or value for each of the tensor's dimensions. A tensor slice can participate in ttl.copy as source or destination with the corresponding destination and source being a block. Tensor slice can only be used in the scope of data movement thread function.

| Function | Description |
| :---- | :---- |
| ttnn.Tensor.\_\_getitem\_\_(   self,   \*index: int | slice) \-\> ttl.TensorSlice | Get a tensor slice from a TT-NN tensor. |

## Example

```py
    g = 2 # granularity

    row_tiles = a.shape[0] // ttl.TILE_SHAPE[0]
    col_tiles = a.shape[1] // ttl.TILE_SHAPE[1]
    cols_per_core = math.ceil(col_tiles / (grid_size(dim = 1)))

    core_num = core(dims = 1)
    start_ct = core_num * cols_per_core
    end_ct = min(start_ct + cols_per_core, col_tiles)

    @ttl.datamovement()
    def dm():
        for ct in range(start_ct, end_ct):
            for rt in range(row_tiles // g):
                # acquire a_blk...

                a_xf = ttl.copy(
                    a[(rt * g):((rt + 1) * g), ct],
                    a_blk)
                a_xf.wait()
```

## Copy

ttl.copy function expresses a variety of data movements that always have two arguments: source and destination. ttl.copy returns a *transfer handle* object. The transfer handle has a wait function that serves as a barrier. When the wait returns the transfer is complete and data in the destination is safe to use.  The ttl.copy can only be used inside of the data movement thread function.

| Function | Description |
| :---- | :---- |
| ttl.copy(   src: ttl.Block,   dst: ttl.TensorSlice) \-\> ttl.TransferHandle ttl.copy(   src: ttl.TensorSlice,   dst: ttl.Block) \-\> ttl.TransferHandle ttl.copy(   src: ttl.Block,   dst: ttl.PipeIdentity) \-\> ttl.TransferHandle ttl.copy(   src: ttl.PipeIdentity,   dst: ttl.Block) \-\> ttl.TransferHandle | Copy data between a block, a tensor slice, or a pipe. **This function is non-blocking.** The compiler statically checks if the shape of block and tensor slice are compatible and if the shape of block sent to a pipe is compatible with the shape of block received from the same pipe. When a pipe is used as a destination there must be a corresponding ttl.copy where the same pipe is used as source. Furthermore, ttl.copy with pipe must be guarded by pipe net’s if\_src and is\_dst where this pipe is destination and source correspondingly. |
| ttl.TransferHandle.wait() | Wait for data transfer to complete. **This function is blocking.** |

## Semaphore

A *semaphore* is a communication primitive for synchronizing data transfers between data movement threads on different Tensix cores. Each semaphore has an associated 32-bit unsigned integer *semaphore value* for each Tensix core. This value can be changed (set or incremented) by a data movement thread on the local or a remote core. When changing semaphore value remotely a single core coordinate for unicast change or a core range for multicast change is specified. Only setting the semaphore value is supported as multicast change. A data movement can wait on semaphore until its value satisfies a condition. It is possible to specify either a condition with exact value or a condition with minimum value. Only local data movement threads can wait on a semaphore.

ttl.Semaphore class is constructed with its initial value that defaults to zero. A ttl.Semaphore instance can be constructed in kernel function scope. A ttl.Semaphore instance provides wait\_eq, wait\_ge and set functions for managing local semaphore value. To change remote semaphore value an instance of ttl.UnicastRemoteSemaphore or ttl.MulticastRemoteSemaphore is obtained by calling get\_remote and get\_remote\_multicast functions correspondingly. The ttl.UnicastRemoteSemaphore supports inc and set while ttl.MulticastRemoteSemaphore supports only set. Functions that change the value or wait on condition can be used only in the scope of a data movement thread function. Functions that obtain remote semaphores can be used in both kernel and thread function scopes.

## One-to-many barrier example

```py
    core_num = core(dims = 1)
    my_barrier = ttl.Semaphore()
    all_barrier = my_barrier.get_remote_multicast()

    @ttl.datamovement()
    def dm():
        if core_num == 0:
            # do something on core 0 while non-0 cores wait...
            all_barrier.set(1)
        else:
            my_barrier.wait_eq(1)
            # core 0 is done
```

## Many-to-one barrier example

```py
    core_num = core(dims = 1)
    my_barrier = ttl.Semaphore()
    core_0_barrier = my_barrier.get_remote((0, 0))
    non_0_core_count = grid_size(dim = 1) - 1

    @ttl.datamovement()
    def dm():
        if core_num != 0:
            # do something on non-0 cores while core 0 waits...
            core_0_barrier.inc(1)
        else:
            my_barrier.wait_eq(non_0_core_count)
            # non-0 cores are done
```

| Function | Description |
| :---- | :---- |
| ttl.Semaphore.wait\_eq(self, value: int) | Wait until the local semaphore value is equal to specified value. **This function is blocking.** Can be used only in the scope of a data movement thread function. |
| ttl.Semaphore.wait\_ge(self, value: int) | Wait until the local semaphore value is greater or equal to specified value. **This function is blocking.** Can be used only in the scope of a data movement thread function. |
| ttl.Semaphore.set(self, value: int) | Set the local semaphore value to specified value. **This function is non-blocking.** Can be used only in the scope of a data movement thread function. |
| ttl.Semaphore.get\_remote(self, ttl.CoreAddr: core) \-\> ttl.UnicastRemoteSemaphore | Get remote unicast semaphore for specified core coordinate. Returns an instance of UnicastRemoteSemaphore. Can be used in both kernel and thread function scopes. |
| ttl.Semaphore.get\_remote\_multicast(self, ttl.CoreRange: core\_range) \-\> ttl.MulticastRemoteSemaphore | Get remote multicast semaphore for specified core range. When called with no arguments returns remote multicast semaphore for the entire grid. Returns an instance of MulticastRemoteSemaphore. Can be used in both kernel and thread function scopes. |
| ttl.UnicastRemoteSemaphore.set(self, value: int) | Set remote unicast semaphore value to specified value. **This function is non-blocking.** Can be used only in the scope of a data movement thread function. |
| ttl.UnicastRemoteSemaphore.inc(self, value: int) | Increment remote unicast semaphore value by specified value. **This function is non-blocking.** Can be used only in the scope of a data movement thread function. |
| ttl.MulticastRemoteSemaphore.set(self, value: int) | Set remote multicast semaphore value to specified value. **This function is non-blocking.** Can be used only in the scope of a data movement thread function. |

## Glossary

| Term | Description |
| :---- | :---- |
| *Domain specific language (DSL)* |  |
| *Kernel function* |  |
| *Thread function* |  |
| *Data movement thread function* |  |
| *Compute thread function* |  |
| *TT-NN tensor* |  |
| *Grid* |  |
| *Core coordinates* |  |
| *Circular buffer* |  |
| *Circular buffer’s shape* |  |
| *Circular buffer’s shape unit* |  |
| *Circular buffer’s buffer factor* |  |
| *Circular buffer’s acquisition functions* |  |
| *Circular buffer’s release functions* |  |
| *Block* |  |
| *Block expression* |  |
| *Pipe* |  |
| *Pipe net* |  |
| *Pipe net’s condition body function* |  |
| *Tensor slice* |  |
| *Transfer handle* |  |
| *Sempahore* |  |
| *Semaphore value* |  |

# Discussion

## Principles

TT-Lang is a Python-based DSL that enables authoring of programs for TT-hardware at the abstraction level similar to SoTA “tile-level” DSLs for GPUs with. Following are the TT-Lang principles, in the order of significance:

1. Ability to express optimizations that achieve **performance within close range (95%)** of hand written TT-Metalium programs;
2. Robust and safe abstractions capable of representing a simplified model of hardware that **eliminates whole classes of mistakes** that are possible when writing TT-Metalium programs; Specifically:
   1. Reduce duplication of information that is typical in mult-threaded separation of kernels;
   2. Infer CBs operations to eliminate errors in asynchronous code that would be causing hangs or data races (All in single threaded, pop/push guarded by “with” scope in multithreaded);
   3. Infer xxx\_init/xxx\_tile(s) etc calls based on functional compute expression;
   4. Use compile time memory allocation (DRAM, L1 and DST register) to eliminate OOMs and clobberring at runtime;
   5. In addition to (d) use relative memory sizing instead of explicit memory sizing to eliminate OOMs at runtime. With such relative memory sizing the actual size can be maximized at compile time by the allocator or autotuned (see below) at runtime;
3. Allow TT-Lang programs to be **portable across multiple generations** of TT-hardware. Enable generation-specific details to be expressed as autotunable hyper-parameters;
4. SoTA **ergonomics**. Specifically:
   1. Functional simulator;
   2. VSCode (or similar) integration via language server;
   3. As-you-type compilation errors;
   4. As-you-type sanitization (based on functional simulator) errors;
   5. VSCode integrated line-by-line profiler (ala NSight);
5. Ability to be authored by **Generative AI** from scratch or in translation from “tile-level” DSLs for GPUs. Ability for the compiler, the sanitizer and the simulator to provide ergonomic errors, warnings, correctness and performance feedback for Generative AI to be able to iterate in an agentic workflow.
6. Ability to **autotune** within a space of user-defined hyper-parameters;
7. Ability to **serve as a bootstrap (EmitMetal)** that generates C++ TT-Metalium program for further optimization;
8. Ability to **augment TT-NN programs** with custom TT-Lang kernels;
9. Being **Python-based** as to support a limited subset of Python to express programs as well as being able to integrate into the Python environment. This makes TT-Lang more familiar and convenient for the target audience;
10. Ability to develop TT-Lang programs **out of tree** and without rebuilding TT-NN from source;

## Outcomes

There is a number of outcomes we are looking for that motivate TT-Lang:

* Adoption by internal Models Team as a tool that materially speeds up supporting new models in inference;
* Adoption by internal Training Team as a tool that enables fast iteration and experimentation without sacrificing performance;
* Adoption by external users on inference and training tracks as an authoring tool that leverages their experiences with “tile-level” DSLs for GPUs and provides robust abstraction over multiple generations of TT-hardware.

## Questions

1) The programming model can be either single-threaded with a program expressed as a synchronous dataflow using load/store and math operations or it can be multi-threaded and asynchronous with separate data movement and compute kernels using abstractions mapped to CBs and NOC transfers. Can both be supported? If so, which one do we start with?
   1) In the initial milestone we plan for multi-threaded model will allow the author to fully control the pipeline and order of operations as well as require explicit synchronization;
   2) The single-threaded model will allow the compiler to “design” the pipeline, reorder operations when necessary and infer necessary synchronization. We will explore the single-threaded model in the context of evaluation of applicability of SoTA “tile-level” DSLs.
2) Explicit loop nests versus metadata declarations. For-temporal? For-spatial?
   1) We want to provide a choice of expressing for-temporal loops as either explicit for statements in Python or implicitly as specified in metadata declarations.
   2) For-spacial looks would only be specified implicitly by grid metadata.
3) Python code for DSL can be either analyzed at AST level or traced. How much empirical runtime code is allowed/needed? What do we need to write a performant FA?
   1) We will take the approach of taking AST representation from the kernel's Python code. This will limit what can be used in kernel’s code to a subset of Python that is representable in Arith, Scf and TT-Kernel dialects. We will allow utility functions with the same limitation to be called from kernel’s code.
4) What is the user experience? Is TT-Lang embedded in TT-NN? In PyTorch? Standalone?
   1) TT-NN integration that allows mixing TT-NN code with TT-Lang. TT-Lang will be installed as a separate wheel compatible with TT-NN.
   2) It is unclear if we need standalone mode or PyTorch integration.

#
