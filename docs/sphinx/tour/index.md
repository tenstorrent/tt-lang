# Tour of TT-Lang

TT-Lang is a Python-based domain-specific language for authoring high-performance custom operations on Tenstorrent hardware.

## Overview

TT-Lang provides an expressive middle ground between TT-NN's high-level operations and TT-Metalium's low-level hardware control. The language centers on explicit data movement and compute kernels with synchronization primitives familiar to TT-Metalium users (dataflow buffers, semaphores) alongside new abstractions (tensor slices, blocks, pipes).

## Key Concepts

- **Operation function**: Python function decorated with `@ttl.operation()` that defines kernel functions.
- **Kernel functions**: Decorated with `@ttl.compute()` or `@ttl.datamovement()`, these define compute and data movement logic.
- **Dataflow buffers**: Communication primitives for passing data between kernels within a node.
- **Blocks**: Memory acquired from dataflow buffers, used in compute expressions or copy operations.
- **Grid**: Defines the space of nodes for operation execution.

## Operation Basics

### Operation Function

An operation function is a Python function decorated with `@ttl.operation()`. Kernel functions defined inside the operation function are automatically collected and compiled into a program.

```python
@ttl.operation()
def foo(x: ttnn.Tensor, y: ttnn.Tensor) -> None:
    @ttl.compute()
    def some_compute():
        # compute logic

    @ttl.datamovement()
    def some_dm0():
        # data movement logic

    @ttl.datamovement()
    def some_dm1():
        # more data movement logic

# Usage
shape = ttnn.Shape([128, 128])
x = ttnn.rand(shape, layout=ttnn.TILE_LAYOUT)
y = ttnn.zeros(shape, layout=ttnn.TILE_LAYOUT)
foo(x, y)
```

### Kernel Functions

Kernel functions are Python functions with no arguments, annotated by `@ttl.compute()` or `@ttl.datamovement()`. They are typically defined in the operation function scope to capture shared objects.

**Compute kernels** execute computations (e.g., math) on blocks. **Data movement (DM) kernels** handle memory transfers and synchronization. An analogy is a restaurant where the host program is the customer who places an order for a multi-course meal. Inside the kitchen (a node), the first data movement kernel acts as a worker fetching ingredients from storage, the compute kernel is the cook preparing each course as soon as the ingredients are available, and the second data movement kernel is the server that brings each finished course to the customer as soon as it's ready. Multiple courses move through this pipeline at once—while one dish is being plated, another is cooking, and a third is being prepped.

```{mermaid}
graph TB
    Host["Host Program<br/>(🧑 Customer)"] -->|sends input data| DRAM["DRAM/L1<br/>(🍚🐟🥒🥑 Ingredients)"]

    subgraph OperationFunction["Operation Function on a node (Kitchen)"]
        subgraph pad[" "]
            subgraph kernels[" "]
                DM1["DM Kernel 1<br/>Reader (🧑🏻 Prep Cook)"]
                CT["Compute Kernel<br/>(👩🏽‍🍳 Cook)"]
                DM2["DM Kernel 2<br/>Writer (👧🏼 Server)"]
            end
        end
    end

    DRAM -->|reads from| DM1
    DM1 -->|writes to| CB1["Dataflow Buffer<br/>(🔔 Ingredients ready)"]
    CB1 -->|provides data| CT
    CT -->|writes to| CB2["Dataflow Buffer<br/>(🔔 Course ready)"]
    CB2 -->|provides data| DM2
    DM2 -->|writes to| DRAM2["DRAM/L1<br/>(🍱 Ready to eat course)"]
    DRAM2 -->|returns results| Host

    classDef invisible fill:none,stroke:none;
    class pad,threads invisible;
```

### Grid and Node Functions

#### Grid Size

`ttl.grid_size(dims)` returns the size of the grid in the specified dimensionality. If requested dimensions differ from grid dimensions, the highest rank dimension is flattened or padded.

An analogy is an office building: a single-chip grid is one floor with an 8x8 arrangement of cubicles (nodes). A 1D view counts all cubicles in a line (64 total). A multi-chip grid adds more floors, and the view can count by floor, by cubicle-within-floor, or flatten everything into one long hallway.

```python
# For (8, 8) single-chip grid
x_size = ttl.grid_size(dims=1)  # x_size = 64

# For (8, 8, 8) multi-chip grid
x_size, y_size = ttl.grid_size(dims=2)  # x_size = 8, y_size = 64

# For (8, 8) single-chip grid
x_size, y_size, z_size = ttl.grid_size(dims=3)  # x_size = 8, y_size = 8, z_size = 1
```

#### Node Coordinates

`ttl.node(dims)` returns zero-based, contiguous node coordinates for the current node.

```python
# For (8, 8) single-chip grid
x = ttl.node(dims=1)  # x in [0, 64)

# For (8, 8, 8) multi-chip grid
x, y = ttl.node(dims=2)  # x in [0, 8), y in [0, 64)

# For (8, 8) single-chip grid
x, y, z = ttl.node(dims=3)  # x in [0, 8), y in [0, 8), z = 0
```

Both functions can be used inside operation functions and kernel functions.

## Dataflow Buffers

A dataflow buffer is a communication primitive for synchronizing the passing of data between kernel functions within one node. An analogy is a conveyor belt in a factory: the producer (data movement kernel) places items onto the belt, and the consumer (compute kernel) picks them up. The belt has a fixed number of blocks, and when full, the producer must wait for the consumer to free up space.

A dataflow buffer is created with the `ttl.make_dataflow_buffer_like` function by passing a TT-NN tensor, shape, and block count.

The TT-NN tensor determines basic properties (likeness) such as data type and shape unit. The shape unit is a whole tile if the tensor has a tiled layout and is a scalar if the tensor has a row-major layout. Shape determines the shape of a block returned by one of the acquisition functions and is expressed in shape units. block count determines the total size of L1 memory allocated as a product of block size and block count. For the most common case block count defaults to 2 to enable double buffering.

```{mermaid}
graph LR
    DM[Data Movement Kernel] -->|reserve/push| DFB[Dataflow Buffer]
    DFB -->|wait/pop| CT[Compute Kernel]
```

### Acquisition Functions

There are two acquisition functions on a dataflow buffer object: `wait` and `reserve`. A dataflow buffer is constructed in the scope of the operation function but its object functions can only be used inside of kernel functions.

Acquisition functions can be used with Python `with` statement, which automatically releases acquired blocks at the end of the `with` scope—like checking out a library book that is automatically returned when leaving the reading room. Alternatively, if acquisition functions are used without `with`, a corresponding release function must be called explicitly: `pop` for `wait` and `push` for `reserve`.

**Producer-consumer flow:**

```{mermaid}
sequenceDiagram
    participant Producer as Data Movement
    participant DFB as dataflow buffer
    participant Consumer as Compute

    Producer->>DFB: reserve() - wait for free entry
    Note over Producer: Write data to block
    Producer->>DFB: push() - mark as filled
    Consumer->>DFB: wait() - wait for filled entry
    Note over Consumer: Read/process data
    Consumer->>DFB: pop() - mark as free
```

### Example

```python
x_dfb = ttl.make_dataflow_buffer_like(x,
    shape = (2, 2),
    block_count = 2)

@ttl.datamovement()
def some_read():
    with x_dfb.reserve() as x_blk:
        # produce data into x_blk ...
        # implicit x_dfb.push() at the end of the scope

@ttl.compute()
def some_compute():
    x_blk = x_dfb.wait()
    # consume data in x_blk ...
    x_blk.pop() # explicit
```

### API Reference

`ttl.CircularBuffer` is preserved as a deprecated alias for `ttl.DataflowBuffer`;
new code should use `DataflowBuffer`.

| Function | Description |
| :---- | :---- |
| `ttl.make_dataflow_buffer_like(ttnn.Tensor: likeness_tensor, shape: ttl.Shape, block_count: ttl.Size) -> ttl.DataflowBuffer` | Create a dataflow buffer by inheriting basic properties from `likeness_tensor`. |
| `ttl.DataflowBuffer.reserve(self) -> ttl.Block` | Reserve and return a block from a dataflow buffer. **This function is blocking** and will wait until a free block is available. A free block is typically used by a producer to write the data into. |
| `ttl.DataflowBuffer.push(self)` | Push a block to a dataflow buffer. This function is called by the producer to signal the consumer that a block filled with data is available. **This function is non-blocking.** |
| `ttl.DataflowBuffer.wait(self) -> ttl.Block` | Wait for and return a block from a dataflow buffer. **This function is blocking** and will wait until a block filled with data is available. A filled block is typically used by a consumer to read data from. |
| `ttl.DataflowBuffer.pop(self)` | Pop a block from a dataflow buffer. This function is called by the consumer to signal the producer that block is free and available. **This function is non-blocking.** |

## Reference

For the complete language specification, see [TT-Lang Specification](../specs/TTLangSpecification.md).
