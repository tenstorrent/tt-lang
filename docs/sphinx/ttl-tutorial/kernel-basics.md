# Kernel Basics

## Kernel Function

A kernel function is a Python function decorated with `@ttl.kernel()`. It constructs a `ttl.Program` object by passing up to three thread functions and returns a callable that accepts TT-NN tensors.

```python
@ttl.kernel()
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

    return ttl.Program(some_compute, some_dm0, some_dm1)(x, y)

# Usage
shape = ttnn.Shape([128, 128])
x = ttnn.rand(shape, layout=ttnn.TILE_LAYOUT)
y = ttnn.zeros(shape, layout=ttnn.TILE_LAYOUT)
foo(x, y)
```

## Thread Functions

Thread functions are Python functions with no arguments, annotated by `@ttl.compute()` or `@ttl.datamovement()`. They are typically defined in the kernel function scope to capture shared objects.

**Compute threads** execute compute operations on blocks. **Data movement threads** handle memory transfers and synchronization. An analogy is a restaurant kitchen where the host program is the customer who places an order and receives the finished meal. Inside the kitchen, the first data movement thread acts as a prep cook fetching ingredients from storage, the compute thread is the line cook preparing the dish, and the second data movement thread serves as the server delivering the finished dish back to the customer.

```{mermaid}
graph TB
    Host[Host Program] -->|sends input data| DRAM[DRAM/L1]

    subgraph KernelFunction[Kernel Function on Tensix Core]
        DM1[Data Movement Thread 1<br/>Reader]
        CT[Compute Thread]
        DM2[Data Movement Thread 2<br/>Writer]
    end

    DRAM -->|reads from| DM1
    DM1 -->|writes to| CB1[Circular Buffer]
    CB1 -->|provides data| CT
    CT -->|writes to| CB2[Circular Buffer]
    CB2 -->|provides data| DM2
    DM2 -->|writes to| DRAM2[DRAM/L1]
    DRAM2 -->|returns results| Host
```

## Grid and Core Functions

### Grid Size

`ttl.grid_size(dims)` returns the size of the grid in the specified dimensionality. If requested dimensions differ from grid dimensions, the highest rank dimension is flattened or padded.

Think of the grid like an office building: a single-chip grid is one floor with an 8x8 arrangement of cubicles (cores). When you ask for a 1D view, you're counting all cubicles in a line (64 total). A multi-chip grid adds more floors, and you can choose whether to count by floor, by cubicle-within-floor, or flatten everything into one long hallway.

```python
# For (8, 8) single-chip grid
x_size = ttl.grid_size(dims=1)  # x_size = 64

# For (8, 8, 8) multi-chip grid
x_size, y_size = ttl.grid_size(dims=2)  # x_size = 8, y_size = 64

# For (8, 8) single-chip grid
x_size, y_size, z_size = ttl.grid_size(dims=3)  # x_size = 8, y_size = 8, z_size = 1
```

### Core Coordinates

`ttl.core(dims)` returns zero-based, contiguous core coordinates for the current Tensix core.

```python
# For (8, 8) single-chip grid
x = ttl.core(dims=1)  # x in [0, 64)

# For (8, 8, 8) multi-chip grid
x, y = ttl.core(dims=2)  # x in [0, 8), y in [0, 64)

# For (8, 8) single-chip grid
x, y, z = ttl.core(dims=3)  # x in [0, 8), y in [0, 8), z = 0
```

Both functions can be used inside kernel functions and thread functions.
