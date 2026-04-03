# Operation Basics

## Operation Function

An operation function is a Python function decorated with `@ttl.operation()`. Kernel functions defined inside are automatically collected and compiled into a program.

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

## Kernel Functions

Kernel functions are Python functions with no arguments, annotated by `@ttl.compute()` or `@ttl.datamovement()`. They are typically defined in the operation function scope to capture shared objects.

**Compute kernels** execute compute operations on blocks. **Data movement (DM) kernels** handle memory transfers and synchronization. An analogy is a restaurant where the host program is the customer who places an order for a multi-course meal. Inside the kitchen (a node), the first data movement kernel acts as a worker fetching ingredients from storage, the compute kernel is the cook preparing each course as soon as the ingredients are available, and the second data movement kernel is the server that brings each finished course to the customer as soon as it's ready. Multiple courses move through this pipeline at once—while one dish is being plated, another is cooking, and a third is being prepped.

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

## Grid and Node Functions

### Grid Size

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

### Node Coordinates

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
