# Dataflow Buffers

## Overview

A dataflow buffer is a communication primitive for synchronizing the passing of data between kernel functions within one node. An analogy is a conveyor belt in a factory: the producer (data movement kernel) places items onto the belt, and the consumer (compute kernel) picks them up. The belt has a fixed number of blocks, and when full, the producer must wait for the consumer to free up space.

A dataflow buffer is created with the `ttl.make_dataflow_buffer_like` function by passing a TT-NN tensor, shape, and block count.

The TT-NN tensor determines basic properties (likeness) such as data type and shape unit. The shape unit is a whole tile if the tensor has a tiled layout and is a scalar if the tensor has a row-major layout. Shape determines the shape of a block returned by one of the acquisition functions and is expressed in shape units. block count determines the total size of L1 memory allocated as a product of block size and block count. For the most common case block count defaults to 2 to enable double buffering.

```{mermaid}
graph LR
    DM[Data Movement Kernel] -->|reserve/push| DFB[Dataflow Buffer]
    DFB -->|wait/pop| CT[Compute Kernel]
```

## Acquisition Functions

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

## Example

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

## API Reference

| Function | Description |
| :---- | :---- |
| `ttl.make_dataflow_buffer_like(ttnn.Tensor: likeness_tensor, shape: ttl.Shape, block_count: ttl.Size) -> ttl.CircularBuffer` | Create a dataflow buffer by inheriting basic properties from `likeness_tensor`. |
| `ttl.CircularBuffer.reserve(self) -> ttl.Block` | Reserve and return a block from a dataflow buffer. **This function is blocking** and will wait until a free block is available. A free block is typically used by a producer to write the data into. |
| `ttl.CircularBuffer.push(self)` | Push a block to a dataflow buffer. This function is called by the producer to signal the consumer that a block filled with data is available. **This function is non-blocking.** |
| `ttl.CircularBuffer.wait(self) -> ttl.Block` | Wait for and return a block from a dataflow buffer. **This function is blocking** and will wait until a block filled with data is available. A filled block is typically used by a consumer to read data from. |
| `ttl.CircularBuffer.pop(self)` | Pop a block from a dataflow buffer. This function is called by the consumer to signal the producer that block is free and available. **This function is non-blocking.** |
