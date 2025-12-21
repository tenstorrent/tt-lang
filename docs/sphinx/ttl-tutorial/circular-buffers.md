# Circular Buffers

## Overview

A circular buffer is a communication primitive for synchronizing the passing of data between thread functions within one Tensix core. Think of it like a conveyor belt in a factory: the producer (data movement thread) places items onto the belt, and the consumer (compute thread) picks them up. The belt has a fixed number of slots, and when full, the producer must wait for the consumer to free up space.

A circular buffer is created with the `ttl.make_circular_buffer_like` function by passing a TT-NN tensor, shape, and buffer factor.

The TT-NN tensor determines basic properties (likeness) such as data type and shape unit. The shape unit is a whole tile if the tensor has a tiled layout and is a scalar if the tensor has a row-major layout. Shape determines the shape of a block returned by one of the acquisition functions and is expressed in shape units. Buffer factor determines the total size of L1 memory allocated as a product of block size and buffer factor. For the most common case buffer factor defaults to 2 to enable double buffering.

```{mermaid}
graph LR
    DM[Data Movement Thread] -->|reserve/push| CB[Circular Buffer]
    CB -->|wait/pop| CT[Compute Thread]
```

## Acquisition Functions

There are two acquisition functions on a circular buffer object: `wait` and `reserve`. A circular buffer is constructed in the scope of the kernel function but its object functions can only be used inside of thread functions.

Acquisition functions can be used with Python `with` statement, which will automatically release acquired blocks at the end of the `with` scope—like checking out a library book that's automatically returned when you leave the reading room. Alternatively, if acquisition functions are used without the `with` the user must explicitly call a corresponding release function: `pop` for `wait` and `push` for `reserve`.

**Producer-consumer flow:**

```{mermaid}
sequenceDiagram
    participant Producer as Data Movement
    participant CB as Circular Buffer
    participant Consumer as Compute

    Producer->>CB: reserve() - wait for free slot
    Note over Producer: Write data to block
    Producer->>CB: push() - mark as filled
    Consumer->>CB: wait() - wait for filled slot
    Note over Consumer: Read/process data
    Consumer->>CB: pop() - mark as free
```

## Example

```python
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

## API Reference

| Function | Description |
| :---- | :---- |
| `ttl.make_circular_buffer_like(ttnn.Tensor: likeness_tensor, shape: ttl.Shape, buffer_factor: ttl.Size) -> ttl.CircularBuffer` | Create a circular buffer by inheriting basic properties from `likeness_tensor`. |
| `ttl.CircularBuffer.reserve(self) -> ttl.Block` | Reserve and return a block from a circular buffer. **This function is blocking** and will wait until a free block is available. A free block is typically used by a producer to write the data into. |
| `ttl.CircularBuffer.push(self)` | Push a block to a circular buffer. This function is called by the producer to signal the consumer that a block filled with data is available. **This function is non-blocking.** |
| `ttl.CircularBuffer.wait(self) -> ttl.Block` | Wait for and return a block from a circular buffer. **This function is blocking** and will wait until a block filled with data is available. A filled block is typically used by a consumer to read data from. |
| `ttl.CircularBuffer.pop(self)` | Pop a block from a circular buffer. This function is called by the consumer to signal the producer that block is free and available. **This function is non-blocking.** |
