# All-gather over device relations

This document proposes how an all-gather is expressed and lowered. The example
[`examples/multidevice_ring_all_gather.py`](https://github.com/tenstorrent/tt-lang/blob/main/examples/multidevice_ring_all_gather.py)
implements the relay lowering described below by hand with existing
constructs.

## Problem

An operation whose input is sharded across a device domain often needs the
complete tensor on every device, for example the K-sharded activation of an
all-gather matmul. A TT-Lang program can express that transfer, but the
program fixes its schedule: when forwarding runs on threads that also feed
compute, every device waits on its neighbor's compute progress once per
forwarding step. In a four-device all-gather matmul with M, K, and N of 4096,
6144, and 18432, device-profiler zones show that compute consumes the complete
gathered K dimension in one N round of about 446 us, while the first N round,
which waits on the gather, takes about 666 us.

## Proposed specification text

This extends the graph pipe net model of
[Pipes on Fabric](https://github.com/tenstorrent/tt-lang/blob/main/docs/development/PipesOnFabric.md),
which defines `DeviceDomain`, `TransferGraph`, and graph pipe nets, and the
[computed DRAM tensor destinations](https://github.com/tenstorrent/tt-lang/blob/main/docs/development/PipeNets.md#computed-dram-tensor-destinations)
of pipe receives. It adds no operation or class.

The remote-shard transfer of an all-gather is a graph pipe net over the
all-to-all relation whose receives write DRAM tensor slices:

```python
device_domain = ttl.DeviceDomain((D, 1))
gather_net = ttl.PipeNet(
    graph=ttl.TransferGraph.all_to_all(device_domain),
    pipes=[ttl.Pipe(src=(0, 0), dst=(0, 0))],
)

@ttl.operation(grid=(GRID_X, GRID_Y), device_domain=device_domain)
def gather_shards(shard, gathered):
    chunk_dfb = ttl.make_dataflow_buffer_like(shard, shape=(CM, CK), block_count=2)

    @ttl.datamovement()
    def gather():
        node_x, node_y = ttl.node(dims=2)
        if node_x == 0 and node_y == 0:
            for m in range(0, M, CM):
                for k in range(0, K, CK):
                    with chunk_dfb.reserve() as chunk_blk:
                        ttl.copy(shard[m : m + CM, k : k + CK], chunk_blk).wait()
                    with chunk_dfb.wait() as chunk_blk:

                        def send_chunk(pipe):
                            ttl.copy(chunk_blk, pipe).wait()

                        gather_net.if_src(send_chunk)

                    def receive_chunk(pipe):
                        col = pipe.source_device_index * K + k
                        region = gathered[m : m + CM, col : col + CK]
                        ttl.copy(pipe, region, shape=(CM, CK)).wait()

                    gather_net.if_dst(receive_chunk)
```

Slot `pipe.source_device_index` of `gathered` receives that device's shard.
The all-to-all relation has no self edges, so the transfer does not write the
local slot; an all-gather that needs the complete tensor also copies the local
shard into its own slot.
Node `(0, 0)` is the only endpoint of `gather_net`; every other launched node
may serve as a forwarder.

1. Transport selection. For a graph pipe net whose receives write DRAM tensor
   slices that are disjoint for the invocation, the compiler may deliver a
   payload through intermediate devices that are destinations of the same
   payload (relay), or with fabric multicast when the destination slice
   resolves to the same address on every destination. Every destination slice
   receives the payload exactly once, and the send and receive completion
   semantics of `ttl.copy` are unchanged. The relation, not the program, fixes
   which devices receive which data.
2. Forwarder nodes. Relay executes on forwarder nodes: launched nodes that are
   not endpoints of any pipe net in the operation. An operation author
   provides forwarders by launching more nodes than the pipe nets' node
   relations name. Without forwarder nodes, the compiler delivers every
   payload by direct unicast.
3. Visibility of received slices. Another data movement thread, on any node of
   the destination device, may read a slice written by a receive after the
   receiving thread's `wait()` on that receive is followed by an `inc` or `set`
   of the reader's `ttl.Semaphore`, obtained with `get_remote` when the reader
   is on another node, and the reader observes that value with `wait_ge` or
   `wait_eq`. Without that ordering, only the receiving thread may read the
   slice.

Rules 1 and 2 add intermediate TENSIX programs to fabric transport, which the
[destination-routed transport](https://github.com/tenstorrent/tt-lang/blob/main/docs/development/PipesOnFabric.md#destination-routed-transport-behavior)
of Pipes on Fabric excludes. Rule 3 lets a consumer on another node wait per
chunk instead of per forwarding step, so the gather runs ahead of compute by
the full destination capacity. It relies on `ttl.Semaphore`, which the
specification defines but neither the compiler nor the simulator implements.

## Roofline

Four Blackhole devices, each owning a 4096 x 1536 BF16 shard (12.6 MB) and
receiving 37.7 MB. Device time is the interval from the first kernel start to
the last kernel end on the slowest device, median of 10 samples after 3
warmups.

Lower bound on link load. Each device receives D - 1 shards over two incoming
link directions, so some direction carries at least (D - 1) / 2 shards: 1.5
shards (18.9 MB) for D = 4. Splitting every shard into halves, sending right
halves forward and left halves backward, attains the bound. On a line, the end
devices receive all D - 1 shards over one link.

TT-Metal reference. For this volume, `ttnn.all_gather` selects its unicast
algorithm under both fabric configurations: every send is a one-hop unicast to
the neighbor, and each device writes a received stripe into its output tensor
in DRAM, reads it back, and forwards it. This is the relay strategy below. The
fabric configuration sets the topology: under `FABRIC_1D_RING` the axis is a
ring whose antipode stripe is split between the two directions, while under
`FABRIC_2D` only torus configurations wrap, so the axis is a line.

| Fabric | TT-Metal topology | Device time | Receive bandwidth per device | Bandwidth per incoming link |
| --- | --- | ---: | ---: | ---: |
| 1D ring | ring | 332 us | 114 GB/s | 57 GB/s |
| 2D | line | 494 us | 76 GB/s | 76 GB/s |

The line carries twice the ring's link load at a higher per-link rate, so
these rows do not show that the 2D fabric itself is slower.
`ttnn.all_gather` also copies the local shard into its output, which the
TT-Lang example does not, so the paired ratios below favor TT-Lang slightly.

TT-Metal's relay differs from the example as follows; no difference has been
measured individually:

- 8 to 12 worker cores per direction per link, chosen from the bytes per link,
  share one fabric mux; the example runs 8 or 4 lanes per direction.
- A reader RISC reads chunks back from DRAM into a multi-page CB while a
  writer RISC sends them. Each example lane serializes receive-wait,
  read-back, and send per chunk on one thread.
- The writer packs several chunks into one scatter packet and signals relay
  progress to the next device with one semaphore increment per batch; the
  example sends and completes each chunk as its own pipe transfer.
- The writer routes its packet headers once at setup; the example's sender
  sets the route on every packet. Both wait for each packet's write to flush
  before sending the next.

For the all-gather matmul in [Problem](#problem), a gather at TT-Metal's 1D
ring rate would take less than one N round of compute.

## Lowering strategies

| Strategy | Link load per direction (D = 4) | Per-hop work | Available |
| --- | --- | --- | --- |
| Direct unicast to each destination | 2 shards | none | yes; route binding selects the direction of a two-hop destination |
| Relay on forwarder nodes | 1.5 shards | DRAM write and read-back per hop | yes; see the relay example |
| Fabric line multicast | 1.5 shards | none | TT-Metal supports it; TT-Lang does not lower device-range destinations |

For large volumes, relay is the target: TT-Metal's all-gather on Blackhole
selects relay once 1 to 4 MB, depending on page size, cross a link, and
multicast only below that. Relay is also the strategy available with the
current compiler.

Relay constraints:

- A forwarder node sends in one ring direction only. A manager with
  connections in two directions is not eligible for the program-local fabric
  mux. A forwarder also cannot choose the direction of a two-hop send, because
  route binding selects it, so relay uses one-hop sends only.
- A forwarder's relayed and local sends use separate pipe nets (seed and
  relay), so each net has one static send and one static receive post per
  forwarding step.
- Fabric receives write disjoint DRAM destination slices, which the `CDA/NR`
  protocol delivers without receiver rendezvous, so a sender never waits for
  the receiving thread.
- The forwarder reads a received slice back through the same slice value that
  the receive wrote. Schedule verification and pipe lowering must enumerate
  every DRAM region a receive writes at a launch location; a slice start that
  depends on the incoming pipe, such as `pipe.source_device_index`, cannot be
  enumerated there. The example therefore indexes destination slots by source
  distance rather than by source device.

## Relay example

`examples/multidevice_ring_all_gather.py` implements the relay strategy by
hand. Measured with `python -m benchmarks.all_gather` on the same devices and
input as the roofline, with 8 x 12-tile chunks. TT-Metal and TT-Lang alternated
in each of two rounds; outputs were checked bit-exactly after the warmup runs
and after the last measured run. Each cell lists both rounds, and each ratio
is to TT-Metal in the same round.

| Fabric | TT-Metal `ttnn.all_gather` | TT-Lang relay, 8 lanes per direction | TT-Lang relay, 4 lanes per direction |
| --- | ---: | ---: | ---: |
| 1D ring | 333, 332 us | 484, 489 us (1.45-1.48x) | 511, 518 us (1.53-1.56x) |
| 2D | 493, 494 us (line) | 522, 526 us (1.06x) | 613, 613 us (1.24x) |

The 1D ring row compares the same topology and algorithm. On 2D, TT-Metal
gathers over a line while the example gathers over the ring, so that row does
not compare like with like. The TT-Lang 1D ring cells require host route
binding to use the ring's closing link, a change outside this proposal.

Limitations of the example relative to the proposal:

- Destination slots are indexed by source distance relative to the receiving
  device, and the local shard is not copied into the destination.
- The ring order and forwarder nodes are written by hand; the relation form
  above is not yet lowered this way.
- Consumers cannot wait per chunk from other nodes (rule 3).

## Compiler work

1. Lower a graph pipe net over the all-to-all relation with tensor-slice
   receives to relay transport, choosing forwarder nodes that are not pipe
   net endpoints and one ring direction per forwarder.
2. Accept destination slice starts that depend on the incoming pipe, such as
   `pipe.source_device_index`, in DRAM region enumeration.
3. Implement `ttl.Semaphore` and the visibility rule for received slices.
4. Measure each difference from TT-Metal's relay listed under
   [Roofline](#roofline) against the paired table.
5. Lower fabric multicast for device-range destinations, listed under
   [remaining capability work](https://github.com/tenstorrent/tt-lang/blob/main/docs/development/PipesOnFabric.md#remaining-capability-work),
   for small volumes.
