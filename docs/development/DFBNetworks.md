# DFB Networks

This document describes the TTL IR that declares networks of dataflow buffers
(DFBs): fork, split, and merge connections among DFBs on one node. It covers
the network operation, its records, the rules the verifier enforces, and the
pass that rejects networks at pipeline entry. Ordinary DFB behavior is
described in [DFBManagement.md](DFBManagement.md).

The design is tracked in issue
[#1057](https://github.com/tenstorrent/tt-lang/issues/1057). No pass lowers a
network, so any module that contains one fails compilation with a diagnostic.

## Overview

An ordinary DFB connects one producer kernel to one consumer kernel. A network
declares how several DFBs are connected, so that one item can reach several
consumers or items from several producers can reach one reader:

- A **fork** delivers every item of its source to each of its outputs.
- A **split** delivers each item of its source to exactly one of its outputs.
- A **merge** reads several inputs through one read handle.

Each connection is a record op inside a `ttl.dfb.network`. Records name DFBs
by the `dfb_id` of their `ttl.bind_cb` bindings, the logical identity that
kernel functions share. Records produce no values and execute nothing; they
describe connections between DFBs that kernels bind.

## Network IR

### `ttl.dfb.network`

```mlir
module {
  ttl.dfb.network @copy_and_exp {
    ttl.dfb.fork @fork 0 : index -> [1, 2] storage = <replicated>
  }
}
```

`ttl.dfb.network` is a symbol whose single block holds only records. It must
be a direct child of `builtin.module`, because a network connects DFBs that
several kernel functions bind and therefore cannot belong to any one of them.
It is isolated from above and is a symbol table, so record symbols resolve
only within one network.

### Records

| Record | Reads | Writes | Choice |
|---|---|---|---|
| `ttl.dfb.fork` | `source` | every DFB in `outputs` | `storage` |
| `ttl.dfb.split` | `source` | one DFB in `outputs` per item | `policy` |
| `ttl.dfb.merge` | every handle in `inputs` | the merged read handle | `policy` |

A **handle** is either a DFB id (`3 : index`) or a symbol reference to a
`ttl.dfb.merge` in the same network (`@m`). A merge creates no DFB: its result
is a merged read handle named by the merge's symbol. Fork and split `outputs`
are DFB ids, listed in output order. Each output's capacity comes from its
`ttl.bind_cb`.

```mlir
ttl.dfb.network @gather_then_distribute {
  ttl.dfb.merge @m [1 : index, 2 : index] policy = <first_ready>
  ttl.dfb.split @split @m -> [3, 4] policy = <round_robin>
}
```

The record attributes are:

- **Fork `storage`** (`#ttl.dfb_network_fork_storage`). `replicated` gives
  each output its own payload storage and writes every item once per output.
  `shared` keeps one payload ring that every output reads; a slot is reused
  only after every output releases it. `auto` lets the compiler choose shared
  storage when it is legal and replicated storage otherwise.
- **Split `policy`** (`#ttl.dfb_network_split_policy`). `round_robin` sends
  item i to output i mod N. `contiguous` sends each output one consecutive run
  of items, in output order.
- **Merge `policy`** (`#ttl.dfb_network_merge_policy`). `round_robin` visits
  the inputs in a fixed rotation and skips exhausted inputs. `first_ready`
  takes an item from the first ready input found by a rotating scan, so the
  order depends on timing.

## Verification

Each record checks its own operands:

- A record appears only directly inside a `ttl.dfb.network`.
- DFB ids are non-negative.
- A fork or split has at least one output, lists no output twice, and does not
  list its source DFB id as an output.
- A merge has at least one input and lists no input twice.

The network checks how its records connect:

- The body contains only fork, split, and merge records.
- Every symbol handle names a `ttl.dfb.merge` in the same network.
- Each handle is read by at most one record, so every connection keeps a
  single reader.
- Each DFB is an output of at most one record, so every connection keeps a
  single writer.
- Records form no cycle. Because each handle has at most one reader, a
  depth-first walk from each record along the readers of its outputs reaches
  everything downstream of it; reaching a record still on the current path
  reports the handle that closes the cycle.

```mlir
ttl.dfb.network @cycle {
  ttl.dfb.split @a 0 : index -> [1, 2] policy = <round_robin>
  // error: 'ttl.dfb.split' op handle 0 : index forms a cycle
  ttl.dfb.split @b 1 : index -> [0, 3] policy = <round_robin>
}
```

Checks that compare records with the kernel functions of the module, such as
whether each DFB id has a `ttl.bind_cb` or whether a handle belongs to two
networks, are not part of the verifier.

## Rejection at pipeline entry

`ttl-reject-dfb-networks` is a module pass that emits an error for each
`ttl.dfb.network` in the module, naming the network, and fails the pass:

```
error: 'ttl.dfb.network' op @copy_and_exp: DFB networks are not supported yet
```

It runs first in `ttl-to-ttkernel-pipeline` and in the pass list that
`python/ttl/ttl_api.py` builds. Network records declare DFB producers and
consumers that no kernel op shows. Passes that infer DFB endpoints from kernel
ops, such as `ttl-insert-cb-sync` and `ttl-verify-dfb-spsc`, would see an
incomplete program and either report errors against the kernels or accept a
program whose connections are ignored. The pass therefore runs before every
function pass, and before any pass that reasons about DFB producers or
consumers.
