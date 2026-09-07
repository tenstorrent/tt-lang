# Block Shape Views

`ttl.block.squeeze` and `ttl.block.unsqueeze` remove or insert size-one block
grid dimensions. The frontend emits `tensor.collapse_shape` and
`tensor.expand_shape`, respectively, with explicit reassociation indices.
An empty dimension list returns the original value. The public dimension
normalization and diagnostic rules remain unchanged.

## Storage identity

Tensor reshape operations support more shape changes than block views. The
shared `getSingletonDimensionShapeViewSource` helper accepts a reshape for DFB
provenance only when both ranked tensor shapes are static, their element types
and encodings are identical, and removing all size-one dimensions from each
shape produces the same ordered sequence. Equal element counts alone are not
sufficient.

Inserting or removing a size-one dimension does not change the row-major
linear index of any tile. The predicate therefore preserves tile order,
element representation, and the acquired DFB slot without allocation or data
movement. MLIR's expand/collapse verifier additionally checks the reassociation
between source and result dimensions. Rank-zero views are accepted only when
the other shape consists entirely of size-one dimensions.

`traceDFBShapeViews` composes checked shape views with identity casts and
one-to-one CB conversion bridges with matching element types. Arbitrary
tensor-to-tensor reinterpretation casts do not establish storage identity.
DFB association, acquisition, lifetime, release placement, accumulation
initialization, and waited-buffer mutation analysis use this shared shape-view
contract. Unsupported
expand/collapse operations terminate provenance tracing before the generic
`ViewLikeOpInterface` fallback; that interface alone does not prove the block
view contract.

## Lowering and validation

TTL-to-TTKernel conversion diagnoses unsupported tile or DFB-backed shape
views before rewriting. Accepted views retain the original buffer identity
while consumers use their logical result shape to calculate tile indices.
Nonzero subblock offsets are composed through checked views and nested slices
so the same linear tile index still reaches the correct root-buffer location.
Dead view chains are removed after consumers lower to buffer and tile
operations.

Compiler tests cover view chains, leading and interleaved singleton axes,
association and acquisition tracing, and unsupported general reshapes.
Frontend tests cover dimension normalization, rank-zero cases, static-shape
requirements, element type, and encoding preservation. Device tests check
value and tile-order preservation.
