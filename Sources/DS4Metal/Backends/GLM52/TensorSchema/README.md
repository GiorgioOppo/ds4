# GLM 5.2 tensor schema

`GLM52TensorSchema` is the fail-fast directory validator for the exact GLM 5.2
GGUF layout.  It validates names, dimensions, dense/control precision, routed
expert quantization and the final nextn block without reading weight payloads.

`GLM52WeightMap` turns that validated directory into typed global and per-layer
lookups. Its descriptors retain only tensor metadata and absolute mmap offsets:
they neither allocate nor copy GGUF payload bytes.

`GLM52ExpertStreamPlanner` converts the router's eight unique expert IDs into
24 bounded reads (gate, up and down for each expert). Expert slices are the
contiguous third-dimension ranges in the GGUF tensor. Row and slice sizes come
from each quantization's block geometry, and every multiplication, file-offset
addition and aggregate byte count is overflow checked. Router rank order is
preserved and adjacent ranges are intentionally not coalesced across experts.

Supported routed gate/up types are IQ2_XXS, Q2_K, Q4_K and Q5_K and must match
within a layer.  Routed down also accepts Q6_K.  This describes the reference
graph contract; kernel availability remains a separate backend capability.
