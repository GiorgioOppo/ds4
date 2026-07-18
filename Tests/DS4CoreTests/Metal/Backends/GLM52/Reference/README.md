# GLM 5.2 layer reference tests

Device-free suites for the layer/forward oracles: the first-token attention
shortcut equals the manual value-projection chain of the pinned primitives,
dense and sparse layers exhibit the pre-norm residual structure, the sparse
path routes with the router oracle on the internally computed logits and
fetches exactly the selected experts in rank order, and the multi-layer
forward equals sequential layer application. Rejection tests cover wrong
shapes. Compositional comparisons use small tolerances for summation-order
drift; routing equality is exact.
