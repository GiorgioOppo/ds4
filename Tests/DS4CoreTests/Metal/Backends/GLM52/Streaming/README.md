# GLM 5.2 streaming tests

Byte-faithfulness of `GLM52PayloadReader` against synthetic pattern files: a
single-descriptor read must return exactly the descriptor's payload slice, and
an executed expert stream plan must land every planned range at its packed
gate|up|down record offset, identically on the concurrent and serial paths.

The rejection suite proves the typed errors fire before any byte moves: wrong
destination sizes, descriptors or planned ranges beyond the real end of file
(truncated download), hand-built plans that mix per-expert byte sizes, empty
plans and unopenable paths.

All fixtures are small block-legal geometries (Q4_K/Q6_K, 16 experts, ~2 MiB)
written to temp files — no Metal device and no real model download involved.
The end-to-end open+plan+read pass over a real sparse GGUF lives in
`../TensorSchema/GLM52RealHeaderIntegrationTests.swift` behind
`DS4_GLM52_SPARSE_GGUF`.
