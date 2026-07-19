**English** | [Italiano](README.it.md)

# DeepSeek V4 kernels

Metal kernel sources for the DeepSeek V4 graph plus the shared generic ops
(normalization, softmax, argsort, copy/cast, unary, reduction glue) that the
DeepSeek decoder drives today. Kernel names are globally unique: Metal compiles
one library from every vendored file, so a future backend can reuse the generic
ops from here without duplicating them.

Editing workflow, embedding (`make embed-kernels`) and the audit status live in
[`../README.md`](../README.md); the runtime/wrapper policy is in
[`docs/BACKEND-METAL.md`](../../docs/BACKEND-METAL.md).
