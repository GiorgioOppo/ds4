**English** | [Italiano](README.it.md)

# GLM 5.2 model contract

This directory owns the portable GLM 5.2 model geometry and GGUF metadata
validation.  It has no Metal dependency and does not register or select a
runtime backend.

`GLM52Configuration` accepts only `general.architecture = "glm-dsa"` and the
exact 79-block GLM 5.2 shape implemented by the reference `glm5.2` branch.  The
79th block contains the stored next-token prediction tensors; normal inference
uses 78 transformer blocks.  Unknown GLM variants fail at load instead of being
run with incompatible dimensions.
