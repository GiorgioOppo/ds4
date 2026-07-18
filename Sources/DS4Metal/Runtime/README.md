# Runtime

The Metal foundation on which loader, graph and decoder rest.

## Structure

- [`Core/`](Core/README.md): device, library, pipelines and `GPUTensor`.
- [`Generated/`](Generated/README.md): embedded and generated kernel sources.

## Flow and dependencies

`MetalRuntime` creates the device and command queue, concatenates the sources
in canonical order and compiles an `MTLLibrary`. The wrappers request
pipelines by name and operate on `GPUTensor`s in unified memory. In
production no `metal/` folder next to the app is needed.

## Modification rules

Keep this layer independent of DeepSeek semantics where possible. Centralize
pipeline creation and caching, propagate descriptive errors, and do not
introduce implicit synchronization in the tensor containers.
