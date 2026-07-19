**English** | [Italiano](README.it.md)

# Metal Runtime Tests

`MetalRuntimeTests.swift` checks device discovery, embedded kernel-library
compilation, and pipeline creation.

Skip with an explicit reason if the host has no Metal device. Failure to compile
embedded kernels on a Metal-capable host is a test failure, not a skip.

