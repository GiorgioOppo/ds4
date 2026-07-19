**English** | [Italiano](README.it.md)

# Model/Backends/Qwen

Extension point reserved for the future Qwen configuration.

At this stage it contains no shape types, validators or factories: the common
detector recognizes `qwen*` identifiers but reports them as
`recognizedButNotImplemented`. Support can be declared only after validating
metadata, variants and an actually present Metal backend.
