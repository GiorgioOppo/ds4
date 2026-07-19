**English** | [Italiano](README.it.md)

# Formats/GGUF

Validated, zero-copy reading of GGUF models.

## Main files

- [`GGUFTypes.swift`](GGUFTypes.swift): constants, value types, quantization
  information and format errors.
- [`GGUFCursor.swift`](GGUFCursor.swift): internal cursor for bounds-checked
  binary reads.
- [`GGUFModel.swift`](GGUFModel.swift): opens and maps the file, indexes
  metadata and tensor descriptors, exposes views and prefetch hints.

## Flow

Initialization verifies the header and tables, computes absolute offsets and
keeps a single mapping. Tokenizer and weight loaders query metadata and
tensors without copying the whole model; pages are materialized by the
system's page cache when accessed.

## Modification rules

Any arithmetic derived from the file must check for overflow and ranges. Do
not turn mmap views into implicit copies. New GGUF types require tests with
valid files, truncated files and unrecognized metadata.
