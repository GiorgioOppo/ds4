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
- [`GGUFWriter.swift`](GGUFWriter.swift): the reader's inverse — serializes a
  GGUF v3 file from ordered typed metadata (`GGUFMetadataValue`) and tensors,
  streaming payloads one at a time so large models need not fit in memory.
- [`GGUFModel+Export.swift`](GGUFModel+Export.swift): read-back helpers
  (`allMetadata`, `tensorData`) that feed a loaded model into the writer,
  closing the read -> edit -> write round-trip.
- [`GGUFShardSet.swift`](GGUFShardSet.swift): presents several GGUF shards as
  one logical model by unioning their tensor directories by name (the DeepSeek
  V4 Pro Q4 layer-range split). Tensor names must be disjoint; global metadata
  is first-shard-wins.

## Flow

Initialization verifies the header and tables, computes absolute offsets and
keeps a single mapping. Tokenizer and weight loaders query metadata and
tensors without copying the whole model; pages are materialized by the
system's page cache when accessed.

## Modification rules

Any arithmetic derived from the file must check for overflow and ranges. Do
not turn mmap views into implicit copies. New GGUF types require tests with
valid files, truncated files and unrecognized metadata.
