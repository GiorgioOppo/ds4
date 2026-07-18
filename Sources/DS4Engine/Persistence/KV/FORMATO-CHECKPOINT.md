# KV checkpoint format

The file combines portable `DS4Core` headers and an explicit Swift body:

```text
KVC header (48 bytes)
u32 modelNameLength + UTF-8 modelName
u32 tokenCount + tokenCount × u32 tokenId
DSV4 payload header (52 bytes)
per layer:
  u32 rawStart
  u32 rawFloatCount + raw Float32
  u8 hasCompressor
  if present:
    u32 stateCount
    u32 stateLength + stateKV Float32
    stateScore Float32
    u32 cacheFloatCount + cache Float32
```

All integers are little-endian. The header identifies quantization, context,
token count, hits and timestamp; the model name and the full sequence prevent
reuse across different models or prefixes.

## Reading

The reader validates header, counts and token frontier, then imports one
batch of layers at a time and frees the intermediate buffers. Truncated data,
inconsistent counts or a different model invalidate the entire entry.

## Writing

The snapshot is moved into a single-ownership container; each layer is
written and released. The file becomes visible as an entry only once the
write completes. Subsequent restores update in place only the header
metadata.

## Compatibility

Do not add fields without updating version/header and test fixtures. A new
reader must not attempt to heuristically interpret an incompatible body.
