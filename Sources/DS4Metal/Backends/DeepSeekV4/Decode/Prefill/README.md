# DeepSeekV4/Decode/Prefill

Efficient prompt ingestion with layer-major processing and batching.

## Main files

- [`StreamingDecoder+Prefill.swift`](StreamingDecoder+Prefill.swift): splits the
  tokens into chunks, orchestrates the phases and chooses the FFN/matrix-matrix
  paths.
- [`PrefillStage.swift`](PrefillStage.swift): staging buffer for a group of tokens.
- [`PrefillGather.swift`](PrefillGather.swift): background expert gather with
  synchronized delivery to the caller.

## Flow

Each chunk is transformed into a contiguous staging area; for each layer the
routes and the expert union are computed, overlapping the next group's I/O
with the current GPU FFN. `DS4_PREFILL_CHUNK`, `DS4_PREFILL_ROUTE_BATCH`,
`DS4_PREFILL_FFN_BATCH` and `DS4_PREFILL_MM` control the variants documented
in the main configuration.

## Modification rules

The result must be equivalent to a sequence of single forwards. A started
worker must always be awaited, including on error paths. Limit temporary
memory to the chunk and measure token/s, peak RAM and SSD pressure
separately.
