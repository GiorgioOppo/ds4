**English** | [Italiano](README.it.md)

# DeepSeekV4/Decode/Diagnostics

Aggregate measurements of the inference hot path.

## Main files

- [`DecodeProfile.swift`](DecodeProfile.swift): accumulates embedding,
  route/attention, expert gather, FFN and output head times; records
  hits/misses and I/O bytes and produces a per-token report.

## Flow and dependencies

The decoder updates the profile at command buffer and I/O operation
boundaries. With `DS4_PROFILE_ROUTE=1` it further breaks down the routing
phase; the service reads the report at the end of the turn. It is not
persistent global telemetry.

## Change rules

State whether a measurement includes GPU waits or only CPU encoding. Avoid
extra synchronizations in the normal profile; diagnostic counters must not
change the numerical results and must be interpreted over the same number of
forwards.
