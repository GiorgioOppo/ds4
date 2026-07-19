**English** | [Italiano](README.it.md)

# DeepSeek V4 backend

Registers the already operational backend: local generation, reasoning and
DSML tools, disk KV, expert routing/cache/bundle and family-specific tuning.
Flash keeps the verified distributed path; Pro distribution is still being
verified.

`DeepSeekV4BackendDefinition.locallyRunnableVariants` is the single
declaration of the locally runnable profiles: it includes Flash and Pro. Both
`BackendSelector` and the model catalog derive their gates from this list,
avoiding inconsistent states.

The factory keeps building the existing concrete `StreamingDecoder`. This
adapter exists to isolate selection and capabilities without adding dispatch
in the hot path. Flash and Pro receive a different `DSV4RuntimeGeometry`;
local Pro support covers the single Q2 GGUF, not the two-shard Q4 package.
