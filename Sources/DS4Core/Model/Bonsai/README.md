# Bonsai metadata contract

Bonsai uses `general.architecture=qwen35` plus Prism's explicit Hadamard
metadata. `BonsaiConfiguration` validates the supported 64-layer 27B checkpoint
and exact matrix/vector layouts before the Metal backend receives any bytes.
PQ2 and PTQ decoders and the signed normalized Hadamard reference are in
`Quantization/Bonsai`. PTQ's multiplication wraps modulo 256 before extracting
the trit; ordinary base-three decoding is incorrect.
