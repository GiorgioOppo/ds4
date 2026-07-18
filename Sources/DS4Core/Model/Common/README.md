# Model/Common

Portable contracts for identifying an architecture without accidentally
selecting the backend of another family.

- `ModelArchitecture.swift` defines the canonical identifier, the family,
  backend availability, the minimum capabilities and the GGUF detector.

The detector recognizes the Qwen family but explicitly reports it as having
no backend at this stage. The fallback on the `deepseek4.*` keys is used
only for old GGUFs lacking `general.architecture`; an explicit value always
remains authoritative.
