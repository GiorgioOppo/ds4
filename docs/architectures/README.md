**English** | [Italiano](README.it.md)

# Architecture backends

This folder contains the documentation that varies by model family. The
common rules and the status matrix are in
[`../ARCHITETTURE-SUPPORTATE.md`](../ARCHITETTURE-SUPPORTATE.md).

Each subfolder must declare:

- recognized GGUF identifiers;
- profiles that are actually runnable;
- tokenizer and conversation format;
- decoder, KV and Metal capabilities;
- applicable settings and distribution limits;
- tests required to consider the backend supported.

A backend documented as planned must not appear in the GUI as operational and
must not silently reuse another family's decoder.

## Documented families

- [`deepseek-v4/`](deepseek-v4/README.md): operational local backend for Flash
  and single-file Pro Q2; Pro Q4 split remains download-only.
- [`qwen/`](qwen/README.md): recognized architecture, backend still in
  preparation.
- [`glm-5.2/`](glm-5.2/README.md): manifest, GGUF contract and progressive
  port; detector/frontend available, decoder not yet runnable.
