[English](README.md) | **Italiano**

# Backend di architettura

Questa cartella contiene la documentazione che cambia in funzione della
famiglia di modello. Le regole comuni e la matrice dello stato sono in
[`../ARCHITETTURE-SUPPORTATE.md`](../ARCHITETTURE-SUPPORTATE.it.md).

Ogni sottocartella deve dichiarare:

- identificatori GGUF riconosciuti;
- profili realmente eseguibili;
- tokenizer e formato conversazionale;
- decoder, KV e capacità Metal;
- impostazioni applicabili e limiti della distribuzione;
- test necessari per considerare il backend supportato.

Un backend documentato come pianificato non deve comparire nella GUI come
operativo e non deve riutilizzare silenziosamente un decoder di un'altra
famiglia.

## Famiglie documentate

- [`deepseek-v4/`](deepseek-v4/README.it.md): backend locale operativo per Flash
  e Pro Q2 singolo; Pro Q4 split resta download-only.
- [`qwen/`](qwen/README.it.md): architettura riconosciuta, backend ancora in
  preparazione.
- [`glm-5.2/`](glm-5.2/README.it.md): manifest, contratto GGUF e port progressivo;
  detector/frontend disponibili, decoder ancora non eseguibile.
- [`laguna-s-2.1/`](laguna-s-2.1/README.it.md): frontend a stadi del branch di
  riferimento `laguna-s2.1` — geometria, tokenizer, protocollo chat/tool,
  schema tensori e catalogo; decoder in sospeso.
