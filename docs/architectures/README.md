# Backend di architettura

Questa cartella contiene la documentazione che cambia in funzione della
famiglia di modello. Le regole comuni e la matrice dello stato sono in
[`../ARCHITETTURE-SUPPORTATE.md`](../ARCHITETTURE-SUPPORTATE.md).

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

- [`deepseek-v4/`](deepseek-v4/README.md): backend locale operativo per Flash
  e Pro Q2 singolo; Pro Q4 split resta download-only.
- [`qwen/`](qwen/README.md): architettura riconosciuta, backend ancora in
  preparazione.
- [`glm-5.2/`](glm-5.2/README.md): manifest dei tre GGUF acquisibili e roadmap;
  stato corrente download-only, senza detector o runtime GLM.
