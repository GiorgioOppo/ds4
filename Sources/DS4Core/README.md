# DS4Core

Fondazione portabile del progetto, scritta in Swift e priva di dipendenze da
Metal. Espone formati, modelli conversazionali, tokenizzazione, campionamento e
tipi condivisi usati da `DS4Metal` e `DS4Engine`.

## Struttura

- [`Conversation/`](Conversation/README.md): tipi comuni e formati chat per backend.
- [`Diagnostics/`](Diagnostics/README.md): avanzamento thread-safe del caricamento.
- [`Formats/`](Formats/README.md): GGUF, checkpoint KV e primitive di quantizzazione.
- [`Generation/`](Generation/README.md): selezione del token successivo.
- [`Model/`](Model/README.md): rilevamento architettura e configurazioni per backend.
- [`Storage/`](Storage/README.md): pianificazione della cache SSD e simulazione RAM.
- [`Tokenization/`](Tokenization/README.md): API comune e tokenizer per backend.

## Dipendenze e flusso

`DS4Core` dipende unicamente dalla libreria standard e da Foundation. Il flusso
tipico è: apertura del GGUF -> costruzione del tokenizer -> rendering della
conversazione -> tokenizzazione -> campionamento dei logits prodotti dal backend.
Le operazioni GPU e l'esecuzione del decoder appartengono a `DS4Metal`.

## Regole di modifica

- Non introdurre import o tipi Metal in questo target.
- Mantenere deterministici parser, rendering e sampler, con test di parità.
- Conservare la compatibilità binaria dei formati persistenti.
- Collocare ogni nuova responsabilità nella sottocartella di dominio e aggiornare
  il relativo README.
