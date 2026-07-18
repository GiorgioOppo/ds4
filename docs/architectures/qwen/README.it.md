# Backend Qwen — predisposizione

Qwen è al momento una famiglia **riconosciuta ma non eseguibile**. Questa
cartella documenta il confine preparato per aggiungere il backend senza
inserire condizioni Qwen nel decoder DeepSeek.

## Già predisposto

- identificazione della famiglia da `general.architecture`;
- errore dedicato quando il backend non è ancora implementato;
- cartelle separate nei livelli Core, Metal ed Engine;
- descrizione e capacità del modello consumabili da demo e GUI;
- separazione dei componenti DeepSeek prima dell'aggiunta di nuovi kernel.

## Non ancora implementato

- scelta della variante Qwen GGUF di riferimento;
- tokenizer, token speciali e template chat;
- mapping dei tensori e validazione della forma;
- prefill, decoder Metal e KV cache;
- reasoning e formato delle chiamate tool;
- preset UI e diagnostica numerica;
- checkpoint, inferenza distribuita e benchmark certificati.

Finché questi elementi non sono completati, caricare un GGUF Qwen deve
terminare prima della costruzione del decoder con un messaggio che indica che
il modello è stato riconosciuto ma il backend non è ancora disponibile.

## Primo passo implementativo

Prima di scrivere kernel occorre scegliere un singolo file GGUF piccolo e
rappresentativo, registrare il valore esatto di `general.architecture`, i
metadati del tokenizer e l'elenco dei tensori, quindi fissare golden test CPU.
Solo da quel contratto si ricavano forma, loader e grafi Metal.

