# Flusso di esecuzione della chat

Questo documento descrive i confini di ownership della funzionalità dell'app
più ricca di stato.

## Nuovo messaggio

1. `ChatView` invia il testo del composer e gli allegati selezionati a
   `ChatStore`.
2. `ChatStore+Generation` costruisce la richiesta e invoca l'`InferenceService`
   condiviso.
3. Gli eventi dell'engine vengono convertiti in stato `UIMessage` e
   renderizzati immediatamente.
4. Le chiamate ai tool vengono validate e smistate da `ChatStore+ToolLoop`; il
   lavoro dei sub-agent è coordinato da `ChatStore+Agents`.
5. La trascrizione completata viene mappata in `StoredMessage` e salvata da
   `ChatSessionStore`.

Lo stream e ogni continuazione portano con sé il `conversationEpoch` del turno.
Stop, New Chat o un cambio di sessione lo invalidano prima di modificare la
trascrizione, così un token o un risultato di tool in ritardo non può eseguire
un altro tool, scrivere attraverso un indice di messaggio riutilizzato o
azzerare lo stato di lavori più recenti. I batch misti di tool mantengono
inoltre uno slot di output ordinato per ciascuna chiamata mentre attendono
eventuali risultati manuali.

## Sessione riaperta

L'apertura di una chat ripristina la cronologia della UI, non lo stato KV in
memoria del modello. Al successivo invio, `sendWithHistory` renderizza di nuovo
la cronologia visibile. Un prefisso KV su disco compatibile può accelerare
quella ricostruzione; i turni successivi tornano alla generazione incrementale.

## Invariante dell'engine condiviso

Chat, Benchmark locale e Server condividono un unico `InferenceService`. La
generazione della Chat e le richieste HTTP sono serializzate, mentre Benchmark
può essere eseguito solo quando la Chat è inattiva perché riscrive lo stato KV.
Le modifiche devono preservare questa regola di ownership a engine singolo per
evitare allocazioni duplicate del modello da diversi gigabyte.
