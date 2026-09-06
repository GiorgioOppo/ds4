[English](DEEPSEEK-VISION.md) | **Italiano**

# DeepSeek Vision Experimental nella chat locale

La chat SwiftUI integra il checkpoint **DeepSeek V4 Flash Vision-Exp** con il
suo encoder separato. È un percorso sperimentale nativo Swift/Metal: non usa
un server esterno e non invia le immagini in rete. Il normale Flash 0731
ha pesi diversi e non acquisisce Vision aggiungendo soltanto l'encoder.

## Configurazione

1. Apri **Modelli e impostazioni → Scarica…** e scegli il filtro **Vision**.
2. Scarica il modello Vision IQ2XXS (86,72 GB) oppure il misto IQ2XXS/Q4_K
   (97,59 GB), più **DeepSeek V4 Flash Vision · encoder immagini** (932,86 MB).
   Puoi anche selezionare file già presenti con i rispettivi pulsanti di scelta.
3. Seleziona il modello principale; nella sezione **DeepSeek Vision** seleziona
   l'encoder, oppure premi **Usa encoder** nella sua riga del catalogo.
4. Carica il modello. Nella chat, **Allega** permette di scegliere immagini
   insieme ai documenti di testo. Trascinare file nel composer funziona allo
   stesso modo. È possibile inviare anche un'immagine senza testo.

L'app ammette fino a quattro immagini per messaggio, ciascuna entro 20 MiB e
40 megapixel. L'encoder accetta immagini fino a 16.384 pixel per lato. PNG e
JPEG sono formati consigliati; la decodifica usa ImageIO e applica l'orientamento
EXIF. Ogni immagine occupa al massimo 384 token del contesto.

Il [catalogo originale](https://huggingface.co/antirez/deepseek-v4-gguf/tree/main)
è fissato al commit `f71f23d552d664e523b422157b2befbf74040380`, con byte count
e SHA-256 verificabili per ogni file. I pesi non vengono scaricati al build.
Il checkpoint MXFP4 e il draft DSpark Vision sono disponibili solo per il
download: queste modalità non sono attive nel runtime Vision Swift.

## Comportamento e limiti

- Anteprime e nomi file restano visibili; contenuti originali di immagini e
  documenti vengono salvati nella chat JSON in Application Support/DwarfStar/chats.
  Riaprendo la chat, il modello riceve di nuovo il contesto degli allegati.
- Le conversazioni precedenti, prive dei nuovi campi, restano leggibili. Il
  contenuto dei vecchi allegati testuali non salvato dalle versioni precedenti
  non può essere recuperato automaticamente.
- Le immagini richiedono il modello Vision anche dopo un cambio modello.
  Il programma segnala il problema prima di consumare la bozza del messaggio.
- Il KV in memoria viene riutilizzato. I checkpoint KV su disco sono esclusi
  dalle chat con immagini, perché la loro chiave attuale contiene soltanto token.
- Il supporto immagini riguarda la chat locale e `InferenceService.sendWithImages`.
  I parser HTTP e il protocollo distribuito non acquisiscono immagini. L'auto-tune
  che ricarica il motore e la decodifica speculativa DSpark sono disabilitati su Vision.
- Il prefill delle immagini privilegia la correttezza dei blocchi bidirezionali;
  è meno ottimizzato del prefill testuale in batch.

## Implementazione e verifica

L'encoder usa preprocessing bicubico, patch 14×14, 32 blocchi transformer BF16,
RoPE 2D e aligner 3×3. Il decoder applica l'epsilon del checkpoint (`1e-20`),
inietta gli embedding visuali, usa `bias_vl` per il router ed esegue l'attenzione
bidirezionale delle immagini mantenendo causali i token compressi. Vedi il
[README dell'encoder](../Sources/DS4Metal/Backends/DeepSeekV4/Vision/README.it.md).

Verifica del 6 settembre 2026: build SwiftPM riuscita; 25 controlli autonomi
superati (10 encoder, 8 decoder/catalogo, 7 persistenza/allegati). I test GPU
includono BF16, MPS, attenzione, aligner e parità del prefill con pesi sintetici
residenti o caricati per esperto, con logits entro `1e-4`. Il preprocessing della
fixture RGB 17×9 è identico al C upstream su 446.880 float. Sono presenti test
XCTest corrispondenti, ma su questa macchina
XCTest manca nei Command Line Tools, quindi `swift test` non è stato eseguito.
La parità numerica e la qualità end-to-end con encoder e modello completi non
sono state validate: non sono stati scaricati pesi da decine di gigabyte.

Per ripetere la suite su un Mac con Xcode completo:

```sh
swift test --filter 'DeepSeekV4Vision|DeepSeekV4ConfigurationTests|ModelDownloaderTests|ChatPersistenceTests|ChatAttachmentTests|PrefillBatchAttnParityTests'
```
