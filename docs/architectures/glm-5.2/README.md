# GLM 5.2

Questa cartella documenta l'introduzione progressiva di GLM 5.2 in DwarfStar.
Lo stato corrente è **catalogo e download soltanto**: la GUI può acquisire i
GGUF pubblicati in `antirez/glm-5.2-gguf`, ma non può selezionarli, caricarli o
usarli per demo, chat, server, benchmark o inferenza distribuita.

## Stato della tranche

| Capacità | Stato |
|---|---|
| Identificatore osservato nel GGUF | `general.architecture = glm-dsa` |
| Catalogo e download GUI | sì |
| Resume, preflight disco e SHA-256 | sì |
| Riuso di un finale della dimensione esatta | sì |
| Riconoscimento nel selettore backend | no |
| Tokenizer e template conversazionale | no |
| Decoder CPU/Metal, prefill e decode | no |
| Selezione GUI o `DS4Demo` | no |
| Server, benchmark e distribuzione | no |

La presenza nel `ModelCatalogRegistry` significa soltanto che l'artefatto è
acquisibile. Tutte le entry GLM 5.2 sono `downloadOnly`; un file completato non
diventa automaticamente il modello attivo.

## Manifest GGUF

Snapshot verificato il 17 luglio 2026 sul repository
[`antirez/glm-5.2-gguf`](https://huggingface.co/antirez/glm-5.2-gguf), revisione
`2638b3b878f5c6cc3ae7334b8dbea1275025f52e`.

| Variante | Filename | Dimensione esatta | SHA-256 |
|---|---|---:|---|
| IQ2_XXS RoutedIQ | `GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf` | 211.075.856.448 byte | `a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0` |
| Q2_K RoutedQ2K | `GLM-5.2-UD-Q2_K_RoutedQ2K.gguf` | 262.036.650.048 byte | `b9fa49d010dad35b96418c45831c212a746715b0646c1121ccfc414455bd6fe5` |
| Q4_K RoutedQ4K | `GLM-5.2-UD-Q4_K_RoutedQ4K.gguf` | 434.170.886.208 byte | `7160879c87756236eea16ec6bfeb19288d16fa94dcfcef3a5ed5f38b1383d3a5` |

I tre file sono alternative monolitiche, non shard di un unico package. Il
downloader ne acquisisce uno per volta, scrive direttamente su `.part`, limita
i buffer e la verifica SHA a blocchi di 8 MiB e non conserva il GGUF in RAM.

## Informazioni upstream

La model card ufficiale [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2)
descrive un modello MoE per inglese e cinese con contesto pubblicizzato di
1.048.576 token, DSA/IndexShare e MTP. La configurazione upstream indica, fra
le altre cose, 78 layer, hidden size 6.144, vocabolario 154.880, 256 esperti
routed, top-8 per token, un esperto condiviso e tre layer densi.

Questi valori sono informazioni di pianificazione, non costanti del runtime
DwarfStar. Prima di usarli il backend deve confrontarli con i metadati e i
tensori del GGUF catalogato. Anche il contesto da un milione di token non è
ancora supportato o validato dall'app.

Il repository quantizzato di antirez non contiene una propria model card o un
file di licenza. L'upstream `zai-org/GLM-5.2` dichiara licenza
[`MIT`](https://huggingface.co/zai-org/GLM-5.2/blob/main/LICENSE); chi usa gli
artefatti quantizzati deve verificare anche le condizioni della pubblicazione
che sta scaricando.

## Lavoro necessario per il runtime

1. Registrare `glm-dsa` nel detector come architettura nota ma non ancora
   eseguibile, con un errore specifico.
2. Inventariare metadati, tipi quantizzati, nomi e forme di tutti i tensori dei
   tre GGUF.
3. Implementare tokenizer, BOS/EOS, template chat, thinking, reasoning effort,
   protocollo tool e stop condition con golden test.
4. Modellare DSA e IndexShare, routing MoE, esperto condiviso, KV e MTP in tipi
   GLM separati dal backend DeepSeek.
5. Implementare loader e kernel Metal con riferimenti CPU e test numerici di
   embedding, prefill, token singolo e output head.
6. Esporre capability e impostazioni GUI senza applicare automaticamente knob
   DeepSeek a GLM.
7. Abilitare selezione e demo soltanto dopo una prova end-to-end bit-exact o con
   tolleranza numerica documentata sul GGUF reale; server e distribuzione
   vengono dopo il runtime locale.

## Regola di avanzamento

Lo stato passa da `downloadOnly` a `runnable` solo quando tokenizer, schema
tensori, decoder, qualità e lifecycle memoria sono verificati insieme. Il solo
fatto che llama.cpp riesca ad aprire lo stesso GGUF non costituisce una prova
del backend nativo DwarfStar.
