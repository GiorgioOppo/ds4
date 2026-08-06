[English](README.md) | **Italiano**

# ModelManagement/Catalog

Contiene il catalogo tipizzato cross-family usato dalla GUI. Oltre ai modelli
monolitici e ai package a shard include Kimi K3, pubblicato come cinque
frammenti consecutivi di un unico GGUF.

Ogni `ModelCatalogEntry` raggruppa uno o più `ModelTarget`: un modello completo
ha un solo artefatto principale, PRO Q4 è un package di due shard indipendenti
e Kimi K3 usa cinque target `splitFragment` ordinati. MTP è un accessorio
separato e non compare nel catalogo dei modelli principali.

Il nome remoto, sorgente/revisione Hugging Face, identificatore, dimensione
esatta disponibile e digest SHA-256 sono centralizzati qui: la GUI non deve
duplicare nomi file o dedurre il supporto dal filename.

## Matrice corrente

| ID | Edizione | Forma | Disponibilità |
|---|---|---|---|
| `q2-imatrix` | Flash | un GGUF Q2/IQ2XXS | `runnable` |
| `q2-q4-imatrix` | Flash | un GGUF mixed Q2/Q4 | `runnable` |
| `q4-imatrix` | Flash | un GGUF Q4 | `runnable` |
| `pro-q2-imatrix` | Pro | un GGUF Q2 | `runnable` |
| `pro-q4-split` | Pro | due shard Q4 | `downloadOnly` |
| `glm-5.2-iq2-xxs` | GLM 5.2 | un GGUF IQ2_XXS | `downloadOnly` |
| `glm-5.2-q2-k` | GLM 5.2 | un GGUF Q2_K | `downloadOnly` |
| `glm-5.2-q4-k` | GLM 5.2 | un GGUF Q4_K | `downloadOnly` |
| `kimi-k3-iq2-xxs-q2-k` | Kimi K3 | un GGUF in cinque parti consecutive | `downloadOnly` |

`ModelCatalogEntry.isSelectable` richiede runtime `runnable`, un solo artefatto
e ruolo `mainModel`. Questa regola impedisce a un package split o a un
accessorio di diventare un modello locale. `DeepSeekV4AccessoryCatalog.mtp`
resta separato. I supporti DSpark originale e 0731 sono componenti opzionali
mostrati soltanto tramite `ModelCatalogRegistry.downloadEntries`: non entrano
mai in `entries` o `selectableEntries`.

`KimiK3ModelCatalog` fissa revisione, dimensione e SHA-256 di ogni parte,
oltre a dimensione e digest del flusso ricostruito. Il downloader conserva le
parti separate: concatenarle oggi richiederebbe temporaneamente altri 858,8 GB.
Il futuro lettore virtuale mapperà gli offset logici sulle cinque parti senza
duplicare il modello.

La disponibilità dei profili singolo-file deriva da
`DeepSeekV4BackendDefinition.locallyRunnableVariants`; il catalogo non mantiene
un secondo flag Pro. Quando si estende il catalogo, fissare filename e SHA-256
dalla fonte remota, conservare gli ID già persistiti e abilitare un profilo solo
dopo una validazione end-to-end di loader, tokenizer, decoder e forma.

`GLM52ModelCatalog` usa il repository `antirez/glm-5.2-gguf` bloccato alla
revisione `2638b3b878f5c6cc3ae7334b8dbea1275025f52e`. Il registro ombrello
`ModelCatalogRegistry` concatena i cataloghi famiglia-specifici; ID e filename
devono essere globalmente unici.
