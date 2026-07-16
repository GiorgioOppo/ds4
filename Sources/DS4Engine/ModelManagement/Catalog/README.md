# ModelManagement/Catalog

Contiene il catalogo tipizzato usato dalla GUI. Le tre varianti complete
DeepSeek V4 Flash e il Pro Q2 in un singolo GGUF sono scaricabili e
selezionabili. Il package PRO Q4 è visibile e scaricabile, ma non avviabile dal
loader locale perché è suddiviso in due shard.

Ogni `ModelCatalogEntry` raggruppa uno o più `ModelTarget`: un modello completo
ha un solo artefatto principale, mentre PRO Q4 è un package di due shard. MTP è
un accessorio separato e non compare nel catalogo dei modelli principali.

Il nome remoto, l'identificatore e il digest SHA-256 sono centralizzati qui:
la GUI non deve duplicare nomi file o dedurre il supporto dal filename.

## Matrice corrente

| ID | Edizione | Forma | Disponibilità |
|---|---|---|---|
| `q2-imatrix` | Flash | un GGUF Q2/IQ2XXS | `runnable` |
| `q2-q4-imatrix` | Flash | un GGUF mixed Q2/Q4 | `runnable` |
| `q4-imatrix` | Flash | un GGUF Q4 | `runnable` |
| `pro-q2-imatrix` | Pro | un GGUF Q2 | `runnable` |
| `pro-q4-split` | Pro | due shard Q4 | `downloadOnly` |

`ModelCatalogEntry.isSelectable` richiede runtime `runnable`, un solo artefatto
e ruolo `mainModel`. Questa regola impedisce a un package split o a un
accessorio di diventare un modello locale. `DeepSeekV4AccessoryCatalog.mtp`
resta separato e non viene iterato da `DeepSeekV4ModelCatalog.entries`.

La disponibilità dei profili singolo-file deriva da
`DeepSeekV4BackendDefinition.locallyRunnableVariants`; il catalogo non mantiene
un secondo flag Pro. Quando si estende il catalogo, fissare filename e SHA-256
dalla fonte remota, conservare gli ID già persistiti e abilitare un profilo solo
dopo una validazione end-to-end di loader, tokenizer, decoder e forma.
