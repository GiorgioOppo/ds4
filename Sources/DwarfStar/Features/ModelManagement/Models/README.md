# Model Management Models

`ModelCatalog.swift` definisce il record leggero `DiscoveredModel` e lo scanner
usato dalla schermata di caricamento. Il catalogo remoto non è duplicato qui:
proviene da `DS4Engine.DeepSeekV4ModelCatalog`.

These are lightweight app catalog records, not GGUF parser types. Keep GGUF
metadata parsing in `DS4Core` and remote-download policy in `DS4Engine`.

Lo scanner automatico ammette solo i `primaryArtifact` delle entry che Engine
dichiara selezionabili, inclusi Flash e Pro Q2 singolo; non presenta MTP, shard,
package Pro Q4 o architetture future come modelli pronti al load. **Browse** resta disponibile
per quantizzazioni custom, ma `ModelPicker` esegue l'ispezione e la selezione del
backend prima di salvare il bookmark.
