[English](README.md) | **Italiano**

# Comando DS4Demo

`main.swift` è il punto di ingresso della CLI. Analizza gli argomenti
posizionali e i knob d'ambiente, inizializza Metal, opzionalmente esegue
l'audit di un GGUF, effettua uno smoke test della forward, poi esegue il
prefill del prompt e il decode in streaming.

Le dipendenze vanno direttamente a `DS4Core`, `DS4Metal` e al framework
`Metal` di Apple; la demo intenzionalmente non dipende da `DS4Engine` né da
SwiftUI. Usa il canonico `ModelArchitectureDetector` di DS4Core prima di
costruire il tokenizer o il decoder DeepSeek. Questo mantiene invariato il
grafo dei target producendo lo stesso rifiuto precoce di Qwen/sconosciuto
della factory dell'engine.

Sono accettati sia Flash sia il profilo Pro Q2 a file singolo. I loro metadati
validati producono una `DSV4RuntimeGeometry` specifica del profilo; il package
di catalogo Pro Q4 non è accettato perché la CLI non assembla i suoi due
shard.

Lo **split Pro Q4 per range di layer** è accettato come lista di path separati da
virgola (`DS4Demo shardA.gguf,shardB.gguf …`): la CLI apre un `GGUFShardSet`, usa
il primo shard per tokenizer/config/metadata e costruisce il decoder con
`StreamingDecoder.fromGGUFShards` (ogni layer caricato dal suo shard). È il
percorso resident semplice — un nodo ad alta RAM che mappa entrambi gli shard.

## Sottocomando `requantize`

`DS4Demo requantize <in.gguf> <out.gguf> RULE [RULE ...]` esegue una
requantizzazione OFFLINE GGUF -> GGUF ([`RequantizeCommand.swift`](RequantizeCommand.swift)).
È intercettato in `main.swift` PRIMA dell'avvio del runtime Metal, quindi
funziona su macchine senza GPU Apple. Una `RULE` è `SRC>DST[@NOME]` con i nomi
dei tipi GGUF (es. `f16>q4_k`, `q8_0>q4_k@ffn_down.weight`); i tensori non
corrispondenti passano invariati. Delega a `GGUFRequantizer` di `DS4Core` e non
tocca codice GPU.

Mantieni la compatibilità degli argomenti documentata nel README padre. Sposta
gli helper riutilizzabili di logging, ispezione del modello o misurazione del
disco in `Diagnostics/` invece di espandere il file del comando. I nuovi
default numerici devono essere espliciti perché questo eseguibile è usato per
confronti di prestazioni e di parità.
