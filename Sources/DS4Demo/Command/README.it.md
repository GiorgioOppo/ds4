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

Mantieni la compatibilità degli argomenti documentata nel README padre. Sposta
gli helper riutilizzabili di logging, ispezione del modello o misurazione del
disco in `Diagnostics/` invece di espandere il file del comando. I nuovi
default numerici devono essere espliciti perché questo eseguibile è usato per
confronti di prestazioni e di parità.
