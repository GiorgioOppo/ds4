# DeepSeekVocabPruner — Pruning del Vocabolario e degli Esperti

Modulo per la riduzione della dimensione del checkpoint tramite pruning del vocabolario (embedding + lm_head + tokenizer) e degli esperti MoE.

## Architettura

```
DeepSeekVocabPruner/
├── VocabPruner.swift          # Facade pubblico per pruning vocabolario
├── VocabPruneSpec.swift       # Specifica di pruning
├── VocabPruneEvent.swift      # Eventi di progresso
├── VocabAnalyzer.swift        # Analisi copertura vocabolario su corpus
├── VocabRewriter.swift        # Riscrittura checkpoint con vocabolario ridotto
├── ExpertPruner.swift         # Facade pruning esperti
├── ExpertPruneSpec.swift      # Specifica pruning esperti
├── ExpertPruneCheckpoint.swift# Checkpoint pruning esperti
├── ExpertAnalyzer.swift       # Analisi importanza esperti
├── ExpertRewriter.swift       # Riscrittura checkpoint esperti ridotti
├── ExpertKeepDecision.swift   # Decisione su quali esperti mantenere
└── PruneCheckpoint.swift      # Facade combinato pruning + riscrittura
```

## VocabPruner

### Funzionamento

1. **Analisi**: scorre un corpus testuale e conta quali token IDs vengono effettivamente usati
2. **Selezione**: mantiene solo i token usati + un margine di sicurezza (`keepMargin`)
3. **Riscrittura**: riscrive `embed.weight`, `head.weight`, `tokenizer.json` e `config.json`
4. **Output**: checkpoint ridotto, pronto per l'inferenza senza fine-tuning

```swift
public enum VocabPruner {
    public static func run(spec: VocabPruneSpec,
                          progress: @escaping (VocabPruneEvent) -> Void) async throws
}
```

### VocabPruneSpec

```swift
public struct VocabPruneSpec: Sendable {
    public var inputDir: URL           // Checkpoint originale
    public var outputDir: URL          // Checkpoint ridotto
    public var corpusDir: URL          // Corpus per analisi
    public var language: String        // "it", "en", "multilingual"
    public var keepMargin: Double      // Margine di sicurezza (default 0.1 = 10%)
    public var minTokens: Int          // Minimo token da mantenere
    public var concurrency: Int        // Thread paralleli per analisi
}
```

## ExpertPruner

### Funzionamento

1. **Analisi**: valuta l'importanza di ogni esperto tramite importanza su dataset di calibrazione
2. **Selezione**: mantiene solo gli esperti più importanti per layer
3. **Riscrittura**: imposta a large-negative i pesi gate per esperti eliminati
4. **Output**: checkpoint con esperti ridotti per layer

```swift
public enum ExpertPruner {
    public static func run(spec: ExpertPruneSpec,
                          progress: @escaping (ExpertPruneEvent) -> Void) async throws
}
```

### ExpertPruneSpec

```swift
public struct ExpertPruneSpec: Sendable {
    public var inputDir: URL
    public var outputDir: URL
    public var calibrationDir: URL      // Dataset calibrazione
    public var expertsPerLayer: Int     // Esperti da mantenere per layer
    public var scoreFunc: ScoreFunc     // .softmax, .sigmoid, .sqrtsoftplus
    public var pruneThreshold: Float    // Soglia importanza
}
```

### ExpertKeepDecision

```swift
public struct ExpertKeepDecision: Sendable {
    public var keep: [Bool]             // true = mantieni, false = elimina
    public var scores: [Float]          // Punteggio importanza
}
```

## Dettaglio Operazioni

### VocabRewriter
1. Legge `embed.weight` [V, D] dal checkpoint
2. Seleziona righe corrispondenti ai token mantenuti
3. Riscrive `embed.weight` ridotto [V', D]
4. Stessa operazione su `head.weight`
5. Riscrive `tokenizer.json` con solo i token mantenuti
6. Aggiorna `config.json` con nuovo `vocabSize`

### ExpertRewriter
1. Per ogni layer MoE, valuta quali esperti eliminare
2. Imposta righe gate weight a -1e9 per esperti eliminati
3. Opzionalmente, elimina fisicamente i pesi degli esperti eliminati
4. Aggiorna `config.json` con `prunedExperts`

## CLI

```bash
# Pruning vocabolario per italiano
vocab_pruner --vocab --input ./checkpoint --output ./checkpoint-it \
  --corpus ./corpus-it --language it --keep-margin 0.1

# Pruning esperti (mantieni 4 esperti per layer)
vocab_pruner --experts --input ./checkpoint --output ./checkpoint-pruned \
  --experts-per-layer 4 --calibration ./calib-data

# Combinato
vocab_pruner --all --input ./checkpoint --output ./checkpoint-optimized \
  --corpus ./corpus --language it --experts-per-layer 4
```