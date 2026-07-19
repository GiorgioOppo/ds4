[English](README.md) | **Italiano**

# Script

Script di build e di analisi.

La metodologia delle prestazioni e la precedenza della configurazione sono
documentate in
[`docs/VALUTAZIONE-DEMO-PERF.md`](../docs/VALUTAZIONE-DEMO-PERF.it.md) e
[`docs/CONFIGURAZIONE-E-PROFILI.md`](../docs/CONFIGURAZIONE-E-PROFILI.it.md).

- **`embed_kernels.sh`** rigenera
  `Sources/DS4Metal/Runtime/Generated/KernelSources.swift` da `metal/*.metal`,
  incorporando il codice sorgente dei kernel nel binario. Viene invocato da
  `make embed-kernels`.
- **`bench.sh`** esegue la matrice di benchmark dei knob di prefill/decode con
  il binario release di `DS4Demo` e raccoglie le righe chiave di ogni run in un
  unico report:
  `scripts/bench.sh <gguf> <prompt-file> [report.txt] [case ...]`.
  La sua matrice integrata preserva intenzionalmente la baseline storica a 16
  slot usata dalle misurazioni di luglio 2026. È un harness A/B, **non** il
  preset attuale della GUI (22 slot, cache Q4 estesa e MetalIO con fallback
  automatico). Registra ogni valore `DS4_*` effettivo in un report prima di
  confrontarlo con una run dell'app attuale.
- **`metal_ab.sh`** valida un singolo knob del runtime Metal in due processi
  release separati. Fissa il campionamento greedy, disabilita lo speculative
  decode e la persistenza dell'utilizzo, confronta gli id dei token generati e
  logit a vocabolario completo limitati, e riporta lo speedup di
  prefill/decode. La traccia è mantenuta in memoria con copy-on-write durante
  la regione cronometrata e serializzata solo in seguito, così l'I/O della
  traccia non viene conteggiato nel throughput. Uso:
  `scripts/metal_ab.sh <gguf> <prompt-file> <DS4_KNOB> [base] [candidate] [max-new] [out-dir]`.
  `metal_ab_compare.py` è il suo analizzatore basato sulla libreria standard e
  sul memory-mapping; esegui
  `python3 scripts/metal_ab_compare.py --self-test` senza un modello per
  verificarlo. Il gate numerico predefinito è `atol=1e-4, rtol=1e-4`; imposta
  `DS4_AB_ATOL=0 DS4_AB_RTOL=0` quando una modifica promette la parità bit a
  bit. Una singola coppia è esplorativa: ripeti con
  `DS4_AB_ORDER=candidate-first` prima di promuovere un knob, perché il calore
  della cache, lo stato termico e la pressione di memoria possono favorire
  l'uno o l'altro processo. Il runner rispetta un `DEVELOPER_DIR` esistente;
  altrimenti usa automaticamente
  `/Applications/Xcode.app/Contents/Developer` quando l'`xcode-select` attivo
  punta a un'installazione standalone di CommandLineTools incompatibile.
- **`metal_autotune.py`** cerca una configurazione combinata con coordinate
  ascent multi-pass. Spazza piccole griglie hardware non monotone, percorre i
  valori ordinati in entrambe le direzioni finché le prestazioni crescono,
  congela un seed di usage-imatrix per processo, rifiuta le run contaminate da
  RAM/swap, richiede `PASS_EXACT` per i knob lossless, conferma i finalisti in
  ordine ABBA e scrive un report riprendibile a prova di crash più
  `final-env.sh`. Il profilo standard regola i knob sicuri di decode/I/O; i
  knob di prefill richiedono un prompt separato di almeno 1024–2048 token.
  Esclude deliberatamente la quantizzazione e le modifiche a esperti ridotti.
  Vedi [`docs/AUTOTUNING-METAL.md`](../docs/AUTOTUNING-METAL.it.md). Esegui i test
  senza modello con `python3 scripts/metal_autotune.py --self-test`.
- **`gguf_spectrum.py`** e **`gguf_to_graph.py`** sono strumenti di analisi
  della compressione GGUF (ispezione dello spettro dei valori singolari ed
  esportazione del grafo fattorizzato). Vedi
  [`README-analisi.md`](README-analisi.it.md).
