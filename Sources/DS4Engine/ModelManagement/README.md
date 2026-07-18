# ModelManagement

Raccoglie operazioni sul modello che non fanno parte del ciclo di inferenza.

## Componenti

- [`Catalog`](Catalog/README.md): registro cross-family, cataloghi DeepSeek V4
  e GLM 5.2, artefatti e disponibilità nel runtime corrente.
- [`Download`](Download/README.md): credenziali, download GGUF riprendibile e
  stato consumato dalla GUI.
- `ExpertBundleTool.swift`: verifica o costruisce il sidecar degli esperti senza
  caricare il decoder completo.
- `ModelFileDiagnostics.swift`: pre-flight del percorso modello — spiega la
  causa reale di un open fallito (file assente, `.part` orfano da riprendere,
  file nella Application Support legacy invisibile alla sandbox) con il rimedio
  nel messaggio; usato da `InferenceService` prima di aprire il GGUF.

La procedura operativa è descritta in
[`GESTIONE-MODELLI.md`](GESTIONE-MODELLI.md).

## Dipendenze e flusso

Il catalogo è la fonte unica e non dipende dalla GUI. Il downloader usa
Foundation/CryptoKit; il token store usa Security. La costruzione del bundle
usa metadati `DS4Core` e logica `DS4Metal`. Il risultato è poi consumato da [`Inference`](../Inference/README.md) o
[`Distributed`](../Distributed/README.md).

## Estensione

Una nuova trasformazione deve produrre un artefatto deterministico, verificabile
e separato dal GGUF originale. Non memorizzare segreti in UserDefaults o log e
non sostituire un file finale prima che download/verifica siano completi.

Essere presenti nel catalogo significa essere acquisibili, non necessariamente
eseguibili. Le tre voci DeepSeek V4 Flash e il Pro Q2 in un singolo GGUF sono
`runnable`; il Pro Q4 resta `downloadOnly` perché è un package multi-shard. I
tre GGUF monolitici GLM 5.2 sono anch'essi `downloadOnly` finché non esiste un
backend `glm-dsa` verificato.
