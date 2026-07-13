# ModelManagement

Raccoglie operazioni sul modello che non fanno parte del ciclo di inferenza.

## Componenti

- [`Download`](Download/README.md): catalogo, credenziali e download GGUF.
- `ExpertBundleTool.swift`: verifica o costruisce il sidecar degli esperti senza
  caricare il decoder completo.

La procedura operativa è descritta in
[`GESTIONE-MODELLI.md`](GESTIONE-MODELLI.md).

## Dipendenze e flusso

Il downloader usa Foundation/CryptoKit; il token store usa Security. La
costruzione del bundle usa metadati `DS4Core` e logica `DS4Metal`. Il risultato
è poi consumato da [`Inference`](../Inference/README.md) o
[`Distributed`](../Distributed/README.md).

## Estensione

Una nuova trasformazione deve produrre un artefatto deterministico, verificabile
e separato dal GGUF originale. Non memorizzare segreti in UserDefaults o log e
non sostituire un file finale prima che download/verifica siano completi.
