# Model/Common

Contratti portabili per identificare un'architettura senza selezionare per
errore il backend di un'altra famiglia.

- `ModelArchitecture.swift` definisce l'identificatore canonico, la famiglia,
  la disponibilità del backend, le capability minime e il detector GGUF.

Il detector riconosce la famiglia Qwen ma la segnala esplicitamente come priva
di backend in questa fase. Il fallback sulle chiavi `deepseek4.*` viene usato
solo per vecchi GGUF privi di `general.architecture`; un valore esplicito resta
sempre autorevole.

