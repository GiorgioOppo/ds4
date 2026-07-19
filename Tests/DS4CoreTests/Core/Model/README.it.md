[English](README.md) | **Italiano**

# Test del modello core

- [`Common/`](Common/README.it.md) verifica identificatori dei backend,
  rilevamento e disponibilità senza GGUF né GPU.
- [`Backends/DeepSeekV4/`](Backends/DeepSeekV4/README.it.md) verifica forme,
  configurazione e alias compatibili con DeepSeek.

Quando aggiungi un invariante di famiglia di modello, copri sia una
configurazione valida sia lo specifico confine non valido che essa rifiuta.
