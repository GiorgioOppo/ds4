[English](README.md) | **Italiano**

# DwarfStar/Features/Tuning

- **`Views/TuningView.swift`** mostra gli slot della cache degli esperti
  (`DS4ExpertCacheSlots`, preset app misurato corrente: 22), la politica mista
  IQ2/Q4 consapevole del layer (`DS4MultiQuantCache`, attiva di default),
  l'hit-rate e la concentrazione del routing per layer tramite la imatrix di
  utilizzo. Vedi il
  [Riferimento di configurazione](../../../../README.it.md#riferimento-di-configurazione)
  nella radice.
- **`Views/AgentsView.swift`** modifica i ruoli degli agenti: prompt, icona,
  tool per agente e import/export JSON. Si tratta di gestione degli agenti più
  che di tuning, quindi è un candidato futuro per una cartella `Agents/`
  dedicata.

Questa feature presenta e modifica la configurazione del motore; non
implementa il routing degli esperti, la politica di cache, l'autorizzazione
dei tool né l'esecuzione degli agenti. Preserva questo confine quando aggiungi
controlli.

`TuningView` è vincolata alle capability: senza `expertRouting` mostra un
placeholder neutrale rispetto all'architettura invece degli stepper della
cache DeepSeek e delle statistiche di utilizzo.
