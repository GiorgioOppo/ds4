[English](README.md) | **Italiano**

# DwarfStar/Features/Distributed

UI per l'inferenza distribuita. L'engine e il protocollo risiedono sotto
`DS4Engine/Distributed`.

- **`Controllers/DistributedController.swift`** guida entrambi i ruoli. Come
  **worker**, questo Mac possiede una fetta di layer. Come **coordinator** da
  Chat -> Distributed, connette i worker ed esegue la chat sul cluster. Espone
  inoltre il coordinator connesso al pannello Benchmark.
- **`Views/DistributedView.swift`** contiene `WorkerView` (il pannello Worker
  della sidebar: porta di ascolto e log — la fetta di layer, il modello e le
  impostazioni sono assegnati dal coordinator al momento della connessione) e
  `CoordinatorChatView` (la chat del coordinator mostrata dentro la scheda
  Chat in modalità Distribuita).

Valori predefiniti: porta worker 9100, lista peer `127.0.0.1:9100`, bit di
attivazione 32, chunk di prefill 32, max token 512, porta di ritorno 9099.
Consulta la [Guida di riferimento alla configurazione](../../../../README.it.md#riferimento-di-configurazione)
alla radice.

Mantieni il ciclo di vita dei ruoli e lo stato della UI nel controller, la
presentazione in `Views/` e ogni modifica di protocollo o trasporto in
`DS4Engine/Distributed`. Chat e Benchmark ricevono il coordinator connesso del
controller invece di creare connessioni proprie.
