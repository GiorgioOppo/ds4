[English](README.md) | **Italiano**

# Viste di Distributed

`DistributedView.swift` contiene il pannello di configurazione del worker,
`CoordinatorChatView` e la presentazione dei log distribuiti.

Le view si collegano a `DistributedController` e alle impostazioni condivise.
Non implementare decisioni di trasporto o di protocollo in SwiftUI. I nuovi
pannelli specifici per ruolo dovrebbero essere suddivisi in file mirati quando
crescono oltre la semplice presentazione.
