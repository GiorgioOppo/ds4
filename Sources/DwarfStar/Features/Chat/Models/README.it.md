# Modelli della Chat

`ChatModels.swift` definisce i modelli di presentazione come `UIMessage` e
`ChatAttachment`. Fanno da ponte tra gli eventi dell'engine e uno stato
SwiftUI stabile e identificabile e intenzionalmente non contengono alcun
comportamento di persistenza o di generazione.

Quando aggiungi campi, aggiorna le mappature di persistenza se il valore deve
sopravvivere a un riavvio dell'app. Mantieni i DTO di conversazione di
proprietà dell'engine in `DS4Core` o `DS4Engine` invece di duplicarli qui.
