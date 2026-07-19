[English](README.md) | **Italiano**

# Test del formato GGUF

`GGUFTests.swift` esercita header GGUF, metadati, array, descrittori di
tensori, allineamento e rifiuto di dati non validi o troncati usando fixture
compatte.

Mantieni i test di parsing indipendenti dal caricamento dei pesi in Metal.
Aggiungi un caso di fixture esplicito ogni volta che supporti un nuovo tipo di
metadati GGUF o una nuova regola di validazione.
