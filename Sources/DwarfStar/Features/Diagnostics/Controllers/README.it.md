[English](README.md) | **Italiano**

# Controller della diagnostica

`DiagnosticsController.swift` converte il percorso del modello e il testo di
input in diagnostica del tokenizer e del chat template tramite `DS4Engine`.
Possiede lo stato osservabile di caricamento, output ed errore sul main actor.

Tieni le chiamate al filesystem e all'engine fuori dalla view SwiftUI. Le
operazioni diagnostiche non devono creare un engine di inferenza né modificare
lo stato KV della chat attiva.
