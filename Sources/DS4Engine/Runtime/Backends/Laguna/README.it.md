[English](README.md) | **Italiano**

# Registrazione runtime Laguna

`LagunaBackendDefinition` è il record di registrazione di Laguna S 2.1: le
capacità frontend che pubblica (chat, tool, reasoning, MoE) e il gate di
runtime inoltrato da `LagunaRuntimeGate` in DS4Metal, attualmente spento.
Finché il gate è spento, il selettore dei backend rifiuta un GGUF Laguna con
un errore esplicito di non-implementato invece di instradarlo verso un altro
decoder.
