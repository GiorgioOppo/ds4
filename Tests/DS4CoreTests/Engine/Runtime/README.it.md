# Test Runtime multi-backend

Questi test esercitano il confine di selezione senza caricare un modello reale:
DeepSeek V4 Flash e Pro devono selezionare il backend concreto, i profili DeepSeek
sconosciuti devono essere rifiutati, Qwen deve essere riconosciuto e
rifiutato come non ancora implementato, un'architettura sconosciuta deve produrre
un errore distinto.

I test numerici del decoder DeepSeek rimangono nelle cartelle Metal esistenti.
