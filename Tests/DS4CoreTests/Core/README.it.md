# Test di Core

Test deterministici per il modulo indipendente dalla piattaforma `DS4Core`. Le
aree figlie coprono il markup delle conversazioni, i formati di file, la
generazione, i metadati dei modelli, la pianificazione dello storage e la
tokenizzazione.

Questi test non dovrebbero richiedere un device Metal, accesso alla rete o un
GGUF di produzione. Preferisci piccole fixture in memoria e asserzioni esatte.
Il nuovo comportamento di `DS4Core` appartiene alla directory figlia
corrispondente.
