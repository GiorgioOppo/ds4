[English](README.md) | **Italiano**

# Test dei kernel Tensor

Operazioni GPU elementari: copy, trasformazioni unarie/binarie,
normalizzazione, softmax, GLU, gathering/scattering di righe, concatenazione,
ordinamento e riduzioni.

Ogni test di kernel deve includere i casi limite di forma/coda e un valore
atteso calcolato su CPU. Usa l'uguaglianza esatta solo quando il contratto
dell'implementazione è identico bit a bit; altrimenti documenta la tolleranza
accanto all'asserzione.
