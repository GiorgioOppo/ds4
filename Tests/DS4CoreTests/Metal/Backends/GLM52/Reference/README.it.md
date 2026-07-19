[English](README.md) | **Italiano**

# Test del riferimento del layer GLM 5.2

Suite senza device per gli oracoli layer/forward: la scorciatoia di attenzione
del primo token è uguale alla catena manuale di value projection delle
primitive fissate, i layer densi e sparsi esibiscono la struttura residuale
pre-norm, il percorso sparso instrada con l'oracolo del router sui logit
calcolati internamente e recupera esattamente gli esperti selezionati in
ordine di rank, e il forward multi-layer è uguale all'applicazione sequenziale
dei layer. I test di rifiuto coprono le forme errate. I confronti
compositivi usano tolleranze piccole per la deriva dovuta all'ordine di
sommatoria; l'uguaglianza del routing è esatta.
