# Backend Metal GLM 5.2

Questo albero possiede il codice numerico per l'architettura `glm-dsa`. Il
codice GLM resta fuori da `Backends/DeepSeekV4` così le decisioni sulla
famiglia di modello vengono prese una sola volta al confine del runtime invece
che dentro gli hot loop dei token.

La prima tranche di correttezza implementa il router sigmoid esatto rispetto
all'architettura: 256 esperti, selezione top-8, bias solo per la selezione,
pesi normalizzati non distorti e scala 2.5. Vengono mantenuti sia un oracolo
scalare sia il kernel Metal, così i successivi percorsi MoE, residenti e di
streaming possono essere verificati contro lo stesso risultato.

La DSA compatta ha ora anche fixture numeriche isolate per la RMSNorm KV-LoRA,
il posizionamento della KV compatta F16, il posizionamento
LayerNorm/affine/RoPE centrato delle chiavi dell'indexer e lo scoring
dell'indexer a geometria fissa. Restano primitive di validazione, non un grafo
eseguibile.

`Streaming/` avvia il passo 1 della roadmap: `GLM52PayloadReader` esegue i
piani validati di weight map e di expert stream contro il payload GGUF reale
con `pread` limitati — descrittori e piani top-8 diventano byte, impacchettati
come record gate|up|down nell'ordine di rank del router.

Il backend non è ancora registrato come eseguibile. L'attention DSA compatta,
il caricamento dei tensori, l'esecuzione dense/MoE e il prefill/decode
completi devono superare le loro fixture prima che
`BackendCapabilities.generation` o la selezione da catalogo vengano abilitati.
