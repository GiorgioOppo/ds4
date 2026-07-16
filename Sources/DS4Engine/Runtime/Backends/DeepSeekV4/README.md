# Backend DeepSeek V4

Registra il backend già operativo: generazione locale, reasoning e DSML tools,
KV su disco, routing/cache/bundle esperti e tuning specifico. Flash mantiene il
percorso distribuito verificato; la distribuzione Pro è ancora in verifica.

`DeepSeekV4BackendDefinition.locallyRunnableVariants` è l'unica dichiarazione
dei profili locali: comprende Flash e Pro. Sia `BackendSelector` sia il catalogo
modelli derivano il proprio gate da questa lista, evitando stati incoerenti.

La factory continua a costruire il `StreamingDecoder` concreto esistente. Questo
adapter serve a isolare la selezione e le capability senza aggiungere dispatch
nel percorso caldo. Flash e Pro ricevono una `DSV4RuntimeGeometry` diversa; il
supporto Pro locale riguarda il GGUF Q2 singolo, non il package Q4 a due shard.
