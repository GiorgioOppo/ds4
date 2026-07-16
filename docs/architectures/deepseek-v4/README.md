# Backend DeepSeek V4

DeepSeek V4 è il backend operativo di riferimento. Flash e Pro condividono lo
stesso decoder Metal concreto, ma ogni istanza riceve la geometria validata del
proprio GGUF. La ristrutturazione multi-modello ne conserva il percorso caldo e
rende esplicite le parti che appartengono alla famiglia.

## Componenti specifici

- metadati `deepseek4.*` e validazione Flash/Pro;
- tokenizer e token di controllo DeepSeek;
- template conversazionale e tool calling DSML;
- Hyper-Connection, NSA, compressori e routing MoE top-6;
- cache e streaming degli esperti, bundle laterale e MTP;
- decoder Metal, payload KV e diagnostica numerica;
- pipeline distribuita con geometria 43/256 per Flash o 61/384 per Pro;
  protocollo, slice e maschere sono testati per entrambi i profili.

## Stato dei profili

- **Flash**: inferenza locale, demo, GUI e distribuzione supportate per le tre
  quantizzazioni a file singolo del catalogo.
- **Pro Q2**: il GGUF singolo è scaricabile, selezionabile ed eseguibile
  localmente da GUI e demo. `DSV4RuntimeGeometry` porta nel decoder i 61 layer,
  7168 canali, 128 head, 384 esperti, indexer top-1024, scala router 2,5 e i
  rapporti di compressione per-layer del profilo.
- **Pro Q4 split**: i due shard sono scaricabili come package, ma restano
  `downloadOnly`; il loader locale non li combina in un modello unico.
- **Pro distribuito**: il GGUF Q2 completo è accettato da pipeline ed
  expert-parallelism. Protocollo e geometria sono coperti dai test; resta da
  eseguire la validazione numerica e prestazionale multi-Mac con il file reale.

Il router usa 256 lane per Flash e una rete bitonica da 512 lane per Pro; nel
secondo caso gli indici 384...511 sono padding a `-inf`, mentre normalizzazione
top-6 e scala dei pesi ricevono i valori della geometria attiva.

Le variabili `DS4_*` restano valide per compatibilità. I controlli specifici
per esperti, NSA, Q4 e streaming devono essere esposti dalla GUI solo quando il
backend dichiara la capacità corrispondente.

## Regressione

Ogni modifica strutturale deve mantenere i test esistenti di tokenizer, DSML,
shape, pesi, prefill, decode, KV, MoE e distribuzione. Questi test restano
esplicitamente DeepSeek: rinominarli “generici” nasconderebbe le assunzioni che
devono proteggere.
