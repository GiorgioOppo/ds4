# Test Metal GLM 5.2

Questi test confrontano i kernel Metal GLM di proprietà dell'architettura con
oracoli scalari indipendenti. Usano input sintetici deterministici e quindi
girano senza scaricare il modello da 200+ GB.

La suite del router copre il bias di sola selezione, i pesi di rotta
normalizzati senza bias, la geometria top-8, gli estremi stabili della
sigmoide e gli spareggi deterministici sugli id degli esperti.

La suite compact-KV verifica gli esatti pattern di bit da F32 a F16, `pos0`
diverso da zero, il rifiuto oltre capacità e la conservazione delle righe di
cache fuori dall'intervallo scritto.

La suite di normalizzazione KV-LoRA verifica che la RMSNorm sia ristretta al
prefisso largo 512 e che ogni bit F32 grezzo nel payload K-RoPE largo 64
sopravviva.

La suite dello store delle chiavi dell'indexer distingue la LayerNorm centrata
dalla RMSNorm, verifica che solo `0..<64` riceva la RoPE, controlla posizioni
di cache diverse da zero e preserva le righe fuori dall'intervallo di
scrittura.

La suite dell'indexer copre la geometria fissa 32x128, le letture della cache
delle chiavi F16, la ReLU per testa prima della pesatura, la scala positiva,
l'output token-major e il mascheramento causale a `-infinity`. I confronti
Metal vengono saltati quando nessun device è disponibile; gli oracoli scalari
indipendenti e i test di validazione restano privi di device.

La suite di streaming (`Streaming/`) dimostra che il lettore di payload è
fedele al byte su file di pattern sintetici e che i suoi controlli sui limiti
rifiutano file troncati e piani malformati prima che un solo byte si muova.

La suite di attenzione compatta confronta i kernel a stadi (`qk_lowrank`,
`attention_indexed`, `value_project`) stadio per stadio con prodotti scalari
di riferimento e, concatenati end-to-end, con `GLM52AttentionCPUReference`
sulla stessa cache arrotondata a F16, oltre ai percorsi di rifiuto di
selezione/geometria.

La suite top-k dimostra che il dispatch multi-blocco argsort+merge è uguale a
`GLM52IndexerCPUReference.causalTopK` su punteggi distinti — percorso
single-pass e percorso merge — e che le righe causali a `-INFINITY` non
vengono mai selezionate.

La suite della catena DSA compone le primitive GPU end-to-end (punteggi
dell'indexer → top-k → attenzione compatta a stadi) contro la stessa catena
eseguita attraverso gli oracoli CPU: le selezioni devono corrispondere
esattamente, l'output dell'attenzione entro tolleranza.

La suite Q8_0 quantizza i pesi delle proiezioni con il quantizzatore di test
condiviso, dequantizza gli stessi byte e richiede che i kernel Q8 corrispondano
alle baseline F32 su quei valori dequantizzati — l'errore di quantizzazione
appartiene alla fixture, mai al kernel.

La suite rope-tail fissa l'identità alla posizione 0, i prefissi nope intatti,
la conservazione della norma per coppia, la composizione inversa e la parità
GPU-vs-oracolo sulle teste di query e sulla singola riga K a posizioni
moderate (la trigonometria fp32 di angoli enormi diverge per riduzione
dell'argomento — il caveat rope documentato).

La suite MoE applica la stessa disciplina ai kernel FFN quantizzati: fixture
Q4_K dal quantizzatore reale, blocchi Q2_K/Q5_K/Q6_K sintetizzati decodificati
dai riferimenti `Quantize`, percorsi Q8_0 densi/output-head su larghezze
multiple di 32 ma non di 256, confronti a stadi e concatenati con
`GLM52FFNCPUReference`, e rifiuti di contratto (tipi, dimensioni, larghezze).
