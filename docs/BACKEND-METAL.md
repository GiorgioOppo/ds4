# Backend Metal

Questa guida descrive i confini del runtime GPU condiviso e il percorso
corretto per modificare un backend, un wrapper o un kernel. Le formule del
backend DeepSeek sono approfondite in
[ARCHITETTURA-MOTORE.md](ARCHITETTURA-MOTORE.md); la separazione multi-modello
è definita in
[ARCHITETTURE-SUPPORTATE.md](ARCHITETTURE-SUPPORTATE.md).

## Livelli del backend

```text
Backend concreto (oggi DeepSeekV4/StreamingDecoder;
GLM52 resta in tranche isolate non collegate)
             |
             v
decoder, stato KV e provider dei pesi del backend
             |
             v
Graph/Core + Graph/Operations
             |
             v
wrapper Swift in Kernels/<Area>
             |
             v
funzioni Metal in metal/*.metal
```

| Area | Responsabilità |
|---|---|
| `Runtime/Core` | device, command queue, compilazione delle pipeline e `GPUTensor` |
| `Runtime/Generated` | copia generata dei sorgenti Metal incorporati nel binario |
| `Kernels` | binding Swift, argomenti, dimensioni di dispatch e buffer |
| `Graph` | composizione delle operazioni e gestione del command buffer |
| `Model/Quantization` | descrittori di quantizzazione realmente condivisi |
| `Backends/DeepSeekV4` | forma, tensori, streaming, esperti, MTP, decoder, prefill e KV DeepSeek |
| `Backends/Qwen` | placeholder documentato; nessuna implementazione GPU corrente |
| `metal` | implementazione GPU autorevole |

La selezione avviene prima di costruire il decoder. Il ciclo per layer usa il
tipo concreto del backend e non consulta un registro o un protocollo dinamico.

## Runtime e tensori

`MetalRuntime` seleziona il device, crea la coda, compila il sorgente
incorporato e conserva le pipeline per nome. `GPUTensor` associa un
`MTLBuffer` a forma, tipo, stride e offset. Un tensore è una vista di memoria:
chi introduce sottoviste deve verificare che il wrapper usato propaghi
correttamente `byteOffset`.

Le risorse Metal sono gestite da ARC, ma la correttezza temporale resta
esplicita: un buffer di staging non può essere riutilizzato prima che il
command buffer che lo legge sia concluso.

## Command buffer e sincronizzazione

`GraphContext` accumula dispatch correlati e decide quando eseguire `commit` e
quando attendere il risultato. Ogni attesa CPU/GPU nel percorso per token è
potenzialmente costosa. Le fusioni esistenti rimuovono passaggi intermedi senza
spostare politica applicativa nei kernel.

Prima di eliminare una sincronizzazione verificare:

- quale buffer produce il dispatch precedente;
- chi lo legge e su quale command buffer;
- se esiste una lettura CPU dei risultati;
- quando uno slot di staging o cache può essere sovrascritto;
- se l'ordine di accumulo numerico deve restare identico.

## Wrapper dei kernel

I wrapper sono raggruppati per funzione:

- `Kernels/Attention` — RoPE, compressione KV, flash attention e indexer;
- `Kernels/Compression` — compressori e hyper-connections;
- `Kernels/Dense` — proiezioni dense e matmul;
- `Kernels/MoE` — router ed esperti quantizzati;
- `Kernels/Tensor` — primitive generali, norm, softmax e copie.

Un wrapper deve validare forma e capacità dei buffer, creare una struttura
argomenti con layout compatibile Metal, impostare ogni offset e scegliere una
griglia coerente con i limiti del kernel. Non deve scegliere quanti esperti
attivare, quale prompt elaborare o quale strategia di cache usare.

## Sorgenti incorporati

I file modificabili sono `metal/*.metal`. Il file
`Sources/DS4Metal/Runtime/Generated/KernelSources.swift` è generato e non deve
essere editato manualmente.

Workflow obbligatorio:

```sh
make embed-kernels
swift test --disable-sandbox
swift build -c release --product DS4Demo --disable-sandbox
```

`make embed-kernels` concatena i sorgenti nello stesso ordine atteso dal
runtime. Se cambia la firma di una funzione, wrapper Swift e test devono essere
aggiornati nello stesso intervento.

## Pesi e quantizzazione

Il backend usa più rappresentazioni in base al tensore:

- F32/F16 per stato, cache e alcune proiezioni;
- Q8_0 per molti pesi densi;
- Q4_K, Q2_K e IQ2_XXS per gli esperti;
- cache derivate Q4/Q8 abilitate soltanto dai relativi toggle.

Nel backend DeepSeek, `LayerWeights` porta la quantizzazione effettiva del
singolo layer. Non bisogna
dedurre il formato da un'impostazione globale quando il GGUF può essere mixed
precision. `GGUFWeights` valida tipo, forma, offset e dimensione prima di
esporre un tensore al grafo.

## Streaming e memoria unificata

Apple Silicon condivide memoria CPU/GPU, ma ciò non elimina i costi di paging,
compressione e I/O. Le strategie principali sono:

- mmap per viste non copiate;
- staging denso con letture `pread` fuori dalla page cache;
- `mlock` best-effort per buffer caldi;
- slot-cache degli esperti;
- bundle contiguo per ridurre seek e numero di letture;
- MetalIO con fallback automatico quando la banda reale è insufficiente.

MetalIO non è accesso arbitrario dell'SSD da parte di uno shader: è caricamento
di risorse coordinato dal runtime Metal. Il command processor continua a usare
buffer Metal validi e il percorso mantiene un fallback CPU/`pread`.

## Regole numeriche

Le ottimizzazioni rientrano in tre categorie, che devono essere dichiarate nei
test e nella documentazione:

1. **bit-identiche**, stesso ordine e stessi risultati;
2. **matematicamente equivalenti**, ma con possibile differenza di pochi ulp;
3. **lossy**, per requantizzazione o riduzione del lavoro del modello.

Un aumento di throughput non basta per accettare un kernel. Servono almeno:

- confronto con l'implementazione CPU o il percorso precedente;
- copertura di dimensioni limite e quantizzazioni supportate;
- controllo NaN/Inf e bounds;
- confronto del testo generato con seed e prompt fissati;
- misura dopo warm-up senza profilazione invasiva.

I rischi dormienti già individuati sono elencati in
[metal/README.md](../metal/README.md); non abilitare un percorso indicato come
non raggiunto senza risolverne o validarne le precondizioni.

## Aggiungere una nuova operazione

1. Stabilire prima se l'operazione è condivisa o appartiene a un solo backend;
   poi scegliere il dominio (`Attention`, `Compression`, `Dense`, `MoE`,
   `Tensor`).
2. Aggiungere o modificare il kernel nel file `.metal` appropriato.
3. Creare un wrapper focalizzato sotto `Sources/DS4Metal/Kernels/<Area>`.
4. Comporlo nel backend proprietario; usare `Graph/Operations` solo per una
   primitiva con contratto realmente comune, senza stato della GUI o del
   servizio.
5. Aggiungere un test kernel con riferimento CPU.
6. Aggiungere un test di grafo se l'operazione partecipa a una catena.
7. Rigenerare `KernelSources.swift` e compilare demo e app.

## Diagnostica

`DecodeProfile` separa I/O esperti, route/attention, FFN e overhead. La
profilazione fine può aggiungere wait e alterare il throughput; usarla per
individuare proporzioni e colli di bottiglia, poi misurare la velocità finale
con la profilazione disattivata.

Vedere anche:

- [VALUTAZIONE-DEMO-PERF.md](VALUTAZIONE-DEMO-PERF.md)
- [PIPELINE-INFERENZA.md](PIPELINE-INFERENZA.md)
- [TESTING-E-VALIDAZIONE.md](TESTING-E-VALIDAZIONE.md)
