[English](AUTOTUNING-METAL.md) | **Italiano**

# Autotuning multi-parametro Metal

`scripts/metal_autotune.py` cerca una configurazione con maggiore throughput
senza promuovere un candidato che peggiora il controllo numerico. È pensato per
misure lunghe su un Mac reale, non per CI.

## Cosa garantisce

La ricerca parte da una configurazione completa e cambia un parametro alla
volta. Per ogni candidato:

1. esegue baseline e candidato in processi `DS4Demo` separati;
2. confronta token, argmax, top-3 e logits completi delle frame registrate;
3. scarta subito crash, valori non finiti, trace incomplete e regressioni;
4. se il risultato è vicino o migliore, ripete nell'ordine inverso;
5. calcola il rapporto bilanciato geometrico delle coppie AB e BA;
6. promuove il valore solo oltre il margine minimo, predefinito al 2%;
7. per i knob ordinati continua nella stessa direzione fino al primo
   peggioramento; per queue depth, slot cache e NSG prova invece tutta la
   piccola griglia, perché questi parametri non sono necessariamente unimodali;
8. ripete tutti i parametri in più passate per misurare le interazioni.

L'ordine complessivo di un finalista è quindi **ABBA**. Il risultato non è una
griglia cartesiana: non combina ogni valore di ogni parametro con tutti gli
altri. Le sweep monodimensionali evitano però di fermarsi, per esempio, a
`PREAD_SPLIT=2` senza misurare il valore `4`. Il coordinate ascent raggiunge un
massimo locale con un numero trattabile di prove.

Per i parametri dichiarati lossless il gate è `PASS_EXACT`: ogni bit Float32
deve coincidere. `PASS_NUMERIC` non è sufficiente, nemmeno con tolleranza zero.
La modalità `--allow-numeric` è esplicita e aggiunge requisiti conservativi:
token, argmax e top-3 ordinata identici, zero non finiti, tutti i valori entro
`atol=1e-4, rtol=1e-5`, errore assoluto massimo `1e-3` e NRMSE aggregato/per
frame non superiore a `1e-5`.

Il comparatore non si fida della top-3 salvata nel JSON: ricalcola direttamente
dai `.f32` top-3, valori finiti e hash FNV di ogni frame. Offset sovrapposti,
metadata stale, trace troncate e mapping token/argmax incoerenti sono un errore
fail-closed. Con i default `max-new=64` e `trace-frames=64` sono coperte tutte le
decisioni che hanno prodotto i token misurati.

Questa è parità sui prompt forniti, non una misura semantica universale. Per
questo il tuner esclude riduzione degli esperti attivi, nuove quantizzazioni,
`COMP_Q8` e gli altri cambiamenti lossy. Quelli devono usare anche il benchmark
teacher-forced di correttezza dell'app.

## Profili

| Profilo | Parametri |
|---|---|
| `io` | split `pread`, slot cache, allocazione usage-driven, look-ahead esperti e dense-ahead |
| `standard` | profilo `io` più `MOE_NSG` e `DENSE_Q4_NSG`; è il punto di partenza consigliato |
| `full` | aggiunge i flag lossless secondari e i knob prefill; può richiedere molte ore |
| `prefill` | union, chunk, route batch e FFN batch su un prompt lungo |
| `numeric` | solo knob numerically-close; richiede `--allow-numeric` |

Il profilo standard non cambia `RAW_RING=1`, le quantizzazioni già scelte o il
numero degli esperti. Il preset predefinito è quello mixed-Q4/IQ2 per M1 Pro
16 GB, backend pread con `RAW_RING=1`; valori `DS4_*` già esportati nel
terminale lo sovrascrivono. `--preset inherit` non inserisce valori impliciti:
per evitare una falsa baseline richiede che ogni knob selezionato sia già
esplicitamente presente nell'ambiente.

## Usage-imatrix deterministica

Cache allocation e look-ahead dipendono dalla storia di routing. Con
`--usage-seed auto`, il tuner cerca il file più ricco per lo stesso nome modello
nelle cartelle Application Support di DwarfStar, ne congela una copia e crea
una copia privata per ogni processo. In questo modo ogni A e B parte dalla
stessa storia e non sovrascrive quella della GUI.

Anche `final-env.sh` punta alla copia stabile congelata, così la configurazione
riproduce davvero i test dei knob usage-dependent.

Se non trova un seed, i parametri usage-dependent vengono saltati. È possibile
passare un file esplicito o usare `--allow-cold-usage`, ma quest'ultima modalità
non rappresenta il comportamento della GUI dopo più conversazioni.

## Esecuzione consigliata

Chiudere la GUI e le applicazioni che occupano molta memoria. Non eseguire il
benchmark mentre è attivo un download: un `.part` modificato negli ultimi dieci
minuti nella directory del modello blocca automaticamente il tuner. Il controllo
viene ripetuto prima di ogni processo, quindi intercetta anche un download
iniziato a ricerca già avviata.

Ogni run registra inoltre `memory_pressure` e i contatori `vm_stat`: sotto l'8%
di memoria libera o oltre 128 MiB di nuovi swapout il candidato viene respinto.
Le soglie sono configurabili, ma non conviene abbassarle per scegliere i default
di produzione.

Preparare il prompt:

```sh
printf '%s\n' \
  'raccontami la storia di roma come se dovessi scrivere un libro di storia' \
  > /tmp/ds4-roma-prompt.txt
```

Avviare il profilo standard:

```sh
cd "/Users/oppog/Documents/Project/DeepSeek v4 Metal/DeepSeek-V4-Pro-MacOS"

python3 scripts/metal_autotune.py \
  "/Users/oppog/Library/Containers/com.dwarfstar.app/Data/Library/Application Support/DwarfStar/models/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf" \
  /tmp/ds4-roma-prompt.txt \
  --profile standard \
  --context 100000 \
  --output /tmp/ds4-autotune-standard
```

`--context` deve corrispondere alla finestra realmente usata: cache KV e budget
RAM cambiano l'ottimo degli slot. Per una configurazione da 100k non va
promosso un risultato misurato soltanto col preset da 4096.

Ogni candidato richiede almeno due caricamenti completi; quelli promettenti ne
richiedono quattro. Prima di iniziare si può ispezionare lo spazio di ricerca
senza build o inferenza:

```sh
python3 scripts/metal_autotune.py MODEL.gguf prompt.txt \
  --profile full --dry-run
```

Per il prefill serve un testo che produca almeno 1024–2048 token. Il prompt
breve di Roma non attraversa i confini di `PREFILL_CHUNK` e non può scegliere
quel valore. Il profilo completo può usare due workload:

```sh
python3 scripts/metal_autotune.py MODEL.gguf prompt-decode.txt \
  --profile full \
  --context 100000 \
  --prefill-prompt corpus-prefill-lungo.txt \
  --output /tmp/ds4-autotune-full
```

Questo è il comando da usare per valutare **tutti i knob lossless** del
manifest. I due knob numerically-close restano esclusi; aggiungere
`--allow-numeric` solo in una seconda ricerca separata se si accetta il gate a
tolleranza.

Il decode resta l'obiettivo dei parametri decode; il prefill diventa l'obiettivo
dei propri parametri. L'altra metrica è una guardia secondaria e non può
regredire oltre la soglia configurata.

## Interruzione e ripresa

Ogni processo e ogni decisione vengono salvati con scrittura atomica. Dopo
`Ctrl-C`:

```sh
python3 scripts/metal_autotune.py MODEL.gguf prompt.txt \
  --profile standard \
  --output /tmp/ds4-autotune-standard \
  --resume
```

La ripresa viene rifiutata se sono cambiati binario, modello, prompt, usage
seed, ambiente iniziale, manifest, script del tuner/comparatore o policy di
accettazione. Un ID di run viene checkpointato prima di avviare il processo:
anche un `kill -9` può lasciare soltanto una cartella orfana, mai rompere il
resume.

## Risultati

La directory scelta contiene:

- `report.md`: configurazione iniziale/finale e tutte le decisioni;
- `results.csv` e `results.json`: dati macchina-legibili;
- `final-env.sh`: ambiente completo da caricare con `source`;
- `state.json` e `events.jsonl`: checkpoint e journal crash-safe;
- `runs/*/run.log`: log di ogni processo e ambiente effettivo.

Le trace Float32 intermedie vengono eliminate dopo il confronto, salvo
`--keep-traces`. `final-env.sh` mostra `VALIDATED` soltanto dopo che la
validazione finale iniziale-vs-finale ha superato il gate.

Applicazione dei vincitori a una demo successiva:

```sh
source /tmp/ds4-autotune-standard/final-env.sh
.build/release/DS4Demo MODEL.gguf 256 "@/tmp/ds4-roma-prompt.txt"
```

## Opzioni operative principali

| Opzione | Effetto |
|---|---|
| `--knobs A,B,C` | usa soltanto i knob elencati |
| `--min-gain 0.02` | margine minimo ABBA per una promozione |
| `--max-passes 2` | numero massimo di passate coordinate |
| `--context 100000` | finestra di contesto della configurazione da ottimizzare |
| `--cooldown 2` | pausa fra processi per pressione memoria/termica |
| `--usage-seed auto|off|PATH` | storia routing congelata |
| `--min-memory-free-percent 8` | respinge run conclusi sotto la soglia RAM |
| `--max-swapout-mib 128` | respinge run che causano troppo swapout |
| `--allow-numeric` | abilita il manifest numerically-close |
| `--keep-traces` | conserva JSON e Float32 di ogni confronto |
| `--skip-build` | riusa `.build/release/DS4Demo` |
| `--self-test` | test sintetici senza modello o GPU |

`--no-final-validation` è disponibile soltanto per esplorazioni interrotte: il
risultato avrà stato `complete_unvalidated` e `final-env.sh` resterà marcato
`NOT FINAL`.

Non usare `--allow-active-download` per una misura da promuovere: esiste solo
per diagnosi consapevoli e il report risultante sarebbe contaminato dall'I/O
esterno.
