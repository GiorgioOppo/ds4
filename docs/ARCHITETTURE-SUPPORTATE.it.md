[English](ARCHITETTURE-SUPPORTATE.md) | **Italiano**

# Architetture modello supportate

Questo documento definisce il confine tra le parti comuni di DwarfStar e i
backend specifici di una famiglia di modelli. È anche la fonte autorevole sullo
stato del supporto: il riconoscimento di un'architettura non implica che il suo
decoder sia già disponibile.

## Stato attuale

| Famiglia | Riconoscimento GGUF | Catalogo/download GUI | Inferenza locale | Selezione GUI e demo | Distribuzione | Note |
|---|---:|---:|---:|---:|---:|---|
| DeepSeek V4 Flash | sì | sì | sì | sì | sì | Tre quantizzazioni complete sono scaricabili, selezionabili ed eseguibili. |
| DeepSeek V4 Pro | sì | sì | sì, Q2 singolo | sì, Q2 singolo | sì, Q2 singolo | Il percorso distribuito è implementato e testato sul protocollo; la prova numerica con il GGUF Pro reale resta da eseguire. Il package Q4 split resta download-only. |
| GLM 5.2 (`glm-dsa`) | sì | sì | no | no, rifiuto esplicito | no | Detector, shape/schema, tokenizer/chat/tool e prime primitive DSA/Metal sono implementati; il decoder end-to-end non è ancora disponibile. |
| Qwen | sì | no | no | no, rifiuto esplicito | no | Struttura predisposta; decoder e template non sono ancora implementati. |
| Sconosciuta | sì | no | no | no, rifiuto esplicito | no | L'identificatore GGUF viene riportato senza tentare un caricamento DeepSeek. |

La selezione usa `general.architecture` del GGUF, normalizzato come
identificatore stabile. Un file Qwen o GLM non deve mai raggiungere il
validatore DeepSeek e produrre un errore fuorviante sui metadati
`deepseek4.*`. GLM dispone già del proprio frontend, ma ciò non cambia la
disponibilità del runtime.

## Catalogo GUI e supporto runtime

`ModelCatalogRegistry` in `DS4Engine/ModelManagement/Catalog` è la fonte
unica cross-family per i modelli remoti mostrati dalla GUI. Mantiene i cataloghi
famiglia-specifici `DeepSeekV4ModelCatalog` e `GLM52ModelCatalog`. Il registro descrive anche
artefatti che questa build può scaricare ma non eseguire: la presenza di una
riga non equivale quindi al supporto del decoder.

Le voci correnti sono:

- DeepSeek V4 Flash Q2 imatrix, mixed Q2/Q4 imatrix e Q4 imatrix: un GGUF
  completo ciascuno, `runnable` e selezionabile;
- DeepSeek V4 Pro Q2: un GGUF completo, `runnable` e selezionabile per il
  runtime locale;
- DeepSeek V4 Pro Q4: package `downloadOnly` di due shard; gli shard non sono
  presentati come modelli locali indipendenti;
- GLM 5.2 IQ2_XXS, Q2_K e Q4_K: tre GGUF monolitici alternativi dal repository
  `antirez/glm-5.2-gguf`, tutti `downloadOnly`;
- MTP: accessorio separato, escluso dal catalogo dei modelli principali e dalla
  selezione GUI.

La scansione automatica della GUI filtra sui filename delle tre voci Flash e
del Pro Q2 singolo selezionabili. **Browse** consente un file esterno, ma esegue
prima l'ispezione metadata-only e la selezione del backend: uno shard Pro Q4,
un GGUF GLM 5.2, MTP, Qwen e architetture sconosciute non possono sostituire silenziosamente il
modello attivo.

Il supporto Pro Q2 copre sia il decoder locale sia pipeline ed
expert-parallelism distribuiti. La geometria distribuita è coperta da test di
protocollo e partizionamento; resta esplicitamente pendente la campagna
numerica/multi-Mac con il GGUF Pro reale, che non era disponibile durante
questa implementazione.

## Confini del codice

Restano comuni alle architetture:

- lettura e mapping GGUF;
- sampling e rappresentazione dei token;
- runtime Metal, gestione di device, command queue e pipeline;
- API applicative, benchmark, server, agenti, tool e MCP;
- trasporto dei file e involucro persistente dei checkpoint;
- stato e persistenza della GUI.

Restano invece nel backend che li possiede:

- forma del modello e validazione dei metadati;
- tokenizer, token speciali e template conversazionale;
- nomi, layout e caricamento dei tensori;
- decoder, prefill, KV payload e diagnostica numerica;
- routing MoE, NSA, Hyper-Connection, MTP e cache esperti;
- capacità distribuite legate alla geometria del modello.

Questa separazione evita opzioni condizionali dentro il ciclo per layer. La
scelta del backend avviene una sola volta durante l'apertura del modello; il
percorso caldo continua a usare tipi concreti.

## Contratto di selezione

Il caricamento segue questa sequenza:

1. apertura del GGUF e lettura dell'identificatore di architettura;
2. costruzione di una descrizione neutrale del modello e delle sue capacità;
3. selezione del backend registrato per quella famiglia;
4. validazione specifica e costruzione del decoder concreto;
5. pubblicazione della descrizione a demo, GUI e diagnostica.

Un'architettura riconosciuta ma non implementata genera un errore distinto da
un file corrotto o da un profilo della stessa famiglia non supportato. Questo è
il comportamento previsto per Qwen e GLM durante le rispettive fasi
preparatorie. Un DeepSeek V4
Pro Q2 valido attraversa invece lo stesso backend concreto di Flash, ma con una
`DSV4RuntimeGeometry` costruita dai metadati: 61 layer, 7168 canali, 128 head,
384 esperti, indexer top-1024 e scala router 2,5 non vengono sostituiti dalle
costanti Flash.

## Compatibilità

La ristrutturazione non cambia:

- i target `DS4Core`, `DS4Metal`, `DS4Engine`, `DS4Demo` e `DwarfStar`;
- il nome dell'app, i bundle identifier e le cartelle Application Support;
- le chiavi `UserDefaults` e le variabili d'ambiente `DS4_*` esistenti;
- i tipi pubblici storici mantenuti come alias o façade;
- formato e lettura dei checkpoint DeepSeek già creati.

I nuovi checkpoint dovranno includere architettura e impronta del modello. Un
eventuale payload KV Qwen riceverà un proprio tipo/versione e non riutilizzerà
il payload DeepSeek cambiandone implicitamente il significato.

## Requisiti prima del backend Qwen

Prima di dichiarare Qwen utilizzabile servono, per una variante GGUF precisa:

1. inventario dei metadati e dei layout tensore reali;
2. tokenizer e template chat con golden test, inclusi reasoning, tool e stop;
3. decoder Metal separato con test numerici di prefill e token singolo;
4. gestione KV e contesto con una versione persistente propria;
5. profilo di impostazioni basato su capacità, senza mostrare knob DeepSeek;
6. smoke test con un GGUF piccolo e benchmark di correttezza;
7. solo in seguito, progettazione esplicita della distribuzione Qwen.

Non viene scelto in anticipo un generico “Qwen”: Qwen2, Qwen2-MoE, Qwen3 e le
relative varianti possono avere contratti diversi. L'implementazione inizierà
da un file GGUF e da una variante dichiarati, non dal solo nome commerciale.

## Requisiti prima del backend GLM 5.2

Il catalogo GLM fissa repository, revisione, filename, byte count e SHA-256.
`glm-dsa` è ora registrato, l'header IQ2_XXS reale è stato validato e frontend,
schema, router e policy DSA/IndexShare hanno test dedicati. Prima di rendere una
voce `runnable` restano il grafo completo MLA/DSA, MoE routed/shared, loader e
streaming pesi, KV runtime, output logits e confronti numerici di
prefill/decode. MTP non fa parte del primo percorso autoregressivo. I knob
DeepSeek non devono essere riutilizzati implicitamente. La checklist è nel
[README GLM 5.2](architectures/glm-5.2/README.it.md).

## Documenti per famiglia

- [`architectures/deepseek-v4/README.md`](architectures/deepseek-v4/README.it.md)
  descrive il backend operativo e i suoi vincoli.
- [`architectures/qwen/README.md`](architectures/qwen/README.it.md) descrive la
  predisposizione e ciò che resta da implementare.
- [`architectures/glm-5.2/README.md`](architectures/glm-5.2/README.it.md) descrive
  contratto verificato, tranche implementate e gate del backend GLM.
- [`STRUTTURA-PROGETTO.md`](STRUTTURA-PROGETTO.it.md) indica dove collocare i file.
