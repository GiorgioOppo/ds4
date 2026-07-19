[English](GUIDA-SVILUPPO.md) | **Italiano**

# Guida allo sviluppo

Questa è la checklist pratica per modificare il repository senza rompere i
confini fra moduli o i flussi di generazione. La mappa completa è in
[STRUTTURA-PROGETTO.md](STRUTTURA-PROGETTO.it.md).

## Prima di modificare

1. Identificare il target proprietario del comportamento.
2. Leggere il `README.md` della cartella e quello del target.
3. Controllare se il file è generato.
4. Individuare i test del dominio speculare.
5. Separare correttezza, qualità e prestazioni nella proposta.

## Confini dei target

```text
DS4Core <- DS4Metal <- DS4Engine <- DwarfStar
    \----------- DS4Demo --------/
```

- Core non importa Metal, rete o SwiftUI.
- Metal non conosce Engine, HTTP o GUI.
- Engine orchestra il modello ma non importa SwiftUI.
- DwarfStar presenta stato e azioni, senza implementare matematica GPU.
- DS4Demo usa Core e Metal direttamente per diagnosi e prestazioni.

## Dove collocare il codice

| Tipo di modifica | Posizione |
|---|---|
| DTO portabile o formato file | `DS4Core` |
| tensore, cache GPU o dispatch | `DS4Metal` |
| API d'inferenza, persistenza o rete | `DS4Engine` |
| stato e vista SwiftUI | `DwarfStar/Features/<Feature>` |
| CLI e audit | `DS4Demo` |
| sorgente GPU | `metal` |
| test | dominio speculare sotto `Tests/DS4CoreTests` |

Preferire un tipo principale o un'estensione coesa per file. Le estensioni
seguono `Tipo+Responsabilita.swift`.

## Politica dei README

Ogni cartella significativa del repository ha un `README.md`. Il file locale
deve rispondere a quattro domande:

1. che cosa possiede la cartella;
2. quali sono i file o tipi principali;
3. da che cosa può dipendere;
4. come si estende e come si verifica.

Il README locale non deve duplicare tabelle enormi o dettagli destinati a
cambiare spesso. Per un sottosistema saliente collega un documento sotto
`docs/`. Dopo aggiunte o spostamenti verificare che ogni nuova directory abbia
il proprio README.

## File generati

Non modificare manualmente:

- `Sources/DS4Metal/Runtime/Generated/KernelSources.swift`;
- `DwarfStar.xcodeproj/project.pbxproj` come fonte primaria.

Per i kernel modificare `metal/*.metal` ed eseguire `make embed-kernels`. Per
il progetto Xcode modificare `project.yml` ed eseguire `xcodegen generate`.

## Workflow di compilazione

```sh
swift build --disable-sandbox
swift test --disable-sandbox
swift build -c release --product DS4Demo --disable-sandbox
xcodegen generate
```

Su macOS impostare `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`
se `xcrun` non usa l'installazione completa di Xcode.

## Modifiche ai kernel

Seguire [BACKEND-METAL.md](BACKEND-METAL.it.md): kernel `.metal`, wrapper Swift,
composizione del grafo, test CPU/GPU, embedding e build Release sono un'unica
unità di modifica.

## Modifiche alla pipeline

Tenere separati:

- dati e rendering;
- stato applicativo;
- stato KV/GPU;
- I/O dei pesi;
- sampling e presentazione.

Consultare [PIPELINE-INFERENZA.md](PIPELINE-INFERENZA.it.md). Una nuova modalità
non deve creare un secondo motore dentro la GUI.

## Modifiche al protocollo distribuito

1. Aggiungere il tipo wire sotto `Distributed/Protocol`.
2. Definire limiti e decoder bound-checked.
3. Aggiungere test round-trip e payload malformati.
4. Aggiornare coordinator e worker separatamente.
5. Incrementare `Dist.protocolVersion` per cambi incompatibili.
6. Aggiornare [INFERENZA-DISTRIBUITA.md](INFERENZA-DISTRIBUITA.it.md).

Non consentire al wire di impostare ambiente arbitrario.

## Modifiche alla GUI o al server

Tenere parsing/protocollo, servizi, controller e viste in file distinti. Le
azioni lunghe devono rispettare cancellazione e actor isolation. Consultare
[GUI-SERVER-E-API.md](GUI-SERVER-E-API.it.md).

## Documentazione

Quando cambia un comportamento:

- aggiornare il README della cartella proprietaria;
- aggiornare il documento tematico;
- correggere esempi e configurazione nel README principale;
- verificare tutti i link relativi;
- evitare di presentare design futuri come funzioni già operative.

I documenti sperimentali devono dichiarare chiaramente stato, data delle misure
e condizioni del benchmark.

## Checklist finale

- [ ] Dipendenze orientate nel verso corretto.
- [ ] Nessun file monolitico cresciuto con responsabilità scollegate.
- [ ] README presente nelle nuove cartelle.
- [ ] Manifesto SwiftPM e XcodeGen aggiornati.
- [ ] Test proporzionati al rischio eseguiti.
- [ ] Kernel incorporati rigenerati quando necessario.
- [ ] Nessun percorso documentale obsoleto.
- [ ] `git diff --check` pulito.

La strategia di test completa è in
[TESTING-E-VALIDAZIONE.md](TESTING-E-VALIDAZIONE.it.md).
