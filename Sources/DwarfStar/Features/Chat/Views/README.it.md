[English](README.md) | **Italiano**

# Viste della Chat

Presentazione SwiftUI per le conversazioni:

- `ChatTabView` sceglie il contenuto locale, distribuito, di caricamento o di
  onboarding.
- `ChatView` compone la trascrizione, la toolbar e l'editor del prompt.
- `MessageRow`, `MarkdownView` e `ToolMessageViews` renderizzano il contenuto
  dei messaggi.
- `AttachmentViews`, `ToolSheets` e `ChatListView` forniscono controlli
  focalizzati.
- `ContentView` contiene un vecchio flusso di caricamento del modello e non è
  la root corrente.

Il form legacy pre-caricamento apre ancora la stessa `DownloadView` usata
dalle Settings; non deve definire una propria lista di modelli remoti.
L'acquisizione corrente dei modelli va documentata ed evoluta sotto
`Features/ModelManagement`.

Le viste osservano `ChatStore` e dovrebbero restare dichiarative. Metti le
decisioni su sessione, generazione, tool e ciclo di vita del modello nel view
model. Se un nuovo componente visivo diventa riutilizzabile in modo autonomo o
sostanzioso, dagli un file dedicato invece di estendere `ChatView.swift`.

L'header usa il descrittore del modello ispezionato/caricato e non assume mai
un fallback DeepSeek. I controlli di tool e reasoning vengono renderizzati
solo quando il backend selezionato dichiara la corrispondente capacità di
runtime.

Durante lo streaming dei token, l'autoscroll della trascrizione è senza
animazione e limitato a cinque aggiornamenti al secondo. Questo impedisce che
animazioni `ScrollViewProxy` sovrapposte sopravvivano allo smontaggio del
pannello Chat quando l'utente passa a un'altra sezione della sidebar; la
generazione resta di proprietà di `ChatStore` e continua a girare attraverso
la navigazione.
