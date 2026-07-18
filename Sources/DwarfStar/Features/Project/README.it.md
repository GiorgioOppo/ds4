# DwarfStar/Features/Project

- **`Views/ProjectView.swift`** gestisce la libreria dei progetti. Importa
  cartelle tramite bookmark della sandbox, le indicizza in `ProjectCache` e
  seleziona il progetto attivo usato dai tool `project_*` e `file_*` degli
  agent. Importare un progetto non aggiunge i suoi file alla memoria della
  chat; il contenuto entra nel modello solo quando un tool o un allegato lo
  fornisce.

Nella libreria convivono due tipi di voci:

- **Cartelle importate dall'utente** — scelte con `NSOpenPanel`, persistite
  come bookmark security-scoped; rimuoverne una la fa soltanto dimenticare (la
  cartella su disco resta intatta).
- **Cloni GitHub** — creati dal tool di chat `github_clone` sotto
  `Application Support/DwarfStar/github-projects`, tracciati tramite semplice
  percorso (il container dell'app non ha bisogno di bookmark).
  `ProjectLibrary.syncClonedRepos()` scansiona quella cartella e tiene la
  lista sincronizzata (viene eseguita quando appare la scheda Project o un
  menu di progetto), così un repo clonato in chat compare automaticamente ed è
  marcato attivo. Rimuovere un clone ELIMINA la copia dal disco — altrimenti
  la sincronizzazione successiva lo rielencherebbe; puoi riclonarlo in
  qualsiasi momento con `github_clone`.

Il livello vista possiede l'interazione con i bookmark e la presentazione
della libreria. Indicizzazione, interrogazione, modifiche sicure,
contenimento dei percorsi e rifiuto dei symlink appartengono a
`DS4Engine.ProjectCache`. Preserva le diverse semantiche di rimozione
descritte sopra.
