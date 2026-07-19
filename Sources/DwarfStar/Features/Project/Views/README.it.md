[English](README.md) | **Italiano**

# View dei progetti

`ProjectView.swift` contiene la UI della libreria dei progetti e l'adapter di
bookmark `ProjectLibrary` rivolto all'app. Le cartelle importate restano al
loro posto; i cloni GitHub gestiti vivono sotto Application Support e possono
essere eliminati alla rimozione.

L'indicizzazione e le operazioni sicure sui file appartengono a
`DS4Engine.ProjectCache`. Preserva la distinzione tra dimenticare un bookmark
importato ed eliminare un clone gestito dall'app, e non seguire mai symlink
non fidati verso file al di fuori della root di progetto selezionata.
