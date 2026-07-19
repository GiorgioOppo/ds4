[English](README.md) | **Italiano**

# Viste delle Impostazioni

`SettingsView.swift` modifica la configurazione condivisa di modello,
contesto, esecuzione, memoria, I/O e Hugging Face tramite `AppSettings`.
`MCPServersView.swift` e il suo `MCPStore` gestiscono le definizioni
persistite dei server MCP e lo stato live delle connessioni.

La sezione Modello presenta sia **Browse** sia **Scarica…**. Browse valida un
GGUF manuale tramite il selettore dell'Engine; Scarica apre lo sheet del
catalogo da `Features/ModelManagement`. La vista non deve dedurre il supporto
runtime da un nome di file: la selezionabilità di Flash e del Pro Q2 a file
singolo, più lo stato solo download del Pro Q4 e le tre voci GLM 5.2 solo
download, provengono dal catalogo dell'Engine.

Le impostazioni che cambiano il layout di memoria dell'engine si applicano al
successivo caricamento del modello. Mantieni le chiavi UserDefaults
centralizzate e stabili, memorizza le credenziali tramite gli helper Keychain
dell'engine e non creare mai una copia specifica per feature delle
impostazioni globali. Il ricaricamento è disabilitato mentre una generazione
di chat è attiva, e l'ingresso nelle Settings locali non ripete l'ispezione
distribuita della geometria del GGUF. Analogamente, benchmark e auto-tuning
non sono disponibili finché la generazione attiva non si ferma.
