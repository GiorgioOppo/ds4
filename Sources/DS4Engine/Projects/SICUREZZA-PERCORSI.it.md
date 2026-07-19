[English](SICUREZZA-PERCORSI.md) | **Italiano**

# Sicurezza dei percorsi progetto

## Confine

La root importata è l'unico spazio autorizzato. Ogni argomento tool è trattato
come percorso relativo: componenti `..`, path assoluti e destinazioni il cui
percorso standardizzato/risolto esce dalla root devono essere rifiutati.

## Link simbolici

I symlink non vengono indicizzati e non sono mai percorsi validi per i tool,
anche quando puntano a una destinazione interna. Prima di leggere, scrivere,
modificare o eliminare si controlla separatamente ogni componente esistente
sotto la root. Il controllo non si limita a risolvere l'URL finale: se la foglia
non esiste, Foundation può lasciare irrisolto un symlink in una directory padre.
Le directory realmente mancanti restano valide per la creazione di nuovi file.
Le scritture riconvalidano il percorso dopo aver creato le directory intermedie
e subito prima dell'I/O, così da ridurre la finestra TOCTOU.

## Indicizzazione

Directory generate o molto pesanti sono escluse, il numero di file e la
dimensione indicizzabile sono limitati, e vengono accettate estensioni testuali
note. File non indicizzati possono essere letti in range soltanto attraverso i
percorsi raw che applicano gli stessi controlli di confine.

## Modifiche

Un edit rilegge il file da disco prima di applicare la sostituzione, evitando di
sovrascrivere silenziosamente cambiamenti effettuati dall'editor o da git. La
cancellazione è limitata ai file; directory e root non sono target validi.

Ogni nuova API di `ProjectCache` deve mantenere queste invarianti e avere test
per traversal, symlink, percorsi inesistenti e cambiamenti concorrenti.
