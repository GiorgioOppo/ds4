# Sicurezza dei percorsi progetto

## Confine

La root importata è l'unico spazio autorizzato. Ogni argomento tool è trattato
come percorso relativo: componenti `..`, path assoluti e destinazioni il cui
percorso standardizzato/risolto esce dalla root devono essere rifiutati.

## Link simbolici

I symlink non vengono indicizzati. Prima di leggere, scrivere o eliminare si
risolve nuovamente il percorso e si verifica che resti sotto la root; questo
controllo deve avvenire al momento dell'I/O per ridurre rischi TOCTOU.

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
