[English](README.md) | **Italiano**

# Builtins/Files

Espone accesso raw e modifiche puntuali ai file sotto la root importata.

## Tool

- `file_read` e `file_lines`: contenuto o intervalli di righe.
- `file_write` e `file_add`: creazione/scrittura controllata.
- `file_modify`: sostituzione mirata.
- `file_delete`: eliminazione di un singolo file, mai directory.

## Flusso e dipendenze

Le specifiche delegano a [`ProjectCache`](../../../Projects/README.it.md), che
standardizza e ricontrolla i percorsi. Sono tutti `projectScoped` e gli output
di lettura sono limitati.

## Estensione

Applicare le invarianti in
[`Projects/SICUREZZA-PERCORSI.md`](../../../Projects/SICUREZZA-PERCORSI.it.md),
evitare glob distruttivi e richiedere testo di ricerca univoco per modifiche.
Una nuova operazione di directory necessita una revisione esplicita del modello
di autorizzazione.
