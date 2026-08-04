[English](README.md) | **Italiano**

# Builtins/Projects

Permette al modello di esplorare e modificare l'indice del progetto senza
inserire l'intero repository nel prompt.

## Tool

- `project_inspect`: superficie batch preferita e in sola lettura. Una richiesta
  limitata può combinare scope Git, albero/liste, ricerche per percorso/contenuto
  e più intervalli sorgente (12 operazioni, tetto rigido di 48.000 caratteri).
- `project_tree`, `project_list`, `project_find`: orientamento e percorsi.
- `project_read`, `project_search`: letture limitate e ricerca opzionalmente
  ristretta a una sottocartella. Restano per compatibilità; gli agenti di
  progetto predefiniti usano `project_inspect` per evitare un round per file.
- `project_write`, `project_edit`: scrittura e sostituzione esatta.
- `project_reload`: ricostruzione dell'indice dopo cambiamenti esterni.

## Flusso e dipendenze

Tutti i tool usano [`ProjectCache`](../../../Projects/README.it.md) e sono
`projectScoped`. Git reindicizza automaticamente dopo le operazioni che mutano
il working tree; editor e script esterni richiedono `project_reload`.

## Estensione

Preferire risposte aggregate e limitate per ridurre i round di inferenza senza
leggere l'intero repository. Separare ricerca dei nomi da ricerca del contenuto
e non caricare file freddi nella cache quando basta una scansione streaming.
