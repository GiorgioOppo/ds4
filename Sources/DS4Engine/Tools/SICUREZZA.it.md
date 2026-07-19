[English](SICUREZZA.md) | **Italiano**

# Sicurezza degli strumenti

Gli argomenti dei tool sono prodotti dal modello e possono derivare da contenuto
non fidato. Vanno quindi trattati come input ostile.

## File e progetti

Tutti i percorsi sono confinati alla root importata, risolti nuovamente prima
dell'I/O e sottoposti alle regole in
[`../Projects/SICUREZZA-PERCORSI.md`](../Projects/SICUREZZA-PERCORSI.it.md).
Output di lettura, lista e ricerca hanno limiti di righe, risultati e byte.

## Rete

`WebClient` accetta soltanto HTTP(S) verso indirizzi pubblici, ricontrolla ogni
redirect e impone timeout/dimensione. `GitHubTool` usa un host fisso e convalida
owner, repository e ref. Il built-in git locale non espone operazioni di rete.

## Sub-agent

Un sub-agent riceve l'intersezione fra tool richiesti e
`ToolRegistry.subAgentGrantable`. Gli strumenti che cambiano il progetto
globale o orchestrano altri agenti non devono essere concessi implicitamente.

## MCP

I server MCP sono codice esterno. I nomi vengono namespaced e la mappatura
inversa è registrata, non dedotta dalla stringa. I processi stdio ereditano il
sandbox dell'app; i server HTTP richiedono fiducia esplicita nell'endpoint.

## Checklist per un nuovo tool

- schema JSON minimale e validazione di tipo/range;
- autorizzazione e confine delle risorse dichiarati;
- cancellazione e timeout per operazioni lente;
- output limitato e nessun segreto nei log;
- test per argomenti malformati e percorsi/URL ostili.
