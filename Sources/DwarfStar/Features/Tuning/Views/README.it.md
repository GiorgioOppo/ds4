# Viste di Tuning

- `TuningView.swift` presenta il dimensionamento della cache degli esperti,
  la politica del pool mixed-quant, l'hit rate e la concentrazione del
  routing forniti dal motore attivo.
- `AgentsView.swift` modifica prompt degli agent, icone, concessioni dei tool
  e import/export JSON.

Le viste modificano configurazione sostenuta dal motore, ma non implementano
il routing né l'autorizzazione dei tool. I binding diretti di tuning sono
disabilitati per l'intera durata del lease del benchmark completo/auto-tune
macchina, così non possono persistere un candidato ibrido mentre un'altra
scheda possiede l'esecuzione. Valida i dati degli agent importati prima di
persisterli e mantieni la telemetria delle prestazioni in sola lettura dal
livello di presentazione.
