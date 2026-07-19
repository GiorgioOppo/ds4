[English](README.md) | **Italiano**

# Viste di Tuning

- `TuningView.swift` renderizza il pannello fornito dalla classe UI
  per-backend (`Settings/BackendUI`): DeepSeek mostra il dimensionamento
  della cache esperti, la politica del pool mixed-quant, l'hit rate e la
  concentrazione del routing; GLM riespone i suoi stepper di
  residenza/streaming con un rimando al benchmark di misura; gli altri
  backend ricadono su una vista non disponibile.
- `AgentsView.swift` modifica prompt degli agent, icone, concessioni dei tool
  e import/export JSON.

Le viste modificano configurazione sostenuta dal motore, ma non implementano
il routing né l'autorizzazione dei tool. I binding diretti di tuning sono
disabilitati per l'intera durata del lease del benchmark completo/auto-tune
macchina, così non possono persistere un candidato ibrido mentre un'altra
scheda possiede l'esecuzione. Valida i dati degli agent importati prima di
persisterli e mantieni la telemetria delle prestazioni in sola lettura dal
livello di presentazione.
