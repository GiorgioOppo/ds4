# ModelManagement/Download

Implementa download nativi e riprendibili dei GGUF, senza `curl` o processi
esterni.

## Componenti

- `ModelDownloader.swift`: `ModelTarget`, catalogo, HTTP Range, file `.part`,
  avanzamento e verifica SHA-256.
- `HFTokenStore.swift`: lettura/scrittura del token Hugging Face nel Keychain,
  forma mascherata e descrizione della fonte attiva.

## Flusso

Il downloader scrive in `<ggufDir>/<nome>.part`, riprende dall'offset presente e
rinomina solo a stream concluso. Un digest configurato in `ModelTarget.sha256`
è un vincolo: un mismatch elimina l'artefatto corrotto. Senza digest noto viene
riportato quello calcolato per poterlo fissare in seguito.

La risoluzione del token segue: esplicito → `HF_TOKEN` →
`~/.cache/huggingface/token`. Il Keychain è consultato dalla GUI, non dal metodo
generico `resolveToken`, così CLI e test non generano prompt inattesi.

## Dipendenze e sicurezza

Foundation gestisce URLSession/file, CryptoKit l'integrità e Security il
Keychain. La verifica del contenuto è indipendente dalla rotazione dei
certificati CDN. Non inserire token in URL, log o stato non protetto.

## Estensione

Per aggiungere un target, usare ID e nome file stabili, dimensione indicativa e
preferibilmente SHA-256 autorevole. Conservare cancellazione, resume e callback
di avanzamento; evitare buffer proporzionali alla dimensione del modello.

Un target scaricabile non implica compatibilità con il decoder. Il catalogo
corrente contiene anche Pro e MTP per acquisizione/ispezione: il runtime locale
e distribuito esegue soltanto Flash e non carica il componente MTP separato.
Ogni nuova voce deve dichiarare esplicitamente questo confine finché non esiste
un percorso di load validato.

Le impostazioni correlate sono nella
[Configuration Reference](../../../../README.md#configuration-reference).
