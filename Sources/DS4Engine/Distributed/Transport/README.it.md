[English](README.md) | **Italiano**

# Distributed/Transport

Incapsula il trasporto TCP asincrono basato su Network.framework.

## Componenti

- `DistError`: errori di frame, rete, versione e trasferimento.
- `DistConnection`: connessione framed con timeout e letture esatte.
- `DistRouteEntry`: indirizzo e slice di un passaggio della route.
- `DistReturnListener`: listener del coordinator per i risultati terminali.

## Flusso e dipendenze

`DistConnection` aggiunge/rimuove soltanto l'header definito in
[`Protocol/Framing`](../Protocol/Framing/README.it.md); la semantica del payload
resta nei codec del protocollo. Coordinator e worker possiedono il ciclo di
vita delle connessioni.

## Estensione

Mantenere cancellabili connect, send e receive; imporre timeout e letture con
lunghezza esatta. TLS, autenticazione o un trasporto alternativo devono
preservare l'interfaccia a frame e avere una configurazione esplicita.
