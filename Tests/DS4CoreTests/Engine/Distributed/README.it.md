# Test dell'Engine distribuito

`DistProtocolTests.swift` copre codifica/decodifica dei messaggi, framing,
handshake, messaggi di file/KV/lavoro, assegnazioni degli esperti e il rifiuto
di input malformati.

Le aggiunte al protocollo richiedono test di round-trip, di confine e di frame
non validi. Evita assunzioni multi-host live; l'integrazione dei transport
dovrebbe usare loopback o stream iniettati con timeout deterministici.
