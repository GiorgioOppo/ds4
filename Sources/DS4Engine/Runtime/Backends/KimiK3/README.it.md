# Backend runtime Kimi K3

Questa cartella è il confine DS4Engine del futuro backend `kimi-k3`.
`KimiK3BackendDefinition` registra architettura e capability portabili, ma
non pubblicizza ancora generazione: il gate resta spento finché tokenizer,
lettore virtuale delle cinque parti, mapping tensori e decoder non superano
test di parità.

Il downloader e il catalogo vivono in `ModelManagement/Catalog`; questa
cartella non deve conoscere URL o politica di download.
