# Tests/DS4CoreTests

Test di parità/unitari per `DS4Core` + `DS4Metal` + `DS4Engine` (XCTest). Ogni file mira a un kernel o componente e ne confronta l'output con un riferimento (spesso una versione CPU faithful).

Aree coperte: kernel Metal (`Graph*`, `Metal*`, `MoE`), `StreamingDecoder`, GGUF loader, Half, KVCFile, downloader (`sha256Hex`, target map).

Da aggiungere quando si tocca un'invariante numerica (es. il **raw-KV ring**): un test che genera N>nSWA token con e senza la feature e asserisce l'uguaglianza.
