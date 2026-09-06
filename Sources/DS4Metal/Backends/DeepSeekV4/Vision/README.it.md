# DeepSeek V4 Flash Vision Experimental

L'encoder nativo richiede due GGUF compatibili: il checkpoint linguistico
**Vision-Exp** e l'encoder BF16 separato di `antirez/deepseek-v4-gguf`.
Il normale Flash usa pesi linguistici differenti. Il caricamento controlla
metadata, nomi, dimensioni e tipi di tutti i 316 tensori del sidecar.

La pipeline usa ImageIO per orientamento EXIF e decodifica, ridimensionamento
bicubico con antialias, patch 14×14, 32 blocchi transformer, RoPE bidimensionale
e aligner 3×3. Produce embedding di 4096 elementi; il blocco linguistico include
anche i vettori appresi per inizio, padding, fine riga e fine immagine.
Il limite per immagine è di 384 token, 40 MiB di file, 40 megapixel e 16.384
pixel per dimensione.

Le moltiplicazioni avvengono tramite Metal Performance Shaders in Float32,
mantenendo gli arrotondamenti BF16 previsti dal modello. Un buffer riutilizzato
di massimo 144 MiB converte i pesi BF16 sulla GPU. Le riduzioni numeriche possono
differire dai kernel upstream; i test sintetici non sostituiscono una verifica
completa con i checkpoint reali.

`promptBlock` richiede la posizione assoluta nella conversazione: il padding
allinea il marcatore iniziale alla posizione 3 modulo 4. `tokens` ed `embeddings`
hanno lo stesso numero di righe. Il decoder deve consumare blocchi interi con
attenzione bidirezionale all'immagine e routing visuale degli esperti.

Riferimento: [antirez/ds4, commit 9ab7053](https://github.com/antirez/ds4/tree/9ab705347c1775e7599ede7eb81a6255ec7dccb5).
La licenza MIT upstream è conservata in `DeepSeekV4VisionKernels.swift`.
