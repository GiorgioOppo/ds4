[English](README.md) | **Italiano**

# Schema tensori Laguna

`LagunaTensorSchema` valida la directory dei tensori di un GGUF Laguna S 2.1
senza leggerne i payload, port di `weights_validate_laguna_layout` del branch
di riferimento `laguna-s2.1`.

Sono accettate due ricette coerenti, identificate dal tensore di embedding (o
dall'attention Q del primo layer per viste solo-layer): il file ufficiale
Q4_K_M attuale con pesi **signal path Q8_0** e la ricetta **legacy**
precedente con attention F16 e pesi signal Q4_K/Q6_K. Gli esperti instradati
possono essere Q4_K, Q3_K o Q2_K e possono variare tra i layer (il file misto
pubblicato usa Q2_K nei primi layer MoE e Q3_K negli ultimi 27), ma le tre
proiezioni instradate di un layer devono essere coerenti; la ricetta legacy
accetta inoltre proiezioni down Q6_K accanto a gate/up Q4_K. Tutto il resto —
larghezze per-layer delle teste query 48/72, larghezze K/V della GQA, forme
delle FFN dense e condivise, tensori del router, testa di output — deve
corrispondere all'esatta geometria S 2.1.
