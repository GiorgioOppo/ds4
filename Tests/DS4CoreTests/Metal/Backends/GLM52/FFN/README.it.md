# Test di riferimento dell'FFN GLM 5.2

Suite senza device per gli oracoli dell'FFN. Il matvec è reimplementato in modo
naïf nei test; silu e RMSNorm sono fissati da verifiche in forma chiusa
(estremi, input uniforme, righe identità — esatte) e poi riutilizzati dentro
aspettative composte, così le suite composte (dense, routed, sparse, output
head) dimostrano proprietà composizionali — ordine delle operazioni,
posizionamento dei pesi, somme — con piccole tolleranze per la deriva
dell'ordine di somma.

La suite routed dimostra che il peso del router moltiplica il mid della SwiGLU
prima della proiezione down e che i pesi di `GLM52RouterReference` (già ×2.5)
entrano invariati; la suite sparse dimostra che routed+shared è una semplice
somma. I test di rifiuto coprono dimensioni errate e input non finiti.
