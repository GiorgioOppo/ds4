# Schema dei tensori GLM 5.2

`GLM52TensorSchema` è il validatore fail-fast della directory per l'esatto
layout GGUF di GLM 5.2. Valida nomi, dimensioni, precisione densa/di
controllo, quantizzazione degli esperti routed e il blocco nextn finale senza
leggere i payload dei pesi.

`GLM52WeightMap` trasforma quella directory validata in lookup tipizzati
globali e per layer. I suoi descrittori conservano solo i metadati dei
tensori e gli offset mmap assoluti: non allocano né copiano byte di payload
GGUF.

`GLM52ExpertStreamPlanner` converte gli otto ID di esperto unici del router
in 24 letture con limiti (gate, up e down per ciascun esperto). Le slice
degli esperti sono gli intervalli contigui della terza dimensione nel tensore
GGUF. Le dimensioni di riga e di slice derivano dalla geometria dei blocchi
di ciascuna quantizzazione, e ogni moltiplicazione, somma di offset nel file
e conteggio aggregato di byte è verificato contro l'overflow. L'ordine di
rank del router è preservato e gli intervalli adiacenti non vengono
deliberatamente accorpati tra esperti diversi.

I tipi routed gate/up supportati sono IQ2_XXS, Q2_K, Q4_K e Q5_K e devono
coincidere all'interno di un layer. Il down routed accetta anche Q6_K.
Questo descrive il contratto del grafo di riferimento; la disponibilità dei
kernel resta una capacità separata del backend.
