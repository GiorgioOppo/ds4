# Contratto del modello GLM 5.2

Questa directory possiede la geometria portabile del modello GLM 5.2 e la
validazione dei metadati GGUF. Non ha alcuna dipendenza da Metal e non registra
né seleziona un backend di runtime.

`GLM52Configuration` accetta solo `general.architecture = "glm-dsa"` e l'esatta
forma GLM 5.2 a 79 blocchi implementata dal branch di riferimento `glm5.2`. Il
79° blocco contiene i tensori memorizzati per la predizione del token
successivo; l'inferenza normale usa 78 blocchi transformer. Le varianti GLM
sconosciute falliscono al caricamento invece di essere eseguite con dimensioni
incompatibili.
