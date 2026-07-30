[English](README.md) | **Italiano**

# Wrapper dei kernel Laguna

Wrapper `extension MetalRuntime` sui kernel di decode di
`metal/laguna/`, nello stile GLM: `[Float]` in ingresso,
`[Float]` in uscita, un command buffer per chiamata, ogni pipeline risolta
con l'esatto nome della funzione Metal. Esistono perché i kernel possano
essere giudicati su hardware contro gli oracoli CPU di
[`../Reference/`](../Reference/README.it.md) prima di qualsiasi lavoro sul
grafo: RMSNorm per-testa + RoPE NeoX per Q e K in una sola griglia, lo store
KV ad anello F16 e l'attention GQA gated di decode (finestre corte, avvolte
e con riduzione split). Il futuro grafo residente codifica le stesse pipeline
senza la sincronizzazione per chiamata.
