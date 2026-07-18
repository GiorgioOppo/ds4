# Kernel GLM 5.2

Sorgenti dei kernel Metal per il backend GLM 5.2 (`glm-dsa`): router sigmoid
top-8, store/normalizzazione della KV-LoRA compatta, store delle chiavi
dell'indexer, scoring dell'indexer e il core di compact-attention a stadi
(`qk_lowrank`, `attention_indexed` sulle righe di cache selezionate,
`value_project`). Ogni kernel ha un wrapper Swift isolato e un oracolo CPU
sotto `Sources/DS4Metal/Backends/GLM52/`; nessuno è ancora collegato a un
decoder.

Gli stadi degli esperti instradati hanno ora kernel di validazione (SwiGLU
gate/up fuso e down su righe Q2_K/Q4_K/Q5_K/Q6_K, un thread per riga di output
con l'accoppiamento di elementi di riferimento), giudicati rispetto a
`GLM52FFNCPUReference` sui pesi dequantizzati; le famiglie ottimizzate per
quant arriveranno più avanti accanto a essi.

Il flusso di lavoro di modifica e l'embedding (`make embed-kernels`) sono
documentati in [`../README.md`](../README.md).
