[English](README.md) | **Italiano**

# Runtime multi-backend

Questo livello separa l'ispezione portabile di un GGUF dalla costruzione del
decoder specifico. `RuntimeBackendFactory` legge prima `general.architecture`,
produce un `RuntimeModelDescriptor` e soltanto dopo abilita un backend concreto.

Il percorso numerico corrente resta DeepSeek V4 e continua a usare direttamente
`StreamingDecoder`: il layer Runtime non introduce dispatch dinamico nel loop di
generazione. Flash e Pro Q2 singolo vengono selezionati localmente e costruiscono
una geometria immutabile distinta; il package Pro Q4 split resta download-only.
La distribuzione Pro è in verifica e non è parte del supporto locale dichiarato.
Qwen è riconosciuto per consentire messaggi chiari e UI capability-driven. GLM
5.2 dispone già di detector e frontend nativi, ma la costruzione viene comunque
rifiutata finché il decoder Metal non supera i gate numerici end-to-end.

Le API pubbliche storiche di `InferenceService` e le variabili `DS4_*` restano
compatibili.
