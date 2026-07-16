# Runtime multi-backend

Questo livello separa l'ispezione portabile di un GGUF dalla costruzione del
decoder specifico. `RuntimeBackendFactory` legge prima `general.architecture`,
produce un `RuntimeModelDescriptor` e soltanto dopo abilita un backend concreto.

Il percorso numerico corrente resta DeepSeek V4 e continua a usare direttamente
`StreamingDecoder`: il layer Runtime non introduce dispatch dinamico nel loop di
generazione. Flash e Pro Q2 singolo vengono selezionati localmente e costruiscono
una geometria immutabile distinta; il package Pro Q4 split resta download-only.
La distribuzione Pro è in verifica e non è parte del supporto locale dichiarato.
Qwen è riconosciuto per consentire messaggi chiari e UI capability-driven, ma la
costruzione viene rifiutata finché decoder e chat format non saranno implementati
insieme.

Le API pubbliche storiche di `InferenceService` e le variabili `DS4_*` restano
compatibili.
