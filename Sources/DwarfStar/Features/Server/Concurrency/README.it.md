# Concorrenza del Server

`RequestGate.swift` è un mutex FIFO consapevole della cancellazione che
serializza l'accesso al singolo engine di inferenza condiviso. Usa un piccolo
lock esplicito così che il rilascio del gate sia sincrono nel `defer` di ogni
richiesta; lo shutdown non lascia mai un task di rilascio non tracciato. Il
livello di rete riceve e analizza completamente una richiesta prima di
acquisire il gate, impedendo ai client lenti di bloccare la generazione.

Non sostituirlo con un task senza lock e non acquisirlo durante la lettura del
corpo di una richiesta. La cancellazione rimuove i waiter in coda e ogni
percorso di errore deve rilasciare lo slot attivo.
