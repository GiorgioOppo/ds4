# Test dei kernel MoE

Copertura per il routing e per l'esecuzione degli esperti fusa/non fusa su
Q2_K, Q4_K, IQ2_XXS e gli altri layout dei pesi supportati.

I test dovrebbero confrontare la selezione delle route, i pesi normalizzati e
le attivazioni finali con riferimenti su CPU. Includi i confini degli esperti
attivi e conteggi di simdgroup non predefiniti dove i risultati devono restare
identici.
