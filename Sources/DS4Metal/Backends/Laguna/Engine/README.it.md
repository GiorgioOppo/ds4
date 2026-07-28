[English](README.md) | **Italiano**

# Gate del motore Laguna

`LagunaRuntimeGate.enabled` è l'unico interruttore di abilitazione di Laguna
S 2.1: selezione del backend, disponibilità a catalogo e dispatch della demo
dipendono tutti da questa sola costante. Resta `false` finché il decoder Metal
del branch di riferimento `laguna-s2.1` non è portato e il gate end-to-end di
parità dei logits non passa su pesi reali su hardware.
