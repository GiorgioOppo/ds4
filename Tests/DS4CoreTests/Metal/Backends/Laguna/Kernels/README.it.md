[English](README.md) | **Italiano**

# Test dei kernel Laguna

Parità GPU/CPU per i kernel di decode Laguna, saltati dove Metal non è
disponibile: norm/RoPE per-testa su entrambi i tipi di blocco (YaRN e
semplice) a più posizioni, uguaglianza bit-per-bit dello store ad anello F16
e attention GQA gated di decode su finestre corte, avvolte e con riduzione
split (>256 chiavi) — tutto giudicato contro gli oracoli di `Reference/` con
le teste di produzione a 128 dimensioni.
