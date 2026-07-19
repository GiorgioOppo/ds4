[English](README.md) | **Italiano**

# Test di esecuzione dei layer GLM 5.2

Suite GPU-contro-oracolo per la composizione dei layer: layer densi e sparsi
del primo token contro `GLM52LayerCPUReference` sui pesi dequantizzati,
uguaglianza del routing (selezione esatta, pesi entro tolleranza), la catena
forward a due layer con la testa di output e i rifiuti sulle forme di input.
Le fixture seguono la disciplina della suite MoE — Q8_0 dal quantizzatore di
test condiviso, righe di esperti Q4_K sintetiche, i byte quantizzati come
unica fonte di verità per entrambi i lati. Salta senza un device Metal.
