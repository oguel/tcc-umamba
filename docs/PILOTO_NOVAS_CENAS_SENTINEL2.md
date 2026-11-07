# Piloto em nova data Sentinel-2

O notebook 15 aplica os checkpoints fixos da U-Net e U-Mamba em um recorte de
10,24 km de lado, com centro aproximado em 46,74° O e 21,30° S, perto de Guaxupé.
Escolhi a janela de julho a setembro de 2024 para observar outra data na mesma
região e em meses comparáveis aos da base original. Pode haver áreas geográficas
em comum. O recorte serve para investigar transferência temporal local.

O código consulta o catálogo [Earth Search Sentinel-2 Collection 1 L2A](https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a).
Ordena as candidatas pela cobertura de nuvens informada no catálogo e pelo
identificador. Seleciona a primeira com pelo menos 90% de pixels válidos no
recorte, usando a classificação de cena SCL e a disponibilidade das quatro
bandas. O relatório guarda as candidatas examinadas e o item do catálogo.

A entrada usa B02, B03, B04 e B08, em grade de 10 m. A conversão de DN em
reflectância exige a escala e o offset declarados no produto. O SCL de 20 m é
alinhado por vizinho mais próximo. Classes 4, 5 e 6 entram na análise. Áreas
excluídas recebem zero na entrada e NoData na saída. A inferência usa janelas
fixas de 256 pixels e limiar 0,5. Compare os contornos considerando as bordas
das janelas e as áreas próximas aos pixels excluídos.

## Kaggle

1. Importe `notebooks/15_infer_new_sentinel.ipynb` e habilite Internet e GPU T4.
2. Anexe o backup completo dos dois modelos como entrada privada.
3. Rode as células em ordem. A última gera um ZIP em `/kaggle/working`.

O ZIP traz probabilidades, classes e comparação em GeoTIFF, um painel PNG,
metadados da cena e as contagens da comparação. O valor 255 representa NoData
nos mapas de classe. Na comparação, 0 indica nenhuma previsão de café, 1 só
U-Net, 2 só U-Mamba e 3 ambas. Essas categorias descrevem a concordância das
previsões. Calcular IoU, F1 ou falsos positivos nesta cena requer uma máscara
de referência independente e compatível com a data.
