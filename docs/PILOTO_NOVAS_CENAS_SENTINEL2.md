# Inferência Sentinel-2 em toda a RGI de Guaxupé

O notebook 15 usa a geometria da RGI 310044 da malha do IBGE de 2025, versionada em `data/external/ibge`. A grade de 10 m cobre todo o polígono e o relatório apresenta a cobertura observada dos nove municípios. As geometrias municipais vêm da API de malhas do IBGE. Pequenas diferenças entre as malhas regionais e municipais são registradas na contagem de pixels regionais sem município atribuído.

Se a API municipal ficar indisponível, o notebook usa a malha regional versionada para montar o mesmo mosaico. A cobertura municipal fica com valor nulo no relatório, junto ao motivo da indisponibilidade. O alvo de cobertura disponível nesse caso é o regional.

## Seleção e preparo

- O catálogo Earth Search fornece as cenas Sentinel-2 Collection 1 L2A de julho a setembro de 2024. A seleção usa a união da cobertura SCL dentro da região por data.
- O mosaico começa pela data com maior cobertura válida e pode preencher lacunas com outras datas, até quatro ao todo. O alvo operacional é 99% de cobertura regional e 98% em cada município. As lacunas restantes aparecem em NoData e no mapa de cobertura. A cobertura alcançada depende das cenas disponíveis.
- Os dois modelos recebem o mesmo mosaico B02, B03, B04 e B08, com a mesma normalização do treino. O SCL aceita classes 4, 5 e 6 e é alinhado por vizinho mais próximo. As janelas têm 256 pixels e usam limiar 0,5.
- O produto combina datas e pode reutilizar áreas geográficas do estudo original. A composição registra a origem de cada pixel e deve ser considerada ao inspecionar contornos e divergências.

## Kaggle

1. Importe `notebooks/15_infer_new_sentinel.ipynb` e habilite Internet e GPU T4.
2. Anexe o backup completo dos dois modelos como entrada privada.
3. Rode as células em ordem. A última gera o ZIP em `/kaggle/working`.

O ZIP traz probabilidades, classes, concordância, cobertura e origem por pixel em GeoTIFF, painel PNG e metadados das cenas. No mapa de concordância, 0 significa nenhuma previsão de café, 1 só U-Net, 2 só U-Mamba e 3 ambas. O valor 255 representa NoData. Em `cobertura.tif`, 0 corresponde ao exterior, 1 a pixels válidos e 2 a lacunas internas. Em `cena_por_pixel.tif`, os índices positivos remetem a `cenas_stac.json`. O relatório apresenta as contagens e proporções regionais e municipais de pixels válidos.

IoU, F1 e falsos positivos exigem uma referência independente compatível com as datas de origem dos pixels. Os produtos desta execução permitem comparar a distribuição das previsões e a concordância espacial.

- [Malhas territoriais do IBGE](https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais.html)
- [API de malhas do IBGE](https://servicodados.ibge.gov.br/api/docs/malhas?versao=4)
- [Earth Search Sentinel-2 Collection 1 L2A](https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a)
