# Aplicação regional Sentinel-2 de 2024

A execução no Kaggle gerou previsões dos dois checkpoints sobre 90,84% da RGI de Guaxupé (310044), com a malha do IBGE de 2025. O ZIP recebido foi `sentinel2_rgi_20261107T030320383782Z.zip`, SHA-256 `e3bfce7a85c96de10c4784ec0ff41a644b7085ec0abb7eea90dbf5dbe9d794ad`. Revisão executada `fce8437a03bacf9ee5f1712d8812951b0a53ff81`.

## Cobertura e datas

Dos 28.144.282 pixels de 10 m dentro da RGI, 25.565.154 tiveram observação válida e 2.579.128 ficaram sem previsão. A lacuna ocupa cerca de 257,91 km² e aparece principalmente a oeste no painel. O alvo regional de 99% ficou pendente. A API municipal do IBGE registrou timeout e as coberturas municipais ficaram indisponíveis.

O mosaico reúne seis cenas de quatro datas. A captura de 7 de setembro responde por 99,57% dos pixels válidos. As outras datas completaram lacunas.

| Data de captura (2024) | Pixels válidos |
|---|---:|
| 7 de setembro | 25.455.812 |
| 22 de setembro | 103.186 |
| 19 de julho | 6.152 |
| 8 de agosto | 4 |

## Previsões sobre os mesmos pixels

| Classe prevista | Pixels | Fração da área válida |
|---|---:|---:|
| Café por ambas | 1.918.758 | 7,51% |
| Café só pela U-Net | 482.181 | 1,89% |
| Café só pela U-Mamba | 1.449.810 | 5,67% |
| Não café por ambas | 21.714.405 | 84,94% |

A U-Net marcou café em 9,34% da área válida e a U-Mamba em 13,18%. As redes discordaram em 7,56% da área válida. O mosaico tem aquisição em novas datas e pode reutilizar áreas geográficas dos recortes originais. Para identificar falsos positivos, medir desempenho e examinar erros de contorno, a análise requer uma referência independente compatível com a data de cada pixel.

A auditoria verificou o CRC do ZIP, a grade comum dos sete rasters, CRS EPSG:32723, resolução de 10 m, validade e NoData. Recalculou as classes no limiar 0,5 e as contagens de concordância e proveniência. Os valores coincidiram com o relatório do Kaggle.

Uma segunda composição pode priorizar o ganho de cobertura nas lacunas em vez de escolher as datas seguintes apenas pela cobertura total. A contribuição das candidatas para o oeste ainda precisa ser medida. A execução recebida permanece como registro da primeira composição.

