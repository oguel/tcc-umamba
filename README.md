# Segmentação semântica de lavouras cafeeiras no Sudoeste Mineiro

TCC de Ciência da Computação, IFSULDEMINAS – Campus Muzambinho. O estudo avalia a U-Mamba em imagens Sentinel-2 da Região Geográfica Imediata de Guaxupé (MG), com a U-Net como referência comparativa.

## Qual execução consultar

| Etapa | Entrada ou código | Evidência a conferir |
|---|---|---|
| Base principal | No Drive do projeto, abra `00_DADOS_OFICIAIS_RGI_GUAXUPE`. Use as duas partes de `Municipios` e a variante `Patches_P512_S300_NN`. | 206 pares na RGI, com 89 para treino, 20 para validação, 20 para teste e 77 excluídos. |
| Treinamento inicial | [Notebook 13 na revisão usada](https://github.com/oguel/tcc-umamba/blob/19d42d664ddf95114cff1115b8cf2ed71d4e4274/notebooks/13_kaggle_training.ipynb) | Os pesos selecionados foram das épocas 79 (U-Net) e 87 (U-Mamba). |
| Teste espacial com máscaras | [Notebook 14](https://github.com/oguel/tcc-umamba/blob/feat/avaliacao-teste-fixado-20261106/notebooks/14_evaluate_locked_test.ipynb) e [revisão executada](https://github.com/oguel/tcc-umamba/tree/afd587d856835aea0f7c0eafb5d065a9fd598969) | IoU de café de 74,89% (U-Net) e 71,11% (U-Mamba). Conferir `protocolo.json` e `resultado_modelo.json` no backup do teste. |
| Extensão até 300 épocas | Backups das duas redes na pasta `models/Backups - base original - 300 épocas - 2026-11-09` do Drive do projeto | Curvas e máximos na validação. O teste espacial usou os pesos das épocas 79 e 87. |
| Aplicação em 2024 | [Notebook 15 na revisão usada](https://github.com/oguel/tcc-umamba/blob/fce8437a03bacf9ee5f1712d8812951b0a53ff81/notebooks/15_infer_new_sentinel.ipynb) | Previsões exploratórias sem máscaras contemporâneas. Consultar `04_GALERIA_TCC/05_SENTINEL2_RGI`. |

Para reproduzir, selecione a revisão da etapa, use cópias dos dados no Kaggle e confira os identificadores da divisão e dos pesos no protocolo antes de executar. A U-Mamba requer Linux, GPU CUDA e o ambiente indicado pelo notebook. O Apêndice A da monografia identifica os backups, as versões executadas e os resultados correspondentes.

## Materiais anteriores

Os [notebooks 00–12 de setembro](https://github.com/oguel/tcc-umamba/tree/df37706849aec4a9f67a7ae8d413a49b9ca58f89/notebooks) documentam a preparação exploratória, incluindo o MapBiomas, e os primeiros testes técnicos. Os guias [Kaggle](docs/KAGGLE.md) e [protocolo inicial](docs/PROTOCOLO_DEFINITIVO_RGI.md) descrevem a preparação anterior às execuções registradas acima. Consulte os notebooks das revisões indicadas para os experimentos da monografia.

Dados, pesos e credenciais ficam fora do GitHub. Código sob licença MIT.
