# Trabalho de Conclusão de Curso – Bacharelado em Ciência da Computação (IFSULDEMINAS)

## Segmentação Semântica de Lavouras Cafeeiras no Sudoeste Mineiro: Aplicação da Arquitetura Híbrida U-Mamba com Dados de Sensoriamento Remoto

Projeto de TCC voltado à segmentação semântica binária de lavouras cafeeiras na Região Geográfica Imediata de Guaxupé – MG, utilizando imagens Sentinel-2 e a arquitetura híbrida U-Mamba.

## Objetivo

Analisar o comportamento e a viabilidade da arquitetura U-Mamba na segmentação das classes **Café** e **Não-Café**, comparando seu desempenho com uma U-Net utilizada como baseline convolucional.

## Área de estudo

- Região Geográfica Imediata de Guaxupé – MG.
- Código IBGE da RGI: `310044`.
- Municípios de interesse: Guaxupé, Guaranésia, Muzambinho, Cabo Verde, Juruaia, Arceburgo, Monte Santo de Minas, São Pedro da União e Jacuí.

## Dados

### Dataset experimental inicial

O projeto possui um conjunto inicial fornecido para testes, com imagens e máscaras já separadas em treino, validação e teste. Esse conjunto é usado para validar o pipeline e produzir os primeiros experimentos.

### Dataset geoespacial definitivo

A base definitiva será construída com:

- Sentinel-2 Level-2A / Surface Reflectance Harmonized;
- bandas B2, B3, B4 e B8;
- resolução espacial de 10 m;
- recorte da RGI de Guaxupé;
- máscaras de referência e posterior refinamento do ground truth;
- patches de 256 × 256 pixels para treinamento.

## Arquiteturas

- **U-Net**: baseline convolucional.
- **U-Mamba**: arquitetura principal do TCC, combinando extração local convolucional com blocos de State Space Models (SSM/Mamba) para contexto espacial de longo alcance.

## Métricas

Serão acompanhadas:

- IoU / Índice de Jaccard;
- F1-Score / Dice;
- Precision;
- Recall;
- Acurácia Global;
- matriz de confusão;
- erros de omissão e comissão;
- uso de VRAM e tempo de processamento.

## Organização

```text
tcc-umamba/
├── notebooks/                # etapas experimentais em Jupyter/Colab
├── src/                      # código Python reutilizável
│   ├── data/                 # aquisição e preparação dos dados
│   ├── models/               # U-Net e U-Mamba
│   ├── config.py
│   ├── config.yaml
│   ├── io.py
│   └── utils.py
├── tests/                    # testes unitários
├── data/external/            # pequenos dados de referência versionados
├── scripts/
├── README.md
├── PLAN.md
└── pyproject.toml
```

Os arquivos grandes não são armazenados no GitHub. O armazenamento canônico do projeto é:

```text
MyDrive/tcc/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── embrapa/
├── models/
│   ├── unet/
│   └── umamba/
├── artifacts/
│   ├── metrics/
│   ├── figures/
│   └── runs/
├── repo/
└── secrets/
```

## Pipeline

```text
Sentinel-2 / dados experimentais
        ↓
pré-processamento
        ↓
máscaras binárias
        ↓
patches
        ↓
divisão treino/validação/teste ou divisão espacial
        ↓
U-Net baseline
        ↓
U-Mamba
        ↓
IoU / F1 / Precision / Recall
        ↓
análise qualitativa e eficiência computacional
```

## Execução prevista

1. preparar o ambiente;
2. validar o dataset experimental;
3. executar uma U-Net baseline;
4. configurar o pipeline Sentinel-2 no Google Earth Engine;
5. gerar e validar patches multiespectrais;
6. preparar o ambiente compatível com U-Mamba;
7. treinar e avaliar U-Mamba;
8. comparar U-Net e U-Mamba;
9. gerar figuras, métricas e artefatos para o TCC.

## Tecnologias

- Python;
- PyTorch;
- Google Colab / Kaggle;
- Google Earth Engine;
- Google Drive;
- Rasterio / GeoPandas;
- Hugging Face Hub para publicação e armazenamento de artefatos quando necessário.

## Estado atual

A infraestrutura inicial do projeto foi criada a partir de uma base de pipeline geoespacial e está sendo adaptada para o objetivo específico deste TCC. A prioridade atual é validar os dados disponíveis com uma U-Net baseline e, em seguida, integrar a arquitetura U-Mamba.

## Licença

Partes do código-base reutilizado permanecem sob os termos da licença MIT original. Alterações e novos componentes deste projeto seguem a mesma licença, salvo indicação em contrário.
