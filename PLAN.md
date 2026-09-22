# PLAN.md — Roteiro de Implementação

Roteiro prático do TCC **Segmentação Semântica de Lavouras Cafeeiras no Sudoeste Mineiro: Aplicação da Arquitetura Híbrida U-Mamba com Dados de Sensoriamento Remoto**.

## 1. Objetivo técnico

Construir um pipeline reproduzível para segmentação binária de **Café** e **Não-Café**, usando U-Net como baseline e U-Mamba como arquitetura principal.

## 2. Princípios

- O Google Drive em `MyDrive/tcc/` é o armazenamento canônico para dados, pesos e resultados.
- O GitHub armazena código, configuração, notebooks e testes, não grandes datasets.
- Todo resultado apresentado deve ser produzido por execução real do pipeline.
- U-Net e U-Mamba devem usar, sempre que possível, os mesmos dados, divisões, métricas e protocolo de avaliação.
- Credenciais nunca são versionadas.

## 3. Área e dados

- RGI de Guaxupé – MG.
- Código IBGE correto: `310044`.
- Sentinel-2 SR Harmonized.
- Bandas: B2, B3, B4 e B8.
- Resolução: 10 m.
- Patch principal: 256 × 256.
- Classes: Café = 1; Não-Café = 0.

## 4. Fases

### Fase A — Dataset experimental inicial

Objetivo: provar que o pipeline de treinamento e inferência funciona antes de depender da aquisição completa via GEE.

- validar imagens e máscaras existentes;
- binarizar máscaras;
- conferir pares imagem/máscara;
- calcular proporção de classe;
- executar U-Net baseline;
- salvar pesos, métricas e previsões no Drive.

### Fase B — Aquisição e preparação geoespacial

- configurar Google Cloud / Earth Engine;
- obter geometria da RGI;
- montar mosaicos Sentinel-2;
- aplicar máscara de nuvens;
- obter/refinar ground truth;
- alinhar raster e máscara;
- normalizar as bandas;
- gerar patches;
- gerar manifesto de dados;
- definir divisão espacial para evitar vazamento.

### Fase C — Baseline U-Net

- implementar Dataset/DataLoader;
- implementar U-Net;
- usar BCE + Dice Loss inicialmente;
- AdamW;
- acompanhar IoU, F1, Precision e Recall;
- salvar melhor checkpoint;
- produzir previsões qualitativas.

### Fase D — U-Mamba

- validar combinação Python/PyTorch/CUDA;
- instalar dependências Mamba compatíveis;
- adaptar U-Mamba para entrada multiespectral de 4 canais;
- saída binária de 1 canal;
- treinar com o mesmo protocolo do baseline;
- registrar uso de VRAM e tempo por época/inferência.

### Fase E — Avaliação

- matriz de confusão;
- IoU;
- F1/Dice;
- Precision;
- Recall;
- Acurácia Global;
- erros de omissão e comissão;
- comparação de custo computacional;
- análise visual de regiões com sombra, mata e bordas de talhões.

## 5. Notebooks

### Existentes / preparação de dados

- `00_setup_environment.ipynb`
- `01_gee_sentinel2_acquisition.ipynb`
- `02_gee_reference_masks.ipynb`
- `03_mask_sources_comparison.ipynb`
- `04_mask_finalization.ipynb`
- `05_preprocessing.ipynb`
- `06_patch_generation.ipynb`
- `07_spatial_kfold_split.ipynb`
- `08_dataset_eda.ipynb`

### A criar

- `09_embrapa_dataset_validation.ipynb`
- `10_unet_baseline.ipynb`
- `11_umamba_environment.ipynb`
- `12_umamba_training.ipynb`
- `13_model_evaluation.ipynb`
- `14_efficiency_analysis.ipynb`
- `15_qualitative_results.ipynb`

## 6. Estrutura de código alvo

```text
src/
├── data/
│   ├── gee_client.py
│   ├── preprocessing.py
│   ├── patch_generation.py
│   ├── spatial_split.py
│   └── dataset.py
├── models/
│   ├── unet.py
│   └── umamba.py
├── losses.py
├── metrics.py
├── trainer.py
├── config.py
├── config.yaml
├── io.py
└── utils.py
```

## 7. Ordem imediata de trabalho

1. limpar referências herdadas do projeto anterior;
2. validar o dataset experimental;
3. executar U-Net baseline;
4. gerar primeiros resultados reais;
5. configurar Earth Engine;
6. consolidar o dataset Sentinel-2;
7. integrar U-Mamba.

## 8. Critério de sucesso da primeira etapa

A primeira etapa está concluída quando for possível executar um notebook do início ao fim e obter:

- loss de treinamento/validação;
- IoU e F1 em teste;
- checkpoint salvo;
- figura contendo imagem original, ground truth e previsão;
- todos os artefatos gravados em `MyDrive/tcc/`.
