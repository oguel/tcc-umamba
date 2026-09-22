# AGENTS.md — Regras do Projeto

## Contexto

Este repositório implementa um TCC de segmentação semântica binária de lavouras cafeeiras usando imagens Sentinel-2. A arquitetura principal é U-Mamba, com U-Net como baseline.

## Idioma

- Markdown e documentação voltada ao TCC: pt-BR.
- Identificadores de código: en-US.
- Comentários técnicos no código: pt-BR, objetivos e impessoais.

## Arquitetura

- Notebooks orquestram cada etapa.
- Lógica reutilizável fica em `src/`.
- Evitar duplicação de código entre notebooks.
- Caminhos e hiperparâmetros devem vir de `src/config.yaml`.
- Dados grandes, pesos e resultados ficam no Google Drive, não no Git.

## Reprodutibilidade

- Fixar seeds de Python, NumPy e PyTorch.
- Registrar configuração usada em cada experimento.
- Salvar métricas e checkpoints por execução.
- Evitar vazamento espacial entre treino, validação e teste.
- Não afirmar resultados que ainda não foram produzidos por execução real.

## Modelos

### U-Net

Usada como baseline para confirmar o funcionamento completo do pipeline de dados, treinamento e avaliação.

### U-Mamba

Arquitetura principal. Deve combinar representação local convolucional com blocos Mamba/SSM para dependências de longo alcance.

A adaptação deverá considerar:

- entrada RGB no dataset experimental inicial;
- entrada B2/B3/B4/B8 no dataset Sentinel-2 definitivo;
- segmentação binária com 1 canal de saída;
- uso de memória compatível com GPU disponível;
- comparação justa com U-Net.

## Dados

- RGI de Guaxupé: código IBGE `310044`.
- Sentinel-2: `COPERNICUS/S2_SR_HARMONIZED`.
- Bandas finais: B2, B3, B4, B8.
- Patch alvo: 256 × 256.
- Classes: Café e Não-Café.
- Máscaras devem ser binárias antes do cálculo de loss/métricas.

## Armazenamento

Raiz canônica:

```text
MyDrive/tcc/
```

Nunca versionar:

- tokens;
- credenciais;
- chaves JSON;
- pesos grandes;
- datasets brutos;
- exportações GeoTIFF grandes.

## Qualidade

- DRY;
- KISS;
- responsabilidade única;
- separação entre dados, modelos, treinamento e avaliação;
- `ruff`, `mypy` e `pytest` para código reutilizável.

## Ambientes

Execução principal em Google Colab ou Kaggle. Código local pode ser usado para edição, testes leves e versionamento.

Dependências específicas de Mamba/CUDA só devem ser fixadas depois que uma combinação funcional de PyTorch, CUDA, `causal-conv1d` e `mamba-ssm` for validada.

## Git

Repositório oficial deste projeto:

```text
https://github.com/oguel/tcc-umamba
```

O bootstrap e os notebooks devem buscar código somente deste repositório.
