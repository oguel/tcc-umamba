# TCC — Segmentação Semântica de Lavouras Cafeeiras na RGI de Guaxupé

Bacharelado em Ciência da Computação, IFSULDEMINAS — Campus Muzambinho.

**Comparação principal:** U-Mamba versus U-Net (baseline) para segmentação binária de café e não-café. A unidade experimental é **a Região Geográfica Imediata de Guaxupé (MG, código 310044)**, e **não** um treinamento separado por município.

## Fonte oficial dos experimentos (a partir de novembro de 2026)

A fonte principal são as novas pastas `Municipios` e `Cenas` fornecidas para o TCC e armazenadas no Google Drive:

```text
MyDrive/tcc/
  00_DADOS_OFICIAIS_RGI_GUAXUPE/
    Municipios/       # downloads fracionados, GeoTIFFs e máscaras
    Cenas/            # cenas e metadados de origem
  01_DOCUMENTACAO_TCC/
  02_HISTORICO_TESTES/
  03_REFERENCIAS_MONOGRAFIA/
  data/               # legado do pipeline; preservar
  models/             # checkpoints EXPLORATÓRIOS, não definitivos
  artifacts/          # métricas EXPLORATÓRIAS, não definitivas
  repo/               # espelho/cópia do código
  secrets/            # nunca versionar credenciais
```

A variante oficial é `Patches_P512_S300_NN` (não `Fora0_M0`). As imagens e máscaras de um mesmo patch podem estar em downloads **001 e 002 distintos**; o leitor as associa pela identidade do patch, sem mover os originais.\n\nO lote contém recortes georreferenciados **512×512** com `stride=300`, e máscaras com `0=não-café`, `1=café` e `255=ignorar`. Nunca usar no lote novo a conversão PNG antiga `(mask > 127)`.

As pastas disponíveis estão organizadas por município/ano **apenas para procedência dos dados**. O treinamento agrega os recortes válidos da RGI. **Três Pontas não integra a RGI de Guaxupé** e fica reservado para possível avaliação externa, não para treino/validação/teste principal.

> Atenção sobre a origem: os metadados de cenas disponíveis apontam para Sentinel-2, enquanto as máscaras são descritas como EMATER. Registrar com precisão a proveniência de imagens e rótulos antes da redação científica. A **inferência regional em futuras cenas Sentinel-2** continua uma etapa posterior, não um resultado já obtido.

## Histórico separado

Os treinamentos exploratórios de setembro com imagens antigas foram testes de funcionamento do pipeline, não a avaliação do TCC. Não comparar seus números diretamente com futuras métricas da base definitiva. Os arquivos de teste foram preservados no Drive.

## Reprodução do dataset regional no Kaggle

1. Criar um dataset privado no Kaggle com cópia fiel de `Municipios/`, após verificar as permissões de uso dos dados.
2. Selecionar uma GPU compatível, conforme a disponibilidade e a cota da conta.
3. Clonar este repositório (ou anexá-lo como código) e instalar dependências da U-Net/dados com `pip install -r requirements-runtime.txt`.
4. No diretório-raiz do repositório, auditar os dados:

```bash
python -m scripts.validate_emater_rgi --root /kaggle/input/NOME_DO_DATASET --output /kaggle/working/auditoria_rgi.json
```

5. Após a auditoria, executar um treino piloto da U-Net:

```bash
python -m scripts.train_emater_rgi --root /kaggle/input/NOME_DO_DATASET --model unet --epochs 2 --output /kaggle/working/tcc_rgi
```

6. Para treinar a U-Mamba, primeiro verificar e preparar o runtime CUDA+Mamba fixado no projeto. **Não assumir que a wheel validada anteriormente em Colab funcionará automaticamente no Kaggle.**

A documentação técnica está em [docs/PROTOCOLO_DEFINITIVO_RGI.md](docs/PROTOCOLO_DEFINITIVO_RGI.md), [docs/AUGMENTACAO.md](docs/AUGMENTACAO.md) e [docs/KAGGLE.md](docs/KAGGLE.md).

### Primeiro teste técnico, antes do treino científico

```bash
python -m scripts.smoke_emater_rgi --synthetic --patch-size 256 --augment
python -m scripts.smoke_emater_rgi --synthetic --patch-size 512 --augment
```

O script realiza uma atualização de uma U-Net compacta em CPU com raster sintético, testa o fluxo de máscaras/augmentação e **não produz resultados científicos**. Para treinar sobre dados reais **após a auditoria espacial**, o runner aceita `--patch-size 256|512`, `--augment` e `--train-views-per-image 1..4`. Em 256, validação e teste abrangem os quatro quadrantes fixos, sem augmentação. Os GeoTIFFs do Drive nunca são alterados.

## Metodologia

- **Modelos:** U-Net e U-Mamba, saída de um canal (Café versus Não-Café).
- **Canais iniciais de imagem:** B02, B03, B04, B08 (confirmar ordem espectral no raster).
- **Pré-processamento:** DN/offset radiométrico da cena; máscaras com pixels ignorados em todos os cálculos.
- **Divisão:** blocos espaciais e exclusão de sobreposição; holdout espacial de teste independente da validação.
- **Critério:** IoU, Dice/F1, precisão, revocação, acurácia, matriz de confusão, tempo de época e VRAM.
- **Reprodutibilidade:** guardar semente, configuração, ambiente e métricas de execução.
- **Artefatos:** imagens e pesos fora do GitHub; apenas código, testes e documentação versionados.

## Estado atual

Para preparar o espaço no Kaggle e iniciar a execução, usar o
[notebook 13](notebooks/13_kaggle_training.ipynb) e o
[guia de partida e retomada](docs/INICIAR_KAGGLE.md).
Há checkpoint por época, retomada e preflight de dados/CUDA; a execução remota
ainda precisa ser validada na conta do Kaggle.

Revisão espacial v1: validação com quatro patches de cada município, teste
anterior preservado e margem de 1km. A divisão computacional congelada tem
89 treino / 20 validação / 20 teste / 77 purgados, para pilotos.
Ver [decisão, limites e hashes](docs/DIVISAO_ESPACIAL_V1.md).
Os números da continuação abaixo descrevem a divisão anterior.

Continuação técnica de 05/11/2026: auditoria NN de 285 pares e dos 18 rasters
municipais de origem, pilotos curtos da U-Net executados em CUDA local e
preflight U-Mamba pendente de Linux compatível. A divisão com margem de 1km
continua candidata, com 119/20/20 patches e 47 purgados; a representatividade
espacial ainda precisa de revisão. Evidências, limitações e comandos estão em
[docs/PILOTOS_CUDA_E_AUDITORIA.md](docs/PILOTOS_CUDA_E_AUDITORIA.md).

O pipeline está na branch principal `main`. Os scripts são independentes de plataforma e os caminhos de entrada/saída são configuráveis. A auditoria integral dos 206 pares de imagem/máscara da RGI foi feita em 05/11/2026, sem incompatibilidades detectadas nos formatos verificados. O holdout regional é **candidato**, com 142 treino, 20 validação, 20 teste fechado e 24 recortes excluídos por sobreposição. **Não há resultados oficiais de treinamento com o novo conjunto confirmados ainda.**

## Tecnologias

Python, PyTorch, Rasterio, NumPy, NVIDIA CUDA/Mamba, Kaggle e Google Drive.

## Licença

Código sob licença MIT, respeitando as licenças de componentes terceiros; dados recebidos de parceiros **não** são implicitamente públicos.
