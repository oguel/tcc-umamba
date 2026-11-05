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

A documentação técnica está em [docs/PROTOCOLO_DEFINITIVO_RGI.md](docs/PROTOCOLO_DEFINITIVO_RGI.md) e [docs/KAGGLE.md](docs/KAGGLE.md).

## Metodologia

- **Modelos:** U-Net e U-Mamba, saída de um canal (Café versus Não-Café).
- **Canais iniciais de imagem:** B02, B03, B04, B08 (confirmar ordem espectral no raster).
- **Pré-processamento:** DN/offset radiométrico da cena; máscaras com pixels ignorados em todos os cálculos.
- **Divisão:** blocos espaciais e exclusão de sobreposição; holdout espacial de teste independente da validação.
- **Critério:** IoU, Dice/F1, precisão, revocação, acurácia, matriz de confusão, tempo de época e VRAM.
- **Reprodutibilidade:** guardar semente, configuração, ambiente e métricas de execução.
- **Artefatos:** imagens e pesos fora do GitHub; apenas código, testes e documentação versionados.

## Estado atual

O pipeline oficial está na branch principal `main`. Os scripts são independentes de plataforma e os caminhos de entrada/saída são configuráveis. **Não há resultados oficiais de treinamento com o novo conjunto confirmados ainda.**

## Tecnologias

Python, PyTorch, Rasterio, NumPy, NVIDIA CUDA/Mamba, Kaggle e Google Drive.

## Licença

Código sob licença MIT, respeitando as licenças de componentes terceiros; dados recebidos de parceiros **não** são implicitamente públicos.
