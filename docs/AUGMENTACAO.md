# Augmentação online — RGI de Guaxupé

O dataset definitivo continua intacto no Google Drive, na variante
`Patches_P512_S300_NN`, com patches GeoTIFF de 512×512 e máscaras
`0=não-café`, `1=café`, `255=ignorar`.

## Transformações

O módulo `src/data/augmentation.py` cria views **em memória**:
- **256×256**: crop real dos pixels originais (resolução de 10 m mantida);
- **512×512**: patch inteiro, sem redimensionar;
- `--augment` no treino: crop aleatório em 256, flips horizontal/vertical
  e rotações `0/90/180/270` (sem interpolação, preservando DN e classes);
- **validação e teste sem augmentação**. Em 256, quatro quadrantes fixos,
  sem sobreposição, cobrem integralmente os pixels do patch de origem.

Toda transformação geométrica usa exatamente os mesmos índices/rotações
para imagem e máscara. O pixel `255` nunca vira café ou não-café.

Com `--augment` desativado, treino 256 usa quatro quadrantes determinísticos
com o padrão de `--train-views-per-image 4`; 512 usa uma view por patch.
Quando `--augment` está ativo, os crops de treino 256 podem variar a cada
época. **As views não são amostras espaciais independentes**.

## Execução

Primeiro, com um par verdadeiro **fora do teste cego**, ou com raster
sintético para confirmar a integração sem dados externos:

```bash
python -m scripts.smoke_emater_rgi --synthetic --patch-size 256 --augment
python -m scripts.smoke_emater_rgi --synthetic --patch-size 512 --augment
```

Para um par existente:

```bash
python -m scripts.smoke_emater_rgi \
  --image /CAMINHO/patch_y300_x600.tif \
  --mask /OUTRO_CAMINHO/patch_y300_x600.tif \
  --patch-size 256 --augment
```

Isso realiza **uma única atualização em uma U-Net pequena na CPU**. O
resultado é um teste de software; não mede generalização nem é uma
execução científica da U-Net/U-Mamba.

Quando a auditoria integral e o split espacial forem aprovados:

```bash
python -m scripts.train_emater_rgi \
  --root /CAMINHO/CONTENDO/AMBAS/PARTES/001_002 \
  --model unet --patch-size 256 --augment \
  --train-views-per-image 4 --epochs 2
```

Para 512, usar `--patch-size 512` e normalmente uma view de treino.
O mesmo argumento funciona para `--model umamba`, **após** validar
CUDA/PyTorch/mamba-ssm no ambiente escolhido. O teste final permanece fechado
sem `--evaluate-test`, que NÃO deve ser acionado em pilotos.

## Controles científicos

1. Primeiro separar por espaço os patches originais 512; só depois
   aplicar crops ou augmentação ao `Subset` de treino.
2. Não incluir dados do teste no treinamento, nem em normalização ou
   escolha de hiperparâmetros.
3. Em comparação 256×512, avaliar as métricas pelo mesmo conjunto de
   pixels geográficos. No runner, as quatro views fixas 256 cobrem os
   512 pixels; a avaliação georreferenciada por pixels únicos será
   implementada antes da campanha oficial.
4. Para comparar sem e com augmentação, manter fixos split, seed,
   arquitetura, critério de seleção e orçamento de treino.
5. Não acrescentar alteração de brilho/contraste arbitrária sem validar
   consistência multiespectral e radiometria com os metadados.

## Limites atuais

Há scripts e testes para carregamento, transformação e smoke técnico. A
escolha final entre 256/512 ainda depende da GPU e da validação espacial;
**não há treinamento científico novo concluído**.
