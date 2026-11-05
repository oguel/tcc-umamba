# Plano de implementação e validação — TCC U-Mamba / RGI Guaxupé

Atualizado em 05/11/2026. O plano **não** afirma que o treino oficial já ocorreu.

## Objetivo e limites

Comparar U-Mamba (arquitetura principal) com U-Net (baseline) na segmentação
binária café / não-café nos **recortes disponíveis da RGI de Guaxupé (310044)**.
A unidade científica é o **conjunto regional agregado**: município identifica
origem, não define um treinamento separado. Uma nova cena Sentinel-2 usada
posteriormente para inferência não integra o teste cego.

Os nove municípios da RGI são Arceburgo, Cabo Verde, Guaranésia, Guaxupé,
Juruaia, Monte Belo, Muzambinho, Nova Resende e São Pedro da União.
O lote recebido cobre cinco deles. Três Pontas não integra a RGI e fica
separado para possível validação externa adicional.

## Fonte de verdade do experimento

- Drive: `MyDrive/tcc/00_DADOS_OFICIAIS_RGI_GUAXUPE/Municipios` e `Cenas`.
- Lote observado: seis conjuntos, 285 patches selecionados; 206 candidatos
  dentro da RGI, 79 de Três Pontas. A contagem deve ser revalidada integralmente.
- Imagens: GeoTIFF de 10 bandas Sentinel-2, DN uint16, EPSG:32723,
  metadados de fonte/BOA_ADD_OFFSET, dimensão 512x512, stride 300.
- Referência supervisionada: máscaras derivadas dos dados EMATER, valores
  0=não-café, 1=café, 255=ignorar. A variante Fora0_M0 também
  codifica pixels sem imagem como 0: o leitor marca estes pixels como 255.
- Não misturar variantes alternativas do mesmo patch, anos e cenas de modo
  indiscriminado. Não copiar dados de parceiros para repositórios públicos.
- Históricos antigos em `02_HISTORICO_TESTES` NÃO entram nos resultados finais.

## Ordem operacional — fases e condições de avanço

| Fase | Passo e resultado | Bloqueio para avançar |
| --- | --- | --- |
| 0 | Confirmar proveniência e licença, conferir os 5 pacotes originais | Arquivos/uso não autorizados |
| 1 | Inventariar GeoTIFF/CSV/JSON, SHA256, par imagem-máscara, variantes e duplicatas | Pares ausentes ou variantes conflitantes |
| 2 | Rodar `scripts.validate_emater_rgi` integralmente e persistir relatório | CRS, bands, DN, offset, rótulos ou nodata divergentes |
| 3 | Auditar visualmente mosaicos, bounds, proporções de café/não-café e bordas/áreas ignoradas | Artefatos/rotulagem incorretos |
| 4 | Corrigir viés de exemplos positivos, adicionar áreas comprovadamente não-café | Sem negativos confiáveis |
| 5 | Congelar versão/fingerprint do conjunto e uma área de teste espacial intocável | Vazamento/overlap de teste |
| 6 | Dividir dados restantes para seleção/validação espacial, verificar tamanho de blocos | Treino/val vazios ou sobrepostos |
| 7 | Preparar runtime GPU/PyTorch/CUDA, smoke test de U-Net e U-Mamba em 512x512 | Kernel incompatível, OOM, NaN |
| 8 | Treinar comparativamente com mesma fonte, folds, bandas, hiperparâmetros e sementes | Orçamentos de treino não comparáveis |
| 9 | Congelar melhores modelos, realizar uma avaliação no teste cego | Teste já reutilizado em tuning |
| 10 | Produzir mapas, matriz de confusão, CSV/JSON, gráficos, métricas e custo computacional | Artefatos sem rastreabilidade |
| 11 | Auditar resultados, executar análise espacial/temporal, arquivar pacote reprodutível | Ausência de código/config/manifest |
| 12 | Só depois: inferência em novas cenas Sentinel-2 e redação final | Não confundir com treino/teste |

**Importante**: o script atual reserva treino/val/test por blocos; ainda
**não implementa uma campanha completa de 5-fold com teste independente**.
Para apresentar média/desvio de 5 folds, desenvolver e executar essa rotina
sobre os dados remanescentes, sem reutilizar o teste final.

## Segurança e validade

- Nenhum pixel de NoData entra em loss ou métricas; um valor de máscara 0
  fora do mosaico não é um exemplo verdadeiro de não-café.
- Patch 512/stride300 sobrepõe outros patches; um split aleatório por arquivo
  é inválido. Excluir interseções treino↔val↔teste e evitar pontuar múltiplas
  vezes o mesmo pixel na avaliação regional final.
- Exigir exemplos negativos com rótulos confiáveis e representatividade
  de mata, pastagem, solo, água e sombras (quando disponíveis).
- Não inventar métricas. IoU/F1 dos testes antigos são apenas depuração.
- Fixar seed, código (SHA de commit), fingerprint de dados, origem e versão
  do raster, definições de classes, estatísticas normalizadas obtidas
  **apenas do treino**, VRAM e nome da GPU.
- `--evaluate-test` é exclusivamente para avaliação final. O teste fica
  fechado durante tuning, varredura de hiperparâmetros e ensaios de notebook.
- O código é portátil: Colab, Kaggle ou outro ambiente CUDA compatível;
  scripts aceitam caminhos de entrada/saída. A compatibilidade de U-Mamba
  deve ser testada em cada combinação de Python/PyTorch/CUDA/Mamba.

## Estrutura recomendada para cada rodada (não sobrescrever)

```text
MyDrive/tcc/artifacts/official_rgi/
  dataset_versions/<fingerprint>/{manifest,qa,normalization,splits}/
  runs/<run_id>/
    metadata.json          # commit, configuração, hardware, dados
    splits.json            # identidade dos patches em cada divisão
    history.csv            # uma linha por época
    metrics.json           # métrica + denominadores
    best_model.pt
    last_model.pt          # para retomar após interrupção
    predictions/           # TIFF e metadados geográficos
    figures/               # RGB/GT/Pred; TP/FP/FN; curvas; confusão
  comparisons/<study_id>/
    aggregated_metrics.csv
    paired_comparison.csv
    figures/
    REPORT.md
```

**Implementação atual:** runner salva histórico, métricas, split e melhor
checkpoint em pasta de execução única. Persistência de último checkpoint,
`predictions/`, figuras, avaliação por pixels geográficos únicos,
comparação estatística e retomada ainda são itens de desenvolvimento
antes da campanha definitiva.

## Referências de implementação

- `docs/PROTOCOLO_DEFINITIVO_RGI.md` — contrato do lote e desenho regional.
- `docs/KAGGLE.md` — *exemplo* de execução na nuvem, não plataforma obrigatória.
- `scripts/validate_emater_rgi.py` — preflight de imagem, máscara e split.
- `scripts/train_emater_rgi.py` — piloto portável com teste final fechado.
- `src/data/emater_dataset.py` — loader tif e rótulos 0/1/255.
- `src/data/emater_spatial.py` — blocos espaciais com descarte de overlap.

A monografia e o projeto de pesquisa devem ser atualizados **após** o
congelamento da metodologia e de dados realmente executados. Preservar os
documentos originais e as três referências do João para posterior consulta.
