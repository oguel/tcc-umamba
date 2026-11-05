# Protocolo experimental — RGI de Guaxupé (05/11/2026)

## Escopo

Atualização espacial: [v1 congelada para pilotos](DIVISAO_ESPACIAL_V1.md).
Essa versão substitui a divisão candidata descrita historicamente abaixo e
não constitui aprovação automática da campanha científica.

- **Um único experimento regional**, com todos os pares válidos da RGI agregados.
- Os nomes de municípios/anos servem como **proveniência geográfica e temporal**,
  não como seis treinamentos independentes.
- **U-Net** é o baseline, **U-Mamba** a arquitetura principal.
- Usar os mesmos folds espaciais, bandas, épocas e procedimentos de avaliação.
- Os testes de setembro foram apenas diagnóstico de software; seus números não
  são resultados definitivos da comparação.
- Sentinel-2 como fluxo de **inferência futura** é uma etapa separada. Os metadados
  de Cenas/Municipios deste lote citam cenas Sentinel-2; confirmar com a EMATER
  a origem exata da imagem e do rótulo antes de escrever a metodologia final.

## Dados recebidos

Armazenamento canônico: `MyDrive/tcc/00_DADOS_OFICIAIS_RGI_GUAXUPE/`

```text
00_DADOS_OFICIAIS_RGI_GUAXUPE/
  Municipios/
    Municipios-20261104T135253Z-1-001/Municipios/
    Municipios-20261104T135253Z-1-002/Municipios/
  Cenas/
    Cenas-20261104T135255Z-1-001/
    Cenas-20261104T135255Z-1-002/
    Cenas-20261104T135255Z-1-003/
```

As partes 001/002 são complementares. Nunca tentar emparelhar imagem/máscara dentro de uma parte isolada. Os nomes de pasta identificam partes do download; não as renomeie ou combine
automaticamente sem primeiro conferir duplicidades de patches e seu manifesto.

**Variante oficial:** `Patches_P512_S300_NN`. A alternativa `Fora0_M0` altera pixels de borda (255 → 0) e não deve ser misturada. Os downloads 001 e 002 repartem imagens e máscaras de cada município: executar descoberta global na raiz que contenha AMBAS as partes.

O conjunto observado contém patches 512 × 512 com deslocamento de 300 pixels,
imagens GeoTIFF e máscaras EMATER com **0=não-café, 1=café,
255=ignorar**. O antigo carregamento PNG com `mask > 127` **não pode ser
aplicado**, pois transforma 255 em café e 1 em não-café.

### Caminhos no Kaggle

1. Disponibilizar as pastas com arquivos GeoTIFF como um **Kaggle Dataset**,
   mantendo a hierarquia e com permissão de uso apropriada para os dados.
   Para dados não públicos, **não publicar como dataset público**.
2. Montar a base somente-leitura sob `/kaggle/input/<nome-do-dataset>/`.
3. Escrever checkpoints, logs e métricas sob `/kaggle/working/`.
4. Nunca colocar token, chave de serviço ou dados brutos no GitHub.
5. Validar no notebook o número de pares, CRS, bandas, classes e proporção
   de pixels ignorados **antes** de treinar.

### Protocolo de avaliação

- Construir primeiro todos os pares, descartando variantes repetidas da mesma
  imagem. Verificar que as duas partes `Municipios` não duplicam patches.
- Usar `region_holdout_from_bounds()` com duas zonas espaciais compactas,
  separando o teste e a validação. Esta configuração está **candidata**;
  não congelar até a auditoria e revisão de representatividade.
- Excluir do treino todos os patches que interceptem validação ou teste.
  A proposta atual também exige margem de 1km entre envelopes, configurada
  em `official_dataset.split.buffer_m`; ver [evidências e limitações](PILOTOS_CUDA_E_AUDITORIA.md).
  Os 206 patches formam um único componente de sobreposição encadeada;
  uma divisão aleatória por patch ou por componente não é apropriada.
- Reservar teste espacial independente antes da seleção de hiperparâmetros;
  folds de validação servem para comparação/ajuste, não substituem teste cego.
- Fazer análise de sensibilidade a sombras, anos e municípios **apenas depois**
  do resultado agregado, se a amostra permitir.
- Reportar IoU, Dice/F1, Precision, Recall, acurácia, matriz de confusão,
  tempo por época e pico de VRAM; métricas devem excluir pixels 255.
- Não comparar épocas desiguais de testes exploratórios com resultados finais.

## Estado de implementação

O leitor GeoTIFF e as funções de divisão espacial foram adicionados como
módulos novos para não quebrar notebooks experimentais antigos. Os 206 pares reais (imagens e máscaras) da variante NN foram inspecionados
integralmente em 05/11/2026: bandas, CRS, máscara, alinhamento, offset e
valores de pixels compatíveis, sem erros detectados nas verificações realizadas.
Há café e não-café em todo patch 512; somente 4 dos 824 quadrantes 256
são puramente negativos, e 1 é totalmente ignorado. A preparação das GPUs,
o treino oficial de U-Net e U-Mamba e a avaliação independente ainda
**não foram executados**.

Continuação: a auditoria passou a abranger os 285 pares NN, 285 pares da
alternativa isolada e os 18 TIFFs de origem. Pilotos técnicos curtos da U-Net
foram executados em CUDA local; U-Mamba aguarda Linux compatível. Isso não
altera o estado de treinamento científico acima.
