# Por que a U-Mamba ficou atrás? Investigação exploratória

## Pergunta e alcance

As duas redes funcionaram. A questão é por que esta variante U-Mamba teve
menor IoU que a U-Net e se a informação espectral contribuiu para a diferença.
O [teste espacial já concluído](RESULTADO_TESTE_ESPACIAL_20261106.md) foi de
74,89% versus 71,11% de IoU. Nesta investigação só se analisam os dados de treino
e as previsões de validação já arquivadas. Não se abriu novo teste, trocou limiar,
alterou os pesos ou fez novo treino/inferência.

## Trabalhos que usam Mamba em sensoriamento remoto

| Fonte primária | O que foi feito | Diferença para este TCC |
|---|---|---|
| [Ma, Li e Wang — U-Mamba, 2024](https://arxiv.org/abs/2401.04722) | Híbrido CNN/SSM para segmentação biomédica | Usamos uma adaptação da classe UMambaEnc 2D com quatro canais e configuração própria; não os pesos biomédicos |
| [Zhu et al. — Samba, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11466675/) | Segmentação de imagens de sensoriamento remoto em LoveDA, Vaihingen e Potsdam | Outra arquitetura, resolução e classes; Vaihingen usa NIR, vermelho e verde |
| [Liu et al. — CM-UNet, 2024](https://arxiv.org/html/2405.10530v1) | ResNet pré-treinada, fusão multiescala e Mamba no decodificador, com varredura em quatro direções | Não é a nossa UMambaEnc; usa outra arquitetura, dados, treinamento e avaliação |
| [Qin, Su e Zhang — SITSMamba, 2024](https://arxiv.org/html/2409.09673v1) | Classificação de culturas com séries Sentinel-2; Mamba na dimensão temporal, PASTIS com 10 bandas e MTLCC com 13 | Explora muitas datas, em vez de uma imagem por recorte; não isola o efeito de adicionar bandas ao nosso modelo |
| [Dewis et al. — OBIA-Mamba, preprint de 2025](https://arxiv.org/abs/2511.10604) | Mapeamento de cobertura com Sentinel-2 em Alberta, superpixels como unidades do Mamba | Outras classes, região e arquitetura; demonstra aplicação, não resultado transferível ao café |

Os trabalhos sustentam que Mamba pode ser usado em imagens orbitais. Os números
de outras bases e métricas não são comparáveis diretamente com a IoU da classe
café deste projeto. A comparação de recortes também precisa considerar a
resolução: 256×256 a 10 m cobre 2,56 km por lado; a quantidade de pixels, sozinha,
não estabelece o contexto geográfico disponível entre estudos.

## Bandas: hipótese plausível, sem evidência de erro grosseiro no NIR

Ambos receberam B02 (azul), B03 (verde), B04 (vermelho) e B08 (infravermelho
próximo), as quatro bandas nativas de 10 m do Sentinel-2. O carregador verifica
ordem, alinhamento, DN uint16, máscara e offset. As quatro bandas entram nas
convoluções iniciais; os blocos Mamba trabalham sobre mapas internos de
características. Não se observou uma incompatibilidade por ter quatro canais.
[Especificações Sentinel-2 do Copernicus](https://sentiwiki.copernicus.eu/web/s2-mission).

Red edge (B05/B06/B07) e SWIR (B11/B12) são candidatos a informação adicional.
Estudos de [café agroflorestal com Sentinel-2 multitemporal](https://www.mdpi.com/2072-4292/14/16/3847)
e de [estágios produtivos do café com séries Landsat/Sentinel-2](https://www.mdpi.com/2072-4292/17/18/3168)
encontraram relevância de algumas dessas bandas em seus próprios cenários.
Isso motiva testar, não demonstra benefício para a RGI nem preferência por
U-Mamba. Essas bandas têm 20 m nativos; alinhar a uma grade de 10 m não cria
detalhe espacial novo. Não trocar B08 por B8A ou adicionar canais ao aplicar
os checkpoints atuais.

## O que já foi medido sem refazer os experimentos

O script `scripts.analyze_validation_spectra` conferiu a radiometria dos 89
patches de origem do treino e dos 20 de validação. Conferiu a divisão congelada,
os hashes dos pesos e dos dados registrados nos diagnósticos e recompôs as matrizes
globais da validação. Só usou os arrays de validação e as referências originais
para localizar erros. O hash da base completa é uma verificação de integridade,
não uma análise de pixels de teste.

**Radiometria.** Em 19.511.796 instâncias válidas do treino, a B08 teve apenas
34 valores acima de 1 e 2 abaixo de 0 antes do clipping. Na validação, nenhum
dos 4.108.662 valores válidos B08 precisou desse clipping. Médias de reflectância
B08: 0,2700 no treino e 0,2684 na validação; desvio padrão 0,0576 e 0,0512.
A banda não é constante nem foi generalizadamente saturada/zerada. Isso não
verifica todas as limitações espectrais, atmosféricas ou de representatividade.

**Localização do excesso de falso café.** Em cada patch, foi identificada a
transição entre vizinhos válidos café/não café em quatro direções. A distância
euclidiana a essa transição separa a faixa de até 2 pixels do interior.
O rótulo 255 não cria borda nem entra nas taxas. Taxa de FP = FP / (FP + TN),
ou a porcentagem dos pixels de referência não café chamados de café.

| Validação, por faixa | Instâncias não café | FP U-Net | FP U-Mamba | Taxa U-Net | Taxa U-Mamba |
|---|---:|---:|---:|---:|---:|
| Até 2 pixels da transição de rótulo | 374.270 | 89.308 | 127.537 | 23,86% | 34,08% |
| Mais de 2 pixels | 2.816.371 | 71.914 | 70.544 | 2,55% | 2,50% |

A U-Mamba tem 38.229 FP a mais na faixa de borda e 1.370 a menos no interior,
somando 36.859 adicionais. Na borda também omite menos café (117.608 FN contra
123.332); no interior omite mais (77.769 contra 68.384). Isso caracteriza uma
troca entre comissão e omissão e mostra onde revisar as previsões primeiro.
Distância é da borda da **máscara de referência**, não de um limite de campo
verificado. O padrão não demonstra erro de anotação, causa espectral ou que o
Mamba produz necessariamente suavização; essas explicações exigem evidência adicional.

**Estratos espectrais.** No intervalo NDVI 0,8–1, a taxa de FP foi 4,04% na
U-Net e 8,01% na U-Mamba, para 410.464 instâncias de não café. NDVI foi calculado
de B08 e B04 após o preparo existente, com denominador zero separado. NDVI alto
não identifica floresta ou outra classe; grupos espectrais podem coincidir com
as bordas. A média RGB foi usada apenas como indicador de intensidade, sem
classificar sombra. As faixas são exploratórias; não houve ajuste por fatores
geográficos nem inferência causal. Sobreposições internas não foram deduplicadas,
e as taxas não devem produzir significância assumindo pixels independentes.

## Como investigar as hipóteses

1. **Contornos e rótulos, primeiro, sem treino.** Revisar exemplos das duas redes
   junto à imagem de referência e, quando disponível, uma fonte independente da
   mesma época. Registrar se são expansões de contorno, ligações entre talhões,
   regiões internas, pixels mistos ou rótulos duvidosos. Só atribuir erros a mata,
   sombra, geada ou pastagem com evidência adicional. Não mudar as máscaras
   originais nem selecionar casos apenas porque favorecem um modelo.
2. **Bandas, se houver um novo experimento.** Fazer comparação controlada
   2 arquiteturas × 2 conjuntos de bandas. Com a base atual é possível isolar
   RGB versus RGB+NIR; para testar bandas ausentes, comparar RGB+NIR versus
   RGB+NIR+red edge/SWIR em dados compatíveis. Fixar locais/datas, referência,
   validade comum, grade, tamanho dos recortes, treino e critério de seleção.
   Medir a mudança dentro de cada arquitetura e se a vantagem relativa muda.
   Treinar cada condição na própria entrada; zerar/remover canal na inferência
   de um checkpoint antigo não substitui essa comparação.
3. **Arquitetura/capacidade, em estudo separado.** As variantes têm 1.927.297 e
   723.251 parâmetros, além de diferenças de largura, normalização e blocos.
   Usar mesma estrutura e receita com/sem os blocos Mamba e controle de
   capacidade comparável para testar sua contribuição. O experimento atual
   compara implementações completas. Tamanho de recorte e diversidade de
   áreas devem ser fatores separados, sem trocar tudo de uma vez.
4. **Verificação externa.** Usar mais de uma semente quando viável e definir
   nova avaliação independente antes de treinar qualquer variante. O teste
   anterior já foi visto e não orienta novas escolhas. Aplicar os pesos atuais
   em outras cenas compatíveis é um estudo de transferência, não de causa
   espectral. Sem máscaras contemporâneas, reportar apenas a comparação visual.

A conclusão sustentada hoje é **excesso de falsos positivos concentrado nas
bordas da validação**. Uma limitação das bandas continua plausível, mas as
associações não separam espectro, arquitetura, amostragem e referência.

## Reprodução

```bash
python -m scripts.analyze_validation_spectra --root /dados/Municipios --unet-run /backup/RUN_UNET --unet-diagnostic /diagnosticos/unet --umamba-diagnostic /diagnosticos/umamba --output /resultados/estudo_exploratorio
```

A saída é JSON com contagens, taxas, estatísticas de bandas e hashes. O comando
não importa kernels Mamba, não executa modelos e não altera dados ou documentos
acadêmicos. O comparador portátil reproduziu todos os números da análise inicial.
173 testes de software passaram localmente, incluindo quatro dos estratos e
contagens, além de Ruff e Mypy. Imagens e arrays permanecem privados.
