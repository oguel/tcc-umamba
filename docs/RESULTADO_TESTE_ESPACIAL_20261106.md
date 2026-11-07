# Teste espacial dos checkpoints fixos — resultado recebido e conferido

## Proveniência e integridade

O arquivo `avaliacao_final_20261107T015017767631Z.zip` foi recebido após a
execução do notebook 14 pelo usuário no Kaggle. O nome usa UTC; a execução
ocorreu na noite de 06/11/2026 no horário de Brasília. O ZIP tem 85.733.172 bytes,
102 entradas e CRC íntegro. SHA-256:
`6d778d4abf1dc56f3874c21472ea73a7ce579b378a0ca9bf3e499d2a69116fc3`.

Ambos os modelos têm status de avaliação concluída. Protocolo, hashes de dados,
divisão e checkpoints correspondem ao [plano fixado antes do teste](AVALIACAO_FINAL_FIXADA_20261106.md).
A revisão efetivamente executada foi `afd587d856835aea0f7c0eafb5d065a9fd598969`.
Pesos selecionados pela validação: U-Net época 79 e U-Mamba época 87; limiar 0,5,
quatro bandas, vistas 256×256, sem augmentation ou ajuste dos pesos na avaliação.
Não houve novo treinamento. A avaliação espacial foi executada; isso não muda
retrospectivamente a classificação histórica dos treinamentos controlados.

O ZIP contém os relatórios dos dois modelos, 40 PNGs, 40 arrays NPZ e 12 GeoTIFFs
derivados. A conferência local refez as matrizes de confusão diretamente das
previsões e referências originais dos vinte patches. Reconstruiu também os
mosaicos por georreferenciamento e refez as médias das probabilidades, verificando
os GeoTIFFs de referência, previsão e probabilidade pixel a pixel, incluindo CRS,
transform e NoData. As métricas principal/secundária e municipais coincidiram
exatamente com o relatório recebido. Os tempos médios/medianos foram recalculados
das 80 medições por modelo. Nenhum modelo foi inferido novamente nesta auditoria.

## Resultado principal: agregação por patches, como na validação

| Medida | U-Net | U-Mamba |
|---|---:|---:|
| IoU café | **74,89%** | 71,11% |
| F1/Dice café | **85,64%** | 83,12% |
| Precisão café | **83,28%** | 79,55% |
| Recall café | **88,14%** | 87,02% |
| Acurácia global | **87,58%** | 85,14% |
| Verdadeiros positivos | 1.616.286 | 1.595.696 |
| Falsos positivos | 324.404 | 410.153 |
| Falsos negativos | 217.448 | 238.038 |
| Verdadeiros negativos | 2.205.086 | 2.119.337 |

A diferença de IoU é **3,78 pontos percentuais** favorável à U-Net. Ela teve
maior IoU em todos os vinte patches. Isso é uma descrição dos recortes; não é um
teste de significância com vinte amostras independentes.

Os vinte patches incluem 4.363.224 instâncias de pixels válidos. As interseções
internas ao teste são contadas novamente nesta convenção. As duas redes acertam
3.597.620 dessas instâncias e ambas erram 424.439; só a U-Net acerta 223.752,
e só a U-Mamba acerta 117.413. Não converter essas contagens em área regional única.

## Resultado secundário: pixels únicos de cada cena

Usando a regra prévia de média das probabilidades na sobreposição e limiar
somente após essa média, foram avaliados 1.704.021 pixels únicos por cena.
As 2.659.203 instâncias adicionais da agregação anterior correspondem à repetição
dos mesmos pixels nos patches. Não são novos dados geográficos.

| Medida | U-Net | U-Mamba |
|---|---:|---:|
| IoU café | **74,71%** | 71,41% |
| F1/Dice café | **85,52%** | 83,32% |
| Precisão café | **83,60%** | 80,06% |
| Recall café | **87,53%** | 86,86% |
| Acurácia global | **88,05%** | 85,97% |

A diferença de IoU é **3,30 pontos percentuais**; a vantagem permanece após
remover a repetição espacial. A agregação secundária também altera a inferência
nas bordas ao fazer média de probabilidades; sua diferença em relação à principal
não é atribuível apenas à deduplicação. Ambas foram fixadas antes dos resultados.

## Subconjuntos municipais do teste

| Recorte | Patches | IoU U-Net por patch | IoU U-Mamba por patch | IoU U-Net único/cena | IoU U-Mamba único/cena |
|---|---:|---:|---:|---:|---:|
| Juruaia/2022 | 2 | 67,30% | 66,55% | 67,36% | 66,70% |
| Nova Resende/2023 | 18 | 75,47% | 71,47% | 75,48% | 71,92% |

O teste concentra-se em Nova Resende; não demonstra desempenho uniforme em toda
a RGI. Em Juruaia, a U-Mamba tem recall maior (81,84% contra 77,89% na convenção
principal), mas também mais falsos positivos, resultando em IoU menor. Não afirmar
que a U-Net foi melhor em toda métrica e em toda condição.

## Eficiência observada na inferência

Tesla T4, Python 3.13.16, torch 2.10.0+cu126, CUDA 12.6. U-Mamba com
mamba-ssm 2.3.2.post1 e arquitetura oficial no commit
`28459e33ca03769800dd35e23c6e62491d1925b5`. Um modelo por vez, batch 1,
três aquecimentos sintéticos, 80 forwards medidos com sincronização CUDA.

| Medida | U-Net | U-Mamba |
|---|---:|---:|
| Média do forward por vista 256 | 10,88 ms | 11,51 ms |
| Mediana do forward por vista 256 | 10,85 ms | 11,22 ms |
| Pico alocado pelo PyTorch na inferência | 70,58 MiB | **48,11 MiB** |
| Parâmetros da variante executada | 1.927.297 | **723.251** |

A variante U-Mamba tem 62,47% menos parâmetros e pico alocado 31,84% menor nessa
inferência, embora não tenha sido mais rápida neste registro. Os tempos excluem
I/O, transferências, sigmoid, composição de mosaico e renderização; não são o
tempo total da aplicação. Pico alocado não é toda a VRAM do dispositivo. Uma
única passagem, sem alternância/repetições de benchmark e em tamanho fixo, não
comprova superioridade geral de velocidade ou escalabilidade O(N). As métricas de
memória do treinamento são distintas e permanecem nos relatórios anteriores.

## O que responde à pergunta do TCC

**Nas variantes e no protocolo avaliados, a U-Net obteve melhor segmentação que
a U-Mamba, tanto na validação quanto no teste espacial reservado.** Na validação,
as IoUs foram 67,30% e 64,75%; no teste principal, 74,89% e 71,11%. Os valores
maiores do teste não indicam novo aprendizado ou melhora após a validação:
são subconjuntos diferentes, com composição geográfica e de classes distintas.

A diferença observada no teste decorre de 85.749 falsos positivos e 20.590 falsos
negativos adicionais da U-Mamba na convenção principal. Isso caracteriza os erros;
não isola uma causa arquitetural. A hipótese de reduzir comissão com U-Mamba não
foi sustentada no agregado deste experimento. As máscaras binárias não permitem
atribuir toda comissão a floresta, nem omissões a sombra ou geada sem evidência
auxiliar. A pequena amostra, os recortes e as diferenças de canais, normalização
interna dos blocos e arquitetura são limites/hipóteses para discussão, não causas comprovadas. Esta
comparação não é uma ablação de adicionar Mamba a uma U-Net de capacidade igual.

A contribuição é registrar essa comparação e a troca observada entre qualidade
de segmentação e memória/parâmetros. Um resultado favorável ao baseline não
invalida o estudo e não exige retreinar para favorecer o modelo proposto.

## Limites e próximas entregas

Uma semente, uma divisão, dependência espacial e duas cenas no teste não
sustentam superioridade universal, intervalo de confiança baseado em pixels
independentes, comprovação de robustez a geada/sombra ou generalização temporal.
O teste está agora exposto: não usá-lo para selecionar novas épocas, limiar,
divisão ou hiperparâmetros. Preservar os checkpoints e os resultados recebidos.

Próximas entregas: aplicação dos dois pesos fixos em novas cenas Sentinel-2,
mapas com radiometria/validade compatíveis, discussão e redação dos resultados,
revisão com o orientador e apresentação. Novas métricas em novas cenas dependem
de referência confiável e contemporânea. Monografia e projeto de pesquisa não
foram modificados. ZIP, imagens, mapas, arrays e pesos permanecem privados.
