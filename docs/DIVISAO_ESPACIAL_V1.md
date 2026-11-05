# Divisão espacial v1 — congelamento computacional

Esta versão substitui a divisão candidata de 119/20/20. Congela identidades,
regras e hash para pilotos comparáveis, sem autorizar treinamento científico
automaticamente. Os resultados dos pilotos anteriores continuam identificados
pela divisão antiga e não devem ser misturados aos futuros.

## Decisão e justificativa

Preservar exatamente os 20 patches de teste já reservados. Não transferir para
teste nenhuma área usada nos pilotos anteriores. Selecionar quatro patches de
validação por município, com margem de 1km entre envelopes das três divisões.
Todos os recortes 256 e todas as transformações herdam a divisão do patch 512.

| Município | Treino | Validação | Teste | Purgados |
|---|---:|---:|---:|---:|
| Cabo Verde | 34 | 4 | 0 | 9 |
| Guaxupé | 18 | 4 | 0 | 16 |
| Juruaia | 7 | 4 | 2 | 14 |
| Muzambinho | 23 | 4 | 0 | 20 |
| Nova Resende | 7 | 4 | 18 | 18 |
| Total | 89 | 20 | 20 | 77 |

A validação passa a abranger os cinco municípios e as quatro cenas existentes.
O custo é reduzir o treino de 119 para 89 patches, preservando ao menos sete
por município. As distâncias mínimas medidas são 1.040m treino/validação,
1.244,51m treino/teste e 1.160m validação/teste. Nenhum pixel geográfico dos
envelopes é compartilhado entre grupos.

O algoritmo cria pequenos grupos próximos dentro de cada município e compara
5.000 combinações com semente 42. Escolhe a maior fração mínima de retenção no
treino, por município, e depois a maior retenção total. Usa somente geometria e
procedência municipal: não recebe rótulos, percentuais de café nem métricas de
modelo. A configuração, a ordem estável das identidades e o hash impedem uma
mudança silenciosa; alterar os dados ou os parâmetros exige nova revisão.

SHA-256 da atribuição espacial:
`1d7327d89c0abc795364a9751223fbd6725d1d0997a080e9a07cc4d98cc02f72`.
SHA-256 dos dados NN da RGI:
`5a0f95d581f65ed1533caea509cdc04bc7d348b9ce36cf6d91181196328f58b0`.
As identidades individuais ficam no manifesto privado produzido pela auditoria;
o GitHub recebe código, configuração e hashes, sem imagens ou máscaras.

## O que esta versão permite afirmar

Compara U-Net e U-Mamba nas mesmas bandas B02/B03/B04/B08, máscara 0/1/255,
normalização radiométrica do loader e divisão. Validação e teste não recebem
augmentation; usar os quatro quadrantes fixos para avaliação em 256. Manter
semente 42, BCE+Dice, AdamW com lr=1e-4 e mesmo orçamento de épocas nos pilotos
comparativos. Partir de inicialização nova; não reutilizar pesos dos pilotos
da divisão antiga. Registrar todas as mudanças de hiperparâmetros na validação.

O teste mede transferência espacial para a zona reservada, principalmente Nova
Resende, e não desempenho uniforme em toda a RGI. Ausência de sobreposição e a
margem não comprovam independência estatística, de gleba ou temporal.

Há quatro cenas, todas do tile 23KLS: Cabo Verde (05/07/2022), Guaxupé
(30/07/2022), Juruaia (25/06/2022), Muzambinho e Nova Resende (mesma cena de
13/09/2023). Cena/ano e município estão confundidos. A versão não é um holdout
temporal nem de cena: não permite alegar generalização para novas datas.

Validação é conjunto de desenvolvimento, não avaliação cega. A inspeção prévia
das máscaras para integridade foi registrada. A escolha espacial desta revisão
não utilizou resultados dos modelos. Não escolher outra divisão em função de
loss, IoU ou F1; uma revisão científica posterior deve ser versionada e explicar
quais áreas já foram usadas.

## Negativos e avaliação final

Manter neste lote os pixels não café existentes, sem fabricar negativos ou
reatribuir rótulos ignorados. A falta de patches 512 inteiramente negativos
limita a inferência em cenas completas. Novos negativos exigem rótulos confiáveis,
auditoria e versão própria do dataset; não podem entrar silenciosamente na v1.

Antes da campanha científica: verificar proveniência/autorização dos rótulos,
validar U-Mamba em Linux CUDA, fixar o orçamento final igual de treinamento e
aprovar o alcance das conclusões acima. O teste permanece fechado até congelar
hiperparâmetros e checkpoints. Monografia/projeto de pesquisa e originais do
Drive não foram alterados; nenhum novo treinamento foi executado nesta revisão.

## Reprodução

```bash
python -m scripts.validate_emater_rgi --root /dados/Municipios --output /resultados/divisao_v1.json --expected-fingerprint 5a0f95d581f65ed1533caea509cdc04bc7d348b9ce36cf6d91181196328f58b0
```

O relatório contém as atribuições, distâncias e cobertura. O código recusa hash
espacial diferente e proíbe sobrescrever parâmetros de uma divisão congelada.
139 testes passaram, junto com Ruff e Mypy. A regeneração com a lista de arquivos
invertida produziu o mesmo hash espacial.
