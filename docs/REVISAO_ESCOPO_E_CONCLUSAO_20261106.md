# Revisão do escopo e caminho até a defesa

## Decisão atual

O usuário decidiu preservar as duas execuções concluídas de 100 épocas e seguir
para análise e conclusão do TCC. Não estão planejados novos treinamentos,
troca para 512, busca de hiperparâmetros, novas sementes ou alteração das máscaras
para tentar inverter a classificação dos modelos. A satisfação com os resultados
não substitui a avaliação independente prevista na metodologia.

Esta revisão não executa treino, inferência, avaliação do teste ou alterações
na monografia/projeto de pesquisa. Não transforma retrospectivamente execuções
controladas em treinamento oficial nem declara avaliação científica final.

## Documentos cruzados

- Projeto de Pesquisa - Miguel, texto integral: questões 1.3, objetivos 3,
  metodologia 4, resultados esperados/preliminares 5, cronograma 6 e divulgação 8.
- MonografiaTCC.docx: objetivos, metodologia 4, resultados/discussão 5,
  considerações finais 6 e tabelas ainda sem resultados.
- Relatório de Pesquisa Bibliográfica: fluxo de dados Sentinel-2 e referências.
- README, protocolo regional, divisão espacial v1, augmentation, preparação
  Kaggle, adaptadores dos modelos, runner/trainer e evidências dos dois backups.
- Diagnósticos e comparação pareada verificados em
  [COMPARACAO_VISUAL_VALIDACAO_20261106.md](COMPARACAO_VISUAL_VALIDACAO_20261106.md).

Os textos acadêmicos foram somente lidos. Expectativas e instruções contidas
neles foram tratadas como material de análise, não como autorização para editar
dados, publicar pesos privados ou produzir resultados favoráveis.

## O que o trabalho precisa responder

O objetivo geral é analisar o comportamento e a viabilidade da U-Mamba para
segmentação de café. Um objetivo específico é compará-la com a U-Net. A expectativa
de superioridade da U-Mamba é uma hipótese a confrontar com os dados; não é um
critério que obrigue a treinar novamente até a U-Mamba vencer.

| Tema previsto | Evidência disponível | O que falta |
|---|---|---|
| Dataset multiespectral e referência | Duas partes Municipios auditadas; divisão e hashes registrados | Descrever procedência efetiva dos rótulos, cobertura e limitações |
| U-Net e U-Mamba implementadas | Código, stack e kernels reais verificados | Descrever as variantes efetivamente usadas |
| Treinamento supervisionado | Duas execuções completas de 100 épocas, backups e contratos conferidos | Consolidar metodologia executada, sem repetir treinos |
| Comparação quantitativa | Validação de ambas reproduzida exatamente | Primeira avaliação dos checkpoints fixos no teste reservado |
| Análise qualitativa | Vinte painéis pareados disponíveis | Classificar e discutir padrões de erro com exemplos favoráveis e desfavoráveis |
| Eficiência computacional | Tempos por época e pico alocado registrados | Consolidar números e medir tempo de inferência comparável, sem treino |
| Sombras, mata, fenologia e geada | São questões do projeto, sem estratificação validada dessas condições | Usar apenas evidências verificáveis; registrar o que não pôde ser respondido |
| Aplicação Sentinel-2 | Fonte prevista no projeto e inferência futura prevista no repositório | Preparar aplicação dos pesos em novas cenas e mapas georreferenciados |
| Comunicação científica | Estrutura da monografia e apresentação histórica existem | Redigir resultados reais, revisão, entrega, slides e defesa |

O projeto prevê Sentinel-2 como fonte de imagens; não especifica por si só um
ensaio obrigatório em uma nova data. A aplicação futura em novas cenas aparece
explicitamente no repositório e foi confirmada como direção pelo usuário.
Sem referência contemporânea, novas cenas permitem demonstração cartográfica,
mas não novas métricas supervisionadas ou comprovação de generalização temporal.

## Resposta já sustentada: a U-Net foi melhor na validação

| Medida | U-Net | U-Mamba |
|---|---:|---:|
| Melhor época pela IoU de validação | 79 | 87 |
| IoU café | 67,30% | 64,75% |
| F1/Dice café | 80,45% | 78,60% |
| Precisão café | 81,83% | 78,49% |
| Recall café | 79,12% | 78,72% |
| Patches com maior IoU que a outra rede | 17 | 3 |
| Soma dos tempos de 100 épocas | 34min46s | 42min37s |
| Maior pico alocado registrado pelo PyTorch | 224,98 MiB | 237,21 MiB |

As IoUs agregadas nos cinco subconjuntos municipais também favorecem a U-Net.
O resultado é descritivo desta configuração, uma semente e uma divisão. Não
constitui superioridade universal, significância estatística ou resultado do
teste ainda fechado. Há sobreposições internas à validação: pixels contabilizados
nos patches não representam área única e os vinte patches não são independentes.

Os tempos incluem treino e validação por época, com sincronização CUDA; não
incluem instalação, toda a sessão ou um benchmark isolado de inferência. A memória
é a alocação de tensores medida pelo PyTorch, não toda a VRAM ocupada no dispositivo.
Uma execução em tamanho fixo não comprova escalabilidade assintótica O(N).

## Explicar a diferença sem inventar uma causa

A explicação diretamente observada é a matriz de confusão: a U-Mamba teve
36.859 falsos positivos e 3.661 falsos negativos adicionais. Portanto, sua menor
IoU vem sobretudo de mais pixels não café classificados como café, junto de menor
acerto de café. Isso não identifica sozinho qual cobertura provocou a confusão.

Não foram isolados experimentalmente o efeito de tamanho da amostra, capacidade,
normalização, inicialização, contexto espacial ou ruído do rótulo. Eles podem ser
discutidos como hipóteses/limites, não apresentados como causas demonstradas.
Fatos relevantes para essa discussão:

- Apenas 89 patches de origem no treino; quatro vistas/patch não criam quatro
  áreas geográficas independentes. A base não contém patches 512 só de não café.
- Os dois modelos receberam as mesmas regras de dados, divisão, orçamento,
  optimizer e loss, mas são implementações distintas: U-Net com canais
  32/64/128/256 e GroupNorm; UMambaEnc com 16/32/64/128 e InstanceNorm, além de
  diferenças de blocos, ativações e inicialização. Não é uma ablação isolada de Mamba.
- A entrada efetiva é 256×256; o contexto do modelo fica limitado ao recorte,
  mesmo quando seu bloco agrega todo esse contexto. Não equivale a ver uma cena inteira.
- As melhores épocas são anteriores à 100. As curvas e os históricos não provam
  que mais épocas fariam a U-Mamba ultrapassar a U-Net.
- Não é possível atribuir genericamente os erros atuais às máscaras antigas do
  MapBiomas: o lote oficial atual é outro e a procedência precisa ser descrita.
- Sem categorias auxiliares confiáveis, não afirmar que um falso positivo é
  floresta ou que um falso negativo decorre de geada/sombra apenas pelo rótulo binário.

É aceitável concluir que a vantagem esperada da U-Mamba não se confirmou na
validação deste experimento e explicar como seus erros diferiram. Determinar
causalmente por que a arquitetura obteve esse resultado exigiria outros ensaios;
eles não são necessários para relatar honestamente a comparação e não estão no
escopo atual autorizado.

## Divergências a resolver na redação final

1. Distinguir o fluxo antigo MapBiomas/GEE do lote oficial recebido. Não atribuir
   ao autor a produção de rótulos que foram fornecidos por terceiros; identificar
   origem, data, elaboração e permissões de uso com a documentação disponível.
2. Distinguir fonte 512×512/stride 300, crops de treino 256 e quadrantes fixos de
   avaliação. Remover da descrição do que foi executado transformações de brilho
   e contraste não usadas. Não afirmar aplicação de máscara de nuvem sem evidência.
3. Registrar a divisão real 89/20/20/77, a margem, a seleção dos checkpoints,
   a semente, as variantes arquiteturais e as diferenças de capacidade.
4. O código regional diverge entre monografia (310027) e projeto/repositório
   (310044). Conferir na fonte oficial IBGE e harmonizar também a lista municipal
   e o alcance territorial. Não declarar cobertura integral a partir de cinco
   subconjuntos ou generalização temporal a partir das cenas disponíveis.
5. Substituir promessas de vitória e exemplos antecipadamente favoráveis à U-Mamba
   por análise do resultado real. Rever afirmações absolutas sobre CNNs, Mamba,
   economia de memória, pioneirismo, safra e geadas; só manter o que tiver suporte.
6. Verificar referências e distinguir a fundamentação teórica do que foi medido.
   Não apresentar comparação experimental com Transformers, que não foram avaliados.
7. Preencher resultados, discussão, conclusão, resumo/abstract, figuras, tabelas
   e listas. Corrigir numeração e campos provisórios; folha de aprovação e banca
   devem conter somente informações confirmadas.
8. Publicação de pesos prevista no texto deve respeitar a procedência e permissões
   dos dados. Manter materiais privados preservados; não publicá-los automaticamente.

## Sequência restante até a apresentação

- [x] Preservar as duas execuções e conferir backups/checkpoints.
- [x] Reproduzir a validação e comparar todos os vinte patches.
- [x] Extrair curvas e custo já registrado, sem repetir execuções.
- [ ] Consolidar discussão dos erros e tabela de eficiência com seus limites.
- [ ] Formalizar com o orientador o alcance das conclusões e a manutenção dos
  checkpoints atuais, sem reabrir busca de hiperparâmetros.
- [ ] Preparar avaliador separado para a primeira avaliação do teste reservado:
  mesmos dados, hashes, bandas, normalização, limiar 0,5 e checkpoints fixos;
  ignorar 255; não usar o resultado para retreinar. Registrar sobreposições e
  definir antes da abertura a agregação por patches e/ou pixels geográficos únicos.
- [ ] Executar essa avaliação uma vez, sem novas épocas. O teste se concentra em
  Nova Resende/Juruaia; reportar esse alcance, sem extrapolar a toda a RGI.
- [ ] Preparar as novas cenas Sentinel-2: seleção e proveniência, bandas,
  resolução, NoData/nuvens, alinhamento e radiometria compatíveis. Não aplicar
  cegamente o offset do lote antigo se a nova fonte já fornecer reflectância.
- [ ] Aplicar ambos os pesos sem ajuste, recompor mapas georreferenciados e medir
  tempo de inferência em condições iguais. Novas métricas só com referência válida.
- [ ] Consolidar tabelas, curvas, mapas e limitações no pacote de resultados.
- [ ] Atualizar a redação acadêmica quando solicitado, revisar com o orientador
  e conferir normas, bibliografia, entrega e requisitos institucionais vigentes.
- [ ] Preparar slides com problema, método, resultados reais, limitações e conclusão;
  ensaiar no tempo concedido e preparar respostas da banca.
- [ ] Apresentar, realizar correções solicitadas e arquivar a versão final.

Treinos 512, novas sementes, aumento de épocas, troca de loss, novos negativos
para treinamento, ablações, ensemble e treinamento de Transformers ficam como
possíveis trabalhos futuros. Não são tarefas obrigatórias para inverter o resultado.
