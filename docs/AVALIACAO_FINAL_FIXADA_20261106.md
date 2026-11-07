# Avaliação espacial com os checkpoints existentes

Atualização após execução: o teste foi concluído com os dois pesos fixados e o
ZIP recebido foi conferido. Resultados e limites em
[RESULTADO_TESTE_ESPACIAL_20261106.md](RESULTADO_TESTE_ESPACIAL_20261106.md).
As referências abaixo a teste fechado/preparação descrevem o protocolo prévio;
o teste agora está exposto e não deve ser usado para ajuste de configuração.

## Escopo fixado antes de abrir o teste

Continuação da [revisão do escopo](REVISAO_ESCOPO_E_CONCLUSAO_20261106.md).
O objetivo é comparar os pesos já recebidos; não há novos treinamentos,
reabertura da busca de hiperparâmetros ou obrigação de a U-Mamba vencer.
As duas execuções continuam identificadas como controladas, sem declaração
retrospectiva de treinamento oficial ou avaliação científica já concluída.

O avaliador `scripts.evaluate_locked_test` fixa ambos os melhores checkpoints
escolhidos pela validação, verificando SHA-256 antes de instanciar o teste:

| Modelo | Época | SHA-256 |
|---|---:|---|
| U-Net | 79 | `addd3b5f680c1495dfa73b7eae6332fe085015c8d3e7035d825dfab4adc1d2cf` |
| U-Mamba | 87 | `038a100adc397ae7482bf12550f958c9046829d5ff3d4f16d56275f69a91f9bf` |

Base NN e partição têm os mesmos hashes dos backups. O protocolo rejeita pesos,
contratos, orçamento de épocas ou atribuições espaciais diferentes; não oferece
opção de limiar, troca de split, seleção de outra época ou treinamento. A
configuração executada é 256×256, quatro bandas e quatro quadrantes fixos por
patch 512, batch 1, sem augmentation. Limiar fixo de probabilidade: 0,5; ignorar 255.

O teste inclui 20 patches: 2 Juruaia/2022 e 18 Nova Resende/2023. Seu alcance é
a zona espacial reservada, não todos os municípios nem uma nova data independente.
Não mover outros patches para o teste depois de conhecer os resultados.

## Métricas definidas previamente

**Principal:** matriz de confusão somada sobre pixels válidos dos vinte patches;
IoU café como critério principal, seguida de F1/Dice, precisão, recall, acurácia,
erro de omissão e erro de comissão. Esta agregação mantém a convenção usada na
validação, incluindo repetições por interseção interna entre patches. Não
interpretar essa contagem como área única ou vinte observações independentes.

**Secundária:** pixels únicos dentro de cada município/cena. Reconstruir as quatro
vistas na grade do patch, posicionar os patches pelo georreferenciamento e tirar
a média aritmética das probabilidades das observações válidas do mesmo pixel;
só depois aplicar o limiar 0,5. Cada pixel com referência válida entra uma vez.
Referências válidas conflitantes, grids desalinhados, resoluções diferentes,
repetição do mesmo patch e mosaicos incompletos provocam erro. Não há
interpolação nem escolha de uma previsão com base no rótulo. Pixels ignorados não
contribuem para médias ou métricas. Não combinar datas/cenas diferentes.

O relatório secundário soma observações únicas por cena, sem alegar área
geográfica única entre datas. Ambas as agregações e os subconjuntos municipais
são publicados juntos; não escolher retrospectivamente a convenção que favorece
um modelo. As métricas podem diferir porque a composição por média é uma regra
de inferência diferente, definida previamente, além da remoção de duplicatas.

## Eficiência e mapas

Avaliar um modelo por vez no mesmo processo e dispositivo CUDA, em modo eval e
inference_mode. Três entradas sintéticas aquecem cada modelo antes do cronômetro.
Medir o forward de cada vista válida com sincronização CUDA; excluir transferência
de dados, sigmoid, leitura, composição e renderização desse tempo. Guardar todas
as durações, média/mediana em ms/vista 256 e número de vistas. Isso não é tempo
total da aplicação, nem comprovação experimental de complexidade O(N).

Guardar número real de parâmetros, alocação basal e pico de tensores PyTorch após
o aquecimento; a medida não representa toda a VRAM do dispositivo. Registrar
GPU, Python, torch/CUDA, arquitetura Mamba e revisão do avaliador. Executar em
processo separado do notebook para não reutilizar modelos residentes do treino.

Exportar todos os vinte painéis por modelo e seus arrays, além de GeoTIFFs de
probabilidade média, previsão e referência de cada cena municipal. São produtos
derivados privados, com CRS, transform e NoData preservados; não alterar os originais.

## Execução em duas etapas

No Kaggle, usar [notebook 14](../notebooks/14_evaluate_locked_test.ipynb).
Ele fixa o avaliador no commit `afd587d856835aea0f7c0eafb5d065a9fd598969`,
localiza os pesos por hash, permite restaurar o backup completo auditado e
mantém a avaliação desativada por padrão. Não executar novamente o notebook de treino.

Sem `--execute`, a rotina confere bytes dos dados, contratos, pesos e metadados
espaciais. Não constrói modelo nem faz inferência sobre o teste. Uma pasta nova
recebe `protocolo.json` e `preflight.json`. No Windows, CUDA-ready será falso:
a avaliação conjunta exige Linux com torch 2.10.0+cu126/CUDA 12.6 e Mamba funcional.

```bash
python -m scripts.evaluate_locked_test --root /kaggle/input --unet-run /caminho/RUN_UNET --umamba-run /caminho/RUN_UMAMBA --output /kaggle/working/preflight_final
```

Depois de verificar o preflight e a configuração, abrir a primeira avaliação:

```bash
python -u -m scripts.evaluate_locked_test --root /kaggle/input --unet-run /caminho/RUN_UNET --umamba-run /caminho/RUN_UMAMBA --output /kaggle/working/avaliacao_final --execute
```

Kernels/arquitetura Mamba são verificados com dados sintéticos antes da primeira
inferência real. As métricas não devem ser usadas para alterar os pesos. O
avaliador não instala dependências automaticamente; preparar o stack validado
em subprocesso, como indicado no notebook de avaliação.

`status.json=completed` e `AVALIACAO_TESTE.json` indicam conclusão conjunta.
Arquivos parciais de modelo não significam que ambos foram avaliados. Falha
preserva o status e os artefatos parciais: pode ter havido exposição ao teste.
Não apagar nem repetir silenciosamente; revisar a causa e registrar qualquer
reexecução estritamente técnica. Uma pasta existente nunca é sobrescrita.
O software não pode impedir uma nova execução em outro ambiente: a única
avaliação planejada e o não uso para tuning também são compromissos metodológicos.

## Estado verificável desta preparação

Preflight local com os backups reais concluído, sem qualquer previsão do teste.
169 testes de software passaram, além de Ruff, Mypy e convenções dos 15 notebooks.
Os testes automatizados usam fixtures e inferência sintética para conferir
isolamento das divisões, substituição de checkpoint, configuração divergente,
ignorados, mosaicos, conflitos, reconstrução e exportação GeoTIFF. Não são
evidência de execução deste novo avaliador completo no Kaggle ou de kernels
Mamba nesta máquina Windows. A avaliação real conjunta continua pendente.

## Continuidade após o teste

Consolidar os resultados, inclusive se a ordem entre modelos mudar; comparar
com a validação sem escolher novos pesos. Discutir padrões de comissão/omissão
e diferenças de arquitetura, deixando hipóteses causais explicitamente limitadas.
Depois aplicar ambos os pesos em novas cenas Sentinel-2 com radiometria,
bandas e validade compatíveis. Não aplicar offset -1000 a uma fonte que já
fornece reflectância harmonizada. Sem referência contemporânea confiável,
essa etapa produz demonstração qualitativa, não métricas supervisionadas.
Monografia, projeto de pesquisa, dados e pesos originais permanecem preservados.
