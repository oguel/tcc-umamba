# Amanhã: montar o espaço e iniciar o treinamento

Preparação feita sem acessar Brave/Kaggle. Código e notebook estão prontos para
importação; conta, cota, GPU e compatibilidade efetiva do Kaggle ainda serão
verificadas lá. Nenhum treinamento científico definitivo foi realizado.

## Arquivos preparados

- `13_kaggle_training.ipynb`: notebook limpo, com configuração inicial, código
  versionado, instalação do stack, auditoria, smoke CUDA, treino e backup.
- `dataset_rgi_v1_privado.zip`: pacote local de 451.098.294 bytes (cerca de
  430 MiB), contendo 206 imagens NN, 206 máscaras e cinco metadados municipais.
  As duas partes são preservadas nas subpastas, sem misturar Fora0_M0 ou Três
  Pontas. Conteúdo copiado fielmente; os originais do Drive não foram alterados.
- O manifesto e os dados são privados: não entram no GitHub. O ZIP não contém
  tokens, monografia, projeto de pesquisa ou checkpoints históricos.

SHA-256 do ZIP local preparado:
`9cae74a47fcc68885f2703a5bf3cd5d9230eadc400a9c5a57c3ed751376d60b4`.
Hash dos pares e hash da divisão continuam os da versão espacial v1.

## Sequência amanhã

1. No Kaggle, criar/anexar um dataset **privado** com o ZIP ou seu conteúdo,
   usando dados com permissão de processamento externo. Nenhum upload foi
   realizado nesta preparação. Não tornar o dataset público.
2. Importar `notebooks/13_kaggle_training.ipynb`, ativar Internet e GPU disponível.
   O código usa uma GPU; duas placas não constituem memória unificada.
3. Manter `MODEL="unet"` para iniciar pela baseline. Executar as células de
   preparação até o smoke. O notebook confere arquivos, divisão e dispositivo.
4. Se o preflight e o smoke passarem, mudar `START_TRAINING=True` e executar a
   célula de treino. O padrão é uma execução nova com 100 épocas, 256×256,
   augmentation, quatro vistas por patch e batch 1. O código não oferece opção
   para inferir o teste reservado. A decisão de iniciar ocorre amanhã.
5. Antes de encerrar a sessão, preservar os outputs no Kaggle e baixar o backup
   gerado na última célula. `working` é armazenamento efêmero da sessão.

O notebook busca o código atual do PR #5, pois a `main` pode estar anterior.
Imprime o commit e registra o hash do código. Para execução reproduzível e para
retomar, substituir `CODE_REF` pelo commit impresso na primeira execução.

## U-Mamba

Em uma execução separada do mesmo notebook, usar `MODEL="umamba"` e
`SETUP_MAMBA=True` na primeira preparação. O código alinha o stack dos dois
modelos em torch 2.10.0+cu126; Mamba é instalada somente quando solicitada.
A matriz selecionada é Linux x86_64, mamba-ssm 2.3.2.post1,
dynamic-network-architectures 0.3.1 e a arquitetura oficial no commit fixado.
Foi confirmada a existência da wheel oficial Linux x86_64 CPython 3.12:
[release Mamba 2.3.2.post1](https://github.com/state-spaces/mamba/releases/tag/v2.3.2.post1).

Isso não comprova compatibilidade com a GPU do Kaggle. O preflight verifica
versões e executa kernels; o smoke precisa completar forward, backward e uma
atualização com quatro bandas antes de liberar a partida. Se falhar, conservar
o JSON de erro e começar pela U-Net, sem trocar silenciosamente de modelo.

## Salvamento e retomada

Cada época concluída grava atomicamente `last.pt`, `history.json` e
`progress.json`; o melhor modelo fica separado em `best_*.pt`. São preservados
modelo atual, otimizador, geradores de shuffle, RNG Python/NumPy/Torch/CUDA,
histórico, configuração, versões, hashes de dados/divisão/código e estado da
execução. A retomada recusa contrato incompatível. O orçamento de épocas é o
total da execução, não épocas adicionais.

Depois de restaurar a pasta completa de outputs de uma sessão anterior, copiar
a execução desejada para `/kaggle/working`, fixar `CODE_REF` no commit original
e preencher `RESUME_RUN` com essa pasta. Manter todos os outros parâmetros.
Usar `last.pt`, nunca `best_*.pt`, para continuar. A época incompleta será
repetida; épocas concluídas não são repetidas.

Os checkpoints locais não sobrevivem automaticamente à perda da sessão Kaggle.
Salvar por época evita recomeçar do zero **somente se os arquivos forem
preservados**. A célula de backup produz um ZIP local; não é cópia automática
para Drive e não existe garantia de recuperação depois de perder a sessão.
Retomar em hardware/versões diferentes pode alterar resultados numéricos;
o teste de equivalência foi feito no mesmo ambiente.

## Verificado hoje

- Preflight real da U-Net aprovou dados e divisão 89/20/20/77 em CUDA local.
- Uma época técnica com os 89 patches de treino (356 vistas) e 20 de validação
  (80 vistas) foi concluída em 51s nesta GTX 1650. Não estimar tempo Kaggle por
  esse número e não tratar o ensaio como resultado científico final.
- Uma falha foi injetada entre as épocas 1 e 2; a retomada real em CUDA
  recuperou a execução e completou a época 2. Teste reservado nunca inferido.
- Um teste automatizado compara execução contínua com interrupção/retomada,
  incluindo shuffle, dropout e RNGs: pesos e losses coincidem no mesmo ambiente.
- ZIP verificado por CRC e hash, notebook com células Python compiláveis,
  sem saídas versionadas e sem início automático de campanha.

As limitações de representatividade, cenas compartilhadas e falta de patches
inteiramente negativos permanecem documentadas em `DIVISAO_ESPACIAL_V1.md`.
Novos dados exigem nova versão; não mudar a divisão com base em métricas do
teste. Monografia e projeto de pesquisa continuam intactos.
