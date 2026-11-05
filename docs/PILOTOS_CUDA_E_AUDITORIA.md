# Auditoria e pilotos técnicos — 05/11/2026

Nenhum treinamento científico definitivo foi executado. Os números abaixo são
diagnósticos de funcionamento e não estimam generalização. A monografia e o projeto
de pesquisa não foram modificados. As leituras usaram cópias dos dados do Drive;
nenhum original foi movido, regravado ou apagado. Dados, pesos e mapas privados
permanecem fora do GitHub.

## Evidência dos dados

A variante de entrada continua `Patches_P512_S300_NN`. As partes 001/002 foram
reunidas logicamente: 285 imagens e 285 máscaras, sem pares ausentes. São 206
pares da RGI (47 Cabo Verde, 38 Guaxupé, 27 Juruaia, 47 Muzambinho e 47 Nova
Resende) e 79 de Três Pontas, que ficam fora do experimento regional.

Foram lidas todas as dez bandas de todos os 285 GeoTIFFs e as 285 máscaras:
512×512, uint16/uint8, bandas B02/B03/B04/B08/B05/B06/B07/B8A/B11/B12,
EPSG:32723, pixels de 10m, grids alinhados, offset finito e rótulos 0/1/255.
Os 18 rasters municipais de origem também foram descomprimidos integralmente;
as imagens e máscaras dos 285 patches NN coincidiram pixel a pixel com suas
janelas de origem. Nenhuma divergência foi encontrada nessas verificações.

SHA-256 da RGI NN:
`5a0f95d581f65ed1533caea509cdc04bc7d348b9ce36cf6d91181196328f58b0`.
SHA-256 NN incluindo Três Pontas:
`5a7668bcd4a03c7defaba1835661f8563731b4e3c60c1de3e814c0efc7635087`.
O identificador depende dos bytes, município e nome do patch, sem caminhos absolutos.
O runner confere por padrão o hash da RGI configurado; `--expected-fingerprint`
permite informar explicitamente outra versão auditada.

A alternativa `Patches_P512_S300_NN_Fora0_M0` também foi lida integralmente,
separada da NN: 285 imagens e 285 máscaras. Hash incluindo Três Pontas:
`1c099048462170deef7bfcfdd8f0b114f91c0577bd64cad04266612fe641aa46`.
Foram encontrados 11.158.804 pixels sem observação que a alternativa rotula
como zero; ao aplicar a validade da imagem, as contagens efetivas coincidem com
NN. Isso confirma a importância de ignorar NoData, sem promover a alternativa
a base principal. Escopo total: 1.140 TIFFs de patches nas duas variantes e
18 TIFFs municipais de origem. HDF5 e prévias PNG não são entradas deste pipeline.

Na RGI, contagens efetivas nos patches: 12.794.976 pixels café, 32.231.342
não café e 8.975.346 ignorados. Há sobreposição entre patches: essas contagens
nunca representam área regional única. A seleção recebida não contém patches
512 inteiramente negativos; augmentation não cria diversidade geográfica.

## Divisão candidata e limites

A divisão antiga por blocos excluía Cabo Verde do treino. A proposta regional
mantém todos os cinco municípios no treino, reserva duas zonas geográficas e
purga patches que se sobreponham ou estejam a menos de 1.000m de outra divisão.
Os centros e raios permanecem os da proposta anterior; o teste não foi usado
para escolher centros, margem ou hiperparâmetros a partir de desempenho.

| Destino | Patches |
|---|---:|
| Treino | 119 |
| Validação | 20 |
| Teste reservado | 20 |
| Purgados | 47 |

Treino: Cabo Verde 15, Guaxupé 38, Juruaia 17, Muzambinho 34, Nova Resende 15.
As menores distâncias entre envelopes são 1.244,51m (treino/validação),
1.160m (treino/teste) e 22.160,18m (validação/teste). Não há área compartilhada
entre splits. A margem de 1km é conservadora e candidata; não comprova ausência
de autocorrelação, dependência por cena/data ou glebas atravessando limites.

**Representatividade ainda pendente:** validação concentra Cabo Verde/Muzambinho;
teste concentra Nova Resende/Juruaia. Corrigiu-se a ausência de município no treino
e o compartilhamento de pixels, mas isso não torna a avaliação uniforme para toda
a RGI. Revisar cobertura, negativos e possível validação espacial dentro do conjunto
de desenvolvimento antes de congelar o protocolo. Não ajustar o protocolo com
métricas do teste. A auditoria não emite aprovação científica automática.

## Execuções verificadas nesta máquina

Windows 11, Python 3.12.14, PyTorch 2.10.0+cu126, CUDA 12.6,
GTX 1650 com 4GB. Semente 42. U-Net completa com canais 32/64/128/256,
quatro bandas, AdamW e BCE+Dice com ignore=255.

| Verificação | Resultado técnico |
|---|---|
| Smoke CUDA sintético 256 | Uma atualização, saída correta, loss/gradientes/pesos finitos; 209,77 MiB de pico alocado |
| Smoke CUDA sintético 512 | Uma atualização, saída correta, loss/gradientes/pesos finitos; 815,77 MiB de pico alocado |
| Piloto real 256 | 5 patches de treino (um por município), 2 de validação, 2 épocas, 4 vistas/patch, augmentation; loss treino 1,254599→1,127542; loss validação 1,166461→1,098945; pico 225,48 MiB |
| Piloto real 512 | Mesmos 5 patches de treino e 2 de validação, 1 época, sem augmentation; loss treino 1,267796; loss validação 1,221375; pico 831,73 MiB |
| U-Mamba | Preflight recusado: o stack binário fixado exige Linux x86_64; nenhuma atualização U-Mamba executada |

Os pilotos diferem em duração e augmentation, portanto não constituem comparação
de qualidade entre 256/512. Tempos e memória descrevem apenas estas execuções.
Nenhum patch de teste foi inferido; a leitura de integridade/hashes do teste faz
parte da auditoria dos dados, sem avaliação do modelo.

## Reprodução da auditoria

Na raiz do repositório, com as duas partes disponíveis somente para leitura:

```bash
python -m scripts.validate_emater_rgi --root /dados/Municipios --output /resultados/rgi.json --expected-fingerprint 5a0f95d581f65ed1533caea509cdc04bc7d348b9ce36cf6d91181196328f58b0
python -m scripts.validate_emater_rgi --root /dados/Municipios --include-external --output /resultados/nn_com_externo.json
python -m scripts.audit_municipal_sources --root /dados/Municipios --output /resultados/origens.json
python -m scripts.validate_emater_rgi --root /dados/Municipios --include-external --variant Patches_P512_S300_NN_Fora0_M0 --output /resultados/alternativa.json
```

A última chamada audita a alternativa separadamente; ela nunca muda a variante
de treinamento. A auditoria registra avisos, estado candidato e atribuições
espaciais; aprovação científica permanece falsa. Diferença no hash esperado
encerra o comando com erro. As saídas detalhadas contêm metadados de parceiros
e devem ficar em armazenamento privado.

## Pilotos CUDA em Linux/Kaggle/Colab

Usar ambiente isolado Linux x86_64 com GPU. Instalar `requirements-runtime.txt`
e o PyTorch CUDA compatível. O stack Mamba selecionado está fixado em
`src/models/umamba_runtime.py`: torch 2.10.0+cu126, mamba-ssm 2.3.2.post1,
dynamic-network-architectures 0.3.1 e arquitetura U-Mamba no commit
`28459e33ca03769800dd35e23c6e62491d1925b5`.

Em runtime descartável Linux, preparar explicitamente as dependências:

```bash
python -c "from src.models.umamba_runtime import install_prebuilt_colab_stack; install_prebuilt_colab_stack()"
```

Se usado em notebook, reiniciar o kernel após trocar o PyTorch. Em seguida:

```bash
python -m scripts.smoke_umamba_gpu --model unet --patch-size 256 --output /kaggle/working/smoke_unet.json
python -m scripts.smoke_umamba_gpu --patch-size 256 --prepare-runtime --output /kaggle/working/smoke_umamba.json
```

`--prepare-runtime` verifica kernels e carrega a arquitetura fixada; não instala
pacotes. Executar o piloto U-Mamba somente se esse preflight passar. Na mesma
pasta do repositório, usar os mesmos argumentos para ambos os modelos:

```bash
python -m scripts.train_emater_rgi --root /kaggle/input/DATASET --model unet --device cuda --epochs 2 --patch-size 256 --augment --pilot-train-per-municipality 1 --pilot-val-limit 2 --num-workers 0 --output /kaggle/working/pilotos
python -m scripts.train_emater_rgi --root /kaggle/input/DATASET --model umamba --device cuda --epochs 2 --patch-size 256 --augment --pilot-train-per-municipality 1 --pilot-val-limit 2 --num-workers 0 --output /kaggle/working/pilotos
```

Para 512, executar primeiro o smoke correspondente e verificar VRAM. Preservar
`splits.json`, configuração, ambiente, `history.json`, `metrics.json` e checkpoint.
O runner cria diretórios únicos, marca toda execução como piloto técnico e não
oferece opção para inferir o teste. Acesso/cota GPU remota, licença/proveniência
dos rótulos, negativos adicionais, congelamento da divisão e treinamento oficial
continuam pendentes.

## Verificações de software

137 testes passaram localmente, além de Ruff, formatação de `src/tests`, Mypy e
validação dos 13 notebooks. Smokes sintéticos CPU em 256/512 com augmentation
também passaram. O único aviso da suíte refere-se à extração tar no bootstrap
legado. A coleta inicial encontrou dependência de tabelas ausente e restrições
da pasta temporária; após preparar o ambiente e usar diretório de teste gravável,
a suíte completa passou. O caminho relativo de exportação Drive foi corrigido
para usar `/` também no Windows.
