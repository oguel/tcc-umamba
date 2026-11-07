# Comparação pareada U-Net/U-Mamba — validação conferida

## Evidência

Diagnóstico U-Mamba recebido do Kaggle: `diagnostico_UMAMBA_VALIDACAO.zip`.
SHA-256: `36c8c02e89068afbaf5c5a972190deb5dd77fa57fe599b5c178510e643551c93`.
CRC íntegro, 20 painéis e 20 arrays. O checkpoint corresponde ao melhor U-Mamba
do backup completo verificado: época 87, SHA-256
`038a100adc397ae7482bf12550f958c9046829d5ff3d4f16d56275f69a91f9bf`.

O diagnóstico reproduz exatamente a IoU 0,6474712884664663 registrada no treino.
O ambiente informado é Linux, Python 3.13.16, torch 2.10.0+cu126, CUDA 12.6,
Tesla T4, mamba-ssm 2.3.2.post1, arquitetura oficial no commit
`28459e33ca03769800dd35e23c6e62491d1925b5`. A U-Mamba foi inferida no Kaggle;
não foi simulada no Windows. A U-Net já havia sido reproduzida localmente.

Os arrays de probabilidades, previsões e validade das duas redes foram lidos
localmente e comparados à referência original dos mesmos 20 patches de validação.
Foram conferidos shape 512×512, probabilidade finita em [0,1], binarização >=0,5,
pixel ignorado 255 e identidade espacial. As matrizes de confusão foram refeitas
pixel a pixel e coincidiram exatamente com os dois relatórios, inclusive por
município. Dados e atribuições espaciais mantêm os hashes congelados.

Nenhuma nova inferência sobre o teste, treinamento, ajuste de limiar ou alteração
de checkpoint foi realizada nesta comparação. Novos painéis foram gerados para
todos os 20 pares, evitando apresentar somente exemplos favoráveis.

## Resultados globais de validação

| Medida | U-Net | U-Mamba |
|---|---:|---:|
| IoU café | 67,30% | 64,75% |
| F1/Dice café | 80,45% | 78,60% |
| Precisão café | 81,83% | 78,49% |
| Recall café | 79,12% | 78,72% |
| Verdadeiros positivos | 726.305 | 722.644 |
| Falsos positivos | 161.222 | 198.081 |
| Falsos negativos | 191.716 | 195.377 |
| Verdadeiros negativos | 3.029.419 | 2.992.560 |

A diferença global de IoU é 2,55 pontos percentuais. A maior diferença entre
essas matrizes está nos falsos positivos: a U-Mamba soma 36.859 a mais, além de
3.661 falsos negativos adicionais. Isso significa mais pixels classificados como
café em áreas que a referência rotula como não café. Não comprova sozinho a causa
do erro nem dispensa verificar a qualidade dos rótulos.

## IoU dos quatro patches de validação por município

| Município | U-Net | U-Mamba |
|---|---:|---:|
| Cabo Verde | 64,45% | 61,22% |
| Guaxupé | 53,15% | 50,48% |
| Juruaia | 66,33% | 63,95% |
| Muzambinho | 59,47% | 58,45% |
| Nova Resende | 78,71% | 75,61% |

A U-Net tem maior IoU agregada nos cinco subconjuntos municipais. Por patch,
vence em 17 e a U-Mamba em 3; não são vinte observações independentes, pois há
interseções geográficas dentro da validação. Os resultados descrevem os recortes
selecionados, não cada município inteiro ou toda a RGI.

Nos mesmos 4.108.662 pixels rotulados contados nos patches, ambas acertam
3.628.581 e ambas erram 266.315. Só a U-Net acerta 127.143; só a U-Mamba acerta
86.623. Isso evidencia erros parcialmente complementares, mas não justifica
declarar ensemble melhor sem um experimento próprio. Sobreposições não foram
deduplicadas e essas contagens não representam área regional única.

## Limites e continuidade

São checkpoints escolhidos pela validação, de uma semente e uma divisão, em
arquiteturas com capacidades distintas. A comparação não constitui avaliação
cega final, superioridade geral ou teste de significância. As regras e parâmetros
comuns são iguais; as transformações aleatórias efetivas não precisam coincidir
porque os modelos podem consumir o RNG de formas diferentes.

Preservar as duas execuções como baselines. Qualquer ajuste deve ser um novo
experimento de desenvolvimento, mantendo o teste fechado até fixar protocolo e
checkpoints. Monografia/projeto de pesquisa não foram alterados; dados, pesos,
imagens e probabilidades continuam privados, fora do GitHub.

## Reprodução

Com os backups e diagnósticos extraídos, usar caminhos locais ou do Kaggle:

```bash
python -m scripts.compare_validation --root /dados/Municipios --unet-run /backup/RUN_UNET --umamba-run /backup/RUN_UMAMBA --unet-diagnostic /diagnosticos/unet --umamba-diagnostic /diagnosticos/umamba --output /resultados/comparacao_pareada
```

A rotina não importa kernels Mamba nem executa modelo: confere os checkpoints,
seleciona a validação pelo manifesto e compara arrays às referências. O diretório
de saída deve ser novo. `--no-panels` realiza a mesma conferência numérica sem
renderizar imagens. Testes protegem o grid comum, os rótulos binários, a exclusão
de 255 e a distinção entre métricas agregadas e erros pareados.
