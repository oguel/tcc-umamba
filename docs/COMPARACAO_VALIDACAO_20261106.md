# Backups U-Net/U-Mamba: comparação controlada de validação

## Integridade

Backup correto recebido: `backup_COMPLETO_UNET_UMAMBA.zip`, 53.710.616 bytes.
SHA-256: `234cb97b21931b4c3b9ec8140003b17ca5319a03cfa899b297140160a6b61a1a`.
CRC conferido. Contém os dois treinamentos, os quatro checkpoints, históricos,
configurações, ambiente e contratos. Ambos concluíram 100 épocas. O checkpoint
U-Net é idêntico ao já conferido, sem alteração de pesos entre os backups.

Foram verificados os contratos de ambos os checkpoints e de seus registros,
a sequência completa de épocas, o melhor checkpoint segundo IoU, a última época
100 e a finitude dos pesos. As métricas de cada época foram recalculadas a partir
das matrizes de confusão salvas e coincidiram com o histórico. Atribuições espaciais
e bytes do dataset correspondem aos hashes congelados. As listas de validação são
as mesmas. Os contratos diferem apenas no campo `model`.

Melhor U-Mamba (época 87), SHA-256:
`038a100adc397ae7482bf12550f958c9046829d5ff3d4f16d56275f69a91f9bf`.
Última U-Mamba (época 100), SHA-256:
`a7fd5321815cc3414ff32619363e7daa03db8389d9ef9d7dfaca38882493d31f`.

## Resultados dos checkpoints selecionados na validação

| Medida | U-Net | U-Mamba |
|---|---:|---:|
| Época selecionada | 79 | 87 |
| IoU café | 67,30% | 64,75% |
| F1/Dice café | 80,45% | 78,60% |
| Precisão café | 81,83% | 78,49% |
| Recall café | 79,12% | 78,72% |
| Soma dos tempos das épocas | 34min46s | 42min37s |

Diferença de IoU: 2,55 pontos percentuais favorável à U-Net nesta configuração.
Não interpretar como superioridade geral, significância estatística ou desempenho
final regional. Uma semente, uma divisão, diferentes arquiteturas/capacidades e
validação espacial pequena não sustentam essas conclusões. Os tempos somados dos
históricos excluem instalação e trabalho fora do cronômetro de épocas; valores de
logs arredondados podem diferir por segundos.

O modelo, a inicialização e o consumo de RNG podem resultar em transformações
amostradas diferentes mesmo com a mesma semente. As regras de augmentation, o
orçamento, a base, a divisão e os parâmetros comuns são os mesmos; não se afirma
igualdade de todos os recortes aleatórios efetivamente apresentados.

## Inferência independente e limite atual

A validação da U-Net já foi reproduzida independentemente com o checkpoint e os
dados locais, obtendo exatamente a IoU registrada. A U-Mamba teve o backup e as
matrizes salvas conferidos, mas sua inferência independente ainda requer o runtime
Linux CUDA do Kaggle. Não foi simulada no Windows nem substituída por outro modelo.

O script `scripts.inspect_validation` agora reconhece os dois modelos e carrega
os pesos correspondentes, com papéis espaciais e hashes verificados. Para U-Mamba,
exige CUDA e permite `--prepare-runtime` para preparar a mesma arquitetura oficial.
Os painéis identificam o modelo e cobrem todos os 20 patches, sem seleção somente
de exemplos favoráveis. Nenhum patch de teste é instanciado no Dataset de inferência.

No Kaggle, a partir de uma cópia da revisão de diagnóstico, sem alterar o checkout
do treinamento, executar:

```bash
python -m scripts.inspect_validation --root /kaggle/input --run-dir /kaggle/working/tcc_runs/20261106T223530560665Z_umamba_seed42 --output /kaggle/working/validacao_umamba_20261106 --device cuda --prepare-runtime
```

O diretório de saída deve ser novo. O comando só faz diagnóstico de validação:
não inicia épocas de treino, não altera os checkpoints e não oferece escolha de
split de teste. Preservar o ZIP completo antes de encerrar a sessão. Os painéis e
probabilidades ficam privados, fora do GitHub.

Monografia e projeto de pesquisa preservados. Nenhuma avaliação cega final ou
treinamento científico definitivo é declarado; são execuções controladas com
validação e limitações registradas. Novos ajustes devem ser experimentos separados,
sem substituir estas baselines nem consultar o teste para escolher parâmetros.
