# U-Net: execução controlada e diagnóstico de validação — 06/11/2026

Treinamento controlado concluído no Kaggle, sem avaliação cega final. Esta
documentação descreve somente treino/validação e não altera a monografia nem
o projeto de pesquisa. Dados, pesos, imagens e registros detalhados privados
não são enviados ao GitHub.

## Evidência recebida e conferida

Backup fornecido pelo usuário: `backup_tcc_20261106T213506969992Z.zip`.
SHA-256: `93f7b256f3d822cb0409644cff7e8f10884db20bb125e7a4ec4587a159235e90`.
CRC dos arquivos íntegro. JSONs, histórico, configuração, ambiente, checkpoints
e contratos concordam entre si; os pesos do melhor e último modelo são finitos.

Código da execução: `19d42d664ddf95114cff1115b8cf2ed71d4e4274`.
Ambiente registrado: Linux x86_64, Python 3.13.16, torch 2.10.0+cu126, CUDA 12.6,
Tesla T4. U-Net 4→1, canais 32/64/128/256, batch 1, seed 42, AdamW 1e-4,
BCE+Dice ignore=255, 256×256, augmentation geométrica apenas no treino.

Foram 100 épocas, 89 patches de treino (356 vistas/época), 20 de validação
(80 quadrantes fixos), 20 de teste reservado e 77 purgados. Histórico completo,
status concluído e última época 100 verificados. Tempos de épocas somam cerca
de 34min46s; instalação e operações fora do trecho cronometrado não estão nesse total.

SHA-256 do melhor checkpoint:
`addd3b5f680c1495dfa73b7eae6332fe085015c8d3e7035d825dfab4adc1d2cf`.
SHA-256 do último checkpoint:
`65a9617ab8e600daf3e7ec01ed236383f4af6f92a6546d3ec574eca388695ad4`.
O primeiro é da época 79; o segundo preserva o estado de retomada da época 100.

## Reprodução independente da validação

Os dados locais foram conferidos pelo hash completo da base e o manifesto
espacial foi validado contra o hash congelado v1. Caminhos Kaggle foram resolvidos
por identidade município/filename; nomes duplicados, sobreposição de papéis ou
mudança na atribuição provocam erro. O Dataset de inferência contém somente
os 20 patches de validação. A rotina não oferece opção de avaliar teste.

O melhor checkpoint reproduziu **exatamente** a IoU registrada: 0,6729763361912007.
Limiar 0,5, pixels ignorados excluídos e quadrantes de 256 reconstruídos em 512,
sem inferência sobre os patches de teste. Nenhum novo treinamento executado.

| Métrica global de validação | Valor |
|---|---:|
| IoU café | 67,30% |
| F1/Dice café | 80,45% |
| Precisão café | 81,83% |
| Recall café | 79,12% |
| Acurácia de pixels rotulados | 91,41% |

Matriz de confusão: TP 726.305; FP 161.222; TN 3.029.419; FN 191.716.
IoU/F1 são calculados da matriz acumulada, não pela média de IoUs dos patches.
Sobreposições dentro da validação não foram deduplicadas; não interpretar os
pixels como áreas regionais únicas ou observações independentes.

## Quatro patches de validação de cada município

| Município | IoU café | Precisão | Recall |
|---|---:|---:|---:|
| Cabo Verde | 64,45% | 73,33% | 84,18% |
| Guaxupé | 53,15% | 79,77% | 61,43% |
| Juruaia | 66,33% | 77,39% | 82,29% |
| Muzambinho | 59,47% | 83,61% | 67,31% |
| Nova Resende | 78,71% | 87,31% | 88,87% |

Esses valores descrevem os quatro patches selecionados por município, sem
representar o município inteiro. Guaxupé e Muzambinho apresentam maior omissão
de pixels de café em relação às respectivas referências. Isso aponta onde
investigar; não comprova a causa nem permite rotular todo erro como falha da
rede sem revisar a qualidade dos rótulos. Município, cena e ano não são fatores
independentes neste lote.

Exportados painéis de **todos os 20 patches**, cada um com RGB ilustrativo,
referência, previsão e mapa de erros (TP/FP/FN/ignorado), além das probabilidades
e máscaras de validade em NPZ. Os painéis e arrays ficam no pacote privado,
evitando escolher apenas exemplos favoráveis para representar o resultado.

## Leitura do treinamento e próxima comparação

A validação melhora rapidamente e depois oscila perto de um patamar. Depois da
época 60 o ganho máximo de IoU foi cerca de 0,17 ponto percentual. A loss de treino
continua caindo sem avanço equivalente na validação; não há degradação contínua
que permita atribuir o comportamento inteiro a sobreajuste. Mais épocas não são
uma melhoria grande demonstrada por este histórico.

Preservar esta execução como baseline. O próximo experimento comparativo é a
U-Mamba no protocolo original, condicionado a smoke CUDA aprovado. Alterações
posteriores (scheduler, amostragem, perdas, novos negativos) devem ser registradas
como novos experimentos, mantendo o teste fechado e sem sobrescrever a baseline.
Nenhum resultado de teste nem treinamento científico definitivo é declarado aqui.

## Reprodução do diagnóstico

Na revisão de diagnóstico, usando o backup extraído e as cópias NN verificadas:

```bash
python -m scripts.inspect_validation --root /dados/Municipios --run-dir /backup/20261106T213914513871Z_unet_seed42 --output /resultados/diagnostico_validacao --device cuda
```

`--device cpu` também é suportado; pequenas diferenças numéricas entre dispositivos
devem ser registradas. O diretório de saída deve ser novo; a rotina não sobrescreve
um diagnóstico anterior nem altera os checkpoints. Checks de seleção de validação,
hash e separação do teste são cobertos por testes automatizados.
