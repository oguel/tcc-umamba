# Marco atual do TCC — Setembro/2026

Este arquivo registra deliberadamente o ponto de parada do desenvolvimento prático atual.

## Limite definido pelo cronograma

Até setembro, o cronograma prevê:

- revisão bibliográfica e estado da arte;
- delimitação geográfica e aquisição de dados;
- pré-processamento e preparação do dataset;
- baseline U-Net;
- integração U-Mamba;
- treinamento e validação métrica;
- início da avaliação de eficiência;
- início da análise crítica.

O objetivo deste marco NÃO é concluir o TCC. O objetivo é obter um resultado parcial reproduzível com o dataset atualmente disponível e deixar a infraestrutura pronta para a continuação futura.

## Situação atual

### Estruturado

- repositório independente `oguel/tcc-umamba`;
- armazenamento canônico em `MyDrive/tcc/`;
- pipeline geoespacial herdado e adaptado para o novo repositório;
- dataset experimental localizado em `MyDrive/tcc/imagens/`;
- validação de pares imagem/máscara;
- U-Net baseline em PyTorch;
- BCE + Dice Loss;
- métricas IoU, F1, Precision, Recall, acurácia, omissão e comissão;
- registro de tempo por época e pico de VRAM;
- integração por adaptador com a classe oficial `UMambaEnc_2d`;
- notebook de diagnóstico do ambiente U-Mamba.

### Execução parcial planejada

1. Executar `09_embrapa_dataset_validation.ipynb`.
2. Executar `10_unet_baseline.ipynb`.
3. Registrar os resultados reais da U-Net.
4. Executar `11_umamba_environment.ipynb` em runtime dedicado.
5. Se o smoke test passar, executar `12_umamba_partial_training.ipynb`.

## Resultado mínimo esperado neste marco

- tabela de integridade do dataset;
- estatísticas das máscaras;
- checkpoint da U-Net;
- histórico de loss/IoU/F1;
- métricas no teste;
- figura Imagem | Ground Truth | Predição;
- tempo por época e pico de VRAM;
- diagnóstico documentado da integração U-Mamba;
- treinamento U-Mamba curto apenas se o ambiente oficial estiver funcional.

## Deliberadamente adiado

- expansão definitiva do dataset Sentinel-2;
- refinamento final de ground truth;
- treinamento longo da U-Mamba;
- busca de hiperparâmetros;
- validação espacial definitiva;
- comparação estatística final;
- XAI avançada;
- conclusões científicas finais;
- publicação de dataset/modelos;
- redação final e formatação ABNT.

Essas etapas serão retomadas quando as novas imagens e a nova validação estiverem disponíveis.
