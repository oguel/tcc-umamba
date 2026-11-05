# Marco atual — 05/11/2026

## Alteração de fase

Os experimentos realizados em setembro com imagens antigas estão encerrados como
**testes exploratórios** (prova de execução da U-Net e U-Mamba). Nenhum resultado
exploratório será apresentado como comparação científica definitiva.

## Novo objeto experimental

- Região Geográfica Imediata de Guaxupé — código 310044.
- Fonte: novos arquivos recebidos em `Municipios` e `Cenas`.
- Recortes de 512×512, stride 300, GeoTIFF.
- Máscara 0/1/255; pixels 255 são ignorados.
- Recortes agregados em UM conjunto RGI para U-Net versus U-Mamba.
- Três Pontas é extrarregional e não entra no resultado principal.
- Kaggle é a plataforma preferencial para a próxima rodada de treinamento.
- Sentinel-2 em novas datas para inferência futura é etapa posterior.

## Infraestrutura preparada na branch de novembro

- Reorganização conservadora dos diretórios principais do Google Drive.
- Leitor GeoTIFF + filtro territorial + deduplicação.
- Validação de pares, rótulos e georreferenciamento.
- BCEDiceLoss e métricas que excluem `ignore_index=255`.
- Divisão espacial com exclusão de sobreposição.
- Scripts de auditoria e treino piloto da RGI.
- Testes automatizados básicos.
- Documentação de uso do Kaggle.

## Pendências reais (não confundir com resultado concluído)

1. Auditar todos os arquivos do lote recém-recebido (GeoTIFF/máscaras/metadados).
2. Confirmar com a EMATER quais arquivos são imagens e quais são rótulos.
3. Confirmar representatividade/cobertura da RGI e seleção de cenas.
4. Validar os scripts com dados reais e executar os testes no runtime.
5. Resolver compatibilidade da U-Mamba no Kaggle.
6. Executar teste piloto de 2 épocas para U-Net, então treino comparável.
7. Executar comparação científica com semente, folds, hiperparâmetros e métricas registradas.
8. Depois: inferência em outras cenas Sentinel-2 e redação da monografia.
