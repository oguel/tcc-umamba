# Inferência de 512 × 512 em Monte Belo

O notebook 16, na revisão `893c1b7b9f34ad5573d8999365ba537f5d54bd04`, aplicou os melhores pesos fixados da U-Net e U-Mamba ao mesmo patch inteiro em uma Tesla T4. Ambos os modelos foram treinados com janelas de 256 px. Monte Belo (IBGE 3143005) integra a RGI de Guaxupé e ficou fora do conjunto Emater usado no treinamento.

- Cena Sentinel-2 L2A `S2A_T23KLS_20240714T131243_L2A`, de 14/07/2024.
- Grade de 10 m em EPSG:32723, limites (359680, 7639680, 364800, 7644800). Os 262.144 pixels foram válidos e vieram de uma única cena.
- Limiar de decisão 0,5. A auditoria confirmou alinhamento de todos os TIFFs e igualdade entre cada máscara prevista e sua probabilidade limiarizada.

| Previsão | Pixels | Fração do patch |
| --- | ---: | ---: |
| U-Net prevê café | 17.825 | 6,80% |
| U-Mamba prevê café | 20.434 | 7,80% |
| Ambas preveem café | 14.856 | 5,67% |
| Só U-Net prevê café | 2.969 | 1,13% |
| Só U-Mamba prevê café | 5.578 | 2,13% |
| Ambas preveem não café | 238.741 | 91,07% |

O patch carece de máscara de referência. As contagens descrevem as previsões e não medem acurácia.

Artefato auditado `monte_belo_patch512_20261107T051431526301Z.zip`, SHA-256 `c493b3433d77dcdc43fde424c6067c3be9a23829594a81734730a082acf6a97a`.

## Continuação dos treinos

Os dois `last.pt` do backup completo foram lidos com sucesso. Ambos contêm a época 100, o histórico, o estado do otimizador e o estado aleatório. O código original fixa o orçamento no contrato de retomada. A extensão requer permitir explicitamente um novo limite de épocas sem abrir mão das verificações de dados, divisão, código e configuração.
