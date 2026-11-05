# Instruções para desenvolvimento — TCC U-Mamba

## Definição científica
- Pesquisa: segmentação binária de cafezais, **RGI de Guaxupé, código 310044**.
- Arquitetura principal: **U-Mamba**; baseline: **U-Net**.
- **Um treino/avaliação regional**, sem experimentos separados por cidade.
- As novas pastas `Municipios` e `Cenas` são o conjunto oficial.
- Os testes com imagens antigas são somente histórico e jamais podem ser anunciados como resultado final.
- **Três Pontas** está fora da RGI 310044 (é 310046). Preservar seus dados separadamente para possível teste externo.

## Contrato de dados
- Patches 512×512, stride 300, TIFF georreferenciado.
- Pares devem ser associados por município/ano, variante e nome do patch; duplicatas idênticas devem ser removidas.
- Máscara oficial `0=Não-Café, 1=Café, 255=Ignorar`.
- Excluir 255 das perdas, métricas e estatísticas supervisionadas; nunca binarizar o lote por `mask > 127`.
- Conferir metadados de radiometria, bandas, georreferenciamento e proveniência antes do treinamento.
- Particionar espacialmente, com barreira/purge entre patches sobrepostos.
- Guardar teste espacial independente para avaliação cega, depois de selecionar modelos.

## Código
- Documentação pt-BR; nomes de variáveis e funções em inglês.
- Lógica reutilizável em `src/`; scripts em `scripts/`.
- Fonte de configuração `src/config.yaml`, seção `official_dataset`. Seções legadas ficam somente para a execução histórica.
- Não colocar datasets, tokens, chaves, pesos grandes ou materiais de parceiros no GitHub.
- Adicionar testes para alterações de loader, validação, loss e split.

## Ambiente
- Kaggle é o destino preferencial para treino; não presumir compatibilidade de PyTorch/mamba-ssm.
- GPU T4 x2 conforme disponibilidade e cota da conta.
- Usar `/kaggle/input` para dados e `/kaggle/working` para resultados.
- Realizar smoke test U-Mamba + CUDA antes de treinar.
- Resultados: usar apenas medições reais, sem inventar experimentos.

## Referências acadêmicas
- Três arquivos recebidos do João estão no Drive em `03_REFERENCIAS_MONOGRAFIA`.
- Contêm material relativo ao TCC **dele**, com SegFormer, autoria e resultados diferentes.
- Reutilizar apenas normas, método de organização e referências verificadas,
  nunca transferir a autoria, o texto, o título ou resultados dele como se fossem deste TCC.
