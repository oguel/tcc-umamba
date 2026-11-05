# Guia de execução — Kaggle (RGI Guaxupé)

## Por que Kaggle?

Utilizar GPUs Kaggle é uma opção razoável diante das cotas do Colab.
**Não há garantia de cota maior para cada conta**: conferir o limite atual
no painel do próprio Kaggle. A GPU **P100 foi retirada em 15/09/2026**;
usar T4 x2 se disponível. Apesar de disponibilizar duas GPUs, o pipeline
inicial trabalha com UMA GPU; 2×T4 não equivale a 32 GB de VRAM unificada.

## Antes de ativar GPU

1. Confirmar autorização da EMATER para armazenar os dados em uma plataforma externa.\n   **Dataset privado também exige permissão de processamento externo**; não publicar sem licença.
2. Exportar `00_DADOS_OFICIAIS_RGI_GUAXUPE/Municipios` para um dataset
   com todos os arquivos originais, preservando as subpastas.
3. A base tem duas pastas exportadas `Municipios-...` e três
   `Cenas-...`. Não reunir variantes de patch como se fossem diferentes
   amostras.
4. Confirmar a origem das bandas e máscaras a partir dos metadados da cena.\n   O script varre ambas as partes; não execute apontando apenas para uma.

## Ambiente de auditoria / baseline

No Notebook do Kaggle:

```bash
!git clone https://github.com/oguel/tcc-umamba.git
%cd tcc-umamba
!git checkout main
!pip install -r requirements-runtime.txt
!python -m scripts.validate_emater_rgi --root /kaggle/input/NOME_DO_DATASET --output /kaggle/working/auditoria_rgi.json
```

O `git clone` pode ser substituído pelo código versionado previamente
anexado ao notebook, se a conexão de rede estiver indisponível.

Conferir visualmente que:
- `pairs_rgi` é maior que zero;
- `tres_pontas_2024` foi excluído;
- CRS, resoluções, canais e máscaras estão corretos;
- nenhuma máscara contém rótulos diferentes de 0, 1 e 255;
- os cinco folds espaciais têm dados de treino e validação.

Depois executar teste piloto U-Net, primeiro com épocas curtas:

```bash
!python -m scripts.train_emater_rgi --root /kaggle/input/NOME_DO_DATASET --model unet --epochs 2 --batch-size 1 --output /kaggle/working/tcc_rgi
```

Para a U-Mamba, não executar até a prova de compatibilidade da GPU,
PyTorch, CUDA e `mamba-ssm` com `ensure_umamba_runtime()`.
O módulo `src/models/umamba_runtime.py` retém a matriz de versões
validada em Colab; sua portabilidade ao Kaggle **não foi comprovada**.
Fazer diagnóstico sem GPU primeiro, instalar stack uma vez e reiniciar
o kernel se o PyTorch mudar. Não misturar versões do
`requirements-legacy-colab-2026.txt` com as dependências oficiais.

## Avaliação

O runner reserva teste espacial, validação espacial e treino sobre os
mesmos pares regionais; repita com as duas arquiteturas e mesma semente.
A avaliação do teste cego foi retirada do comando de treinamento. Será implementado um avaliador independente após congelar protocolo e pesos. O teste de 2 épocas é apenas um *smoke test*. Não publicar métricas
como resultados finais antes da campanha de treino controlada.

Salvar do diretório `/kaggle/working/tcc_rgi/` os pesos, métricas e
histórico no Google Drive, dentro de artefatos da fase oficial,
depois de conferir integridade. Não enviar esses arquivos ao GitHub.

## Observações de memória

Recortes 512×512 exigem mais VRAM que os 256×256 experimentais.
Usar batch size 1 inicialmente. Caso a U-Mamba não caiba na GPU,
documentar a limitação e projetar recortes/crops 256×256 internos
**sem confundir seus resultados com treino 512×512**.
