# Quickstart PyTorch HFL fiel ao modelo SPN

Este projeto executa uma simulação Flower/PyTorch seguindo a lógica hierárquica:

```text
cloud -> clientes -> edges -> cloud
```

A versão foi ajustada para ficar alinhada ao modelo SPN descrito:

1. A cloud inicializa o modelo.
2. A cloud seleciona os clientes.
3. A cloud envia o modelo para os clientes.
4. Cada cliente treina localmente.
5. Cada cliente envia seus pesos para o edge correspondente.
6. Cada edge espera `CLIENTS / EDGES` clientes.
7. Cada edge agrega os pesos dos seus clientes em um único update.
8. Cada edge envia um update agregado para a cloud.
9. A cloud espera todos os `EDGES`.
10. A cloud agrega os updates dos edges e fecha a rodada.

## Diferença importante desta versão

Esta versão não apenas registra eventos de edge. A agregação do modelo também é hierárquica:

```text
clientes -> agregação no edge -> agregação na cloud
```

Ou seja, a cloud não faz mais um FedAvg direto sobre todos os clientes. Primeiro, cada edge agrega seu grupo de clientes. Depois, a cloud agrega os modelos dos edges.

## Configuração principal

No `pyproject.toml`:

```toml
[tool.flwr.app.config]
num-server-rounds = 12
num-executions = 3
warmup-rounds = 1
num-supernodes = 10
num-selected-clients = 10
num-evaluate-clients = 0
num-edges = 2
local-epochs = 1
learning-rate = 0.1
batch-size = 32
log-dir = "logs"
```

- `num-supernodes`: total de clientes disponíveis no Flower.
- `num-selected-clients`: clientes selecionados por rodada.
- `num-edges`: quantidade de edges lógicos.
- `num-selected-clients` deve ser divisível por `num-edges`.
- `clients_per_edge = num-selected-clients / num-edges`.

Exemplo:

```text
num-selected-clients = 10
num-edges = 2
clients_per_edge = 5
```

Logo:

```text
edge 0: clientes 0, 1, 2, 3, 4
edge 1: clientes 5, 6, 7, 8, 9
```

## Rodando

Recomendo usar `python -m flwr` para evitar erro de venv apontando para outro Python.

```powershell
python -m pip install -U pip
python -m pip install -e .
python -m pip install -U "flwr[simulation]" torch torchvision datasets "flwr-datasets[vision]" matplotlib
python -m flwr federation simulation-config --num-supernodes 10 local
python -m flwr run . local --stream
```

## Limpando logs e processos no Windows

```powershell
taskkill /F /IM python.exe
taskkill /F /IM ray.exe
taskkill /F /IM flower-supernode.exe
taskkill /F /IM flower-superlink.exe
Remove-Item -Recurse -Force logs
```

## Arquivos gerados

```text
logs/fl_timing_exec_001.csv
logs/fl_timing_exec_002.csv
logs/fl_timing_exec_003.csv
logs/compiled_with_warmup.csv
logs/model_parameters_for_spn.csv
```

## Arquivo mais importante para preencher o SPN

Use este arquivo:

```text
logs/model_parameters_for_spn.csv
```

Ele contém apenas os parâmetros que devem ser colocados no modelo:

```text
IM
SC
ECS
T
EPPE
AGE
EC
AGC
TARGET_ROUND
```

O parâmetro `ECS` é calculado como residual calibrado:

```text
ECS = tempo_total_medio_por_rodada_hfl_reconstruido_com_warmup
      - (SC + T + EPPE + AGE + EC + AGC)
```

Isso evita usar diretamente o `envia_para_clientes` bruto do Flower/Ray, que pode não representar exatamente o mesmo conceito do SPN.

## Interpretação dos logs

As etapas principais são:

```text
inicializa_modelo
seleciona_clientes
envia_para_clientes
treinamento_por_cliente
enviar_pesos_para_edge
edge_recebe_clientes
agregar_no_edge
enviar_cloud_por_edge
cloud_recebe_edges
agregacao_de_edges_na_cloud
```

O total reconstruído para comparação com o SPN segue:

```text
seleciona_clientes
+ envia_para_clientes
+ max_por_edge(
    max_por_cliente(treinamento_por_cliente + enviar_pesos_para_edge)
    + agregar_no_edge
    + enviar_cloud_por_edge
  )
+ agregacao_de_edges_na_cloud
```

O `tempo_wallclock_medio_strategy_start_com_warmup` é apenas diagnóstico do Flower/Ray. Ele inclui overheads do runtime, escalonamento, carregamentos e outros custos que não devem ser somados diretamente aos parâmetros do SPN.
