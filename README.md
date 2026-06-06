# Quickstart PyTorch HFL Aligned with the SPN Model

This project runs a Flower/PyTorch simulation following the hierarchical logic:

```text
cloud -> clients -> edges -> cloud
```

This version was adjusted to align with the described SPN model:

1. The cloud initializes the model.
2. The cloud selects the clients.
3. The cloud sends the model to the clients.
4. Each client trains locally.
5. Each client sends its weights to the corresponding edge.
6. Each edge waits for `CLIENTS / EDGES` clients.
7. Each edge aggregates the weights of its clients into a single update.
8. Each edge sends one aggregated update to the cloud.
9. The cloud waits for all `EDGES`.
10. The cloud aggregates the edge updates and completes the round.

## Important Difference in This Version

This version does not only record edge events. The model aggregation is also hierarchical:

```text
clients -> edge aggregation -> cloud aggregation
```

In other words, the cloud no longer performs a direct FedAvg over all clients. First, each edge aggregates its group of clients. Then, the cloud aggregates the models received from the edges.

## Main Configuration

In `pyproject.toml`:

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

* `num-supernodes`: total number of clients available in Flower.
* `num-selected-clients`: number of clients selected per round.
* `num-edges`: number of logical edges.
* `num-selected-clients` must be divisible by `num-edges`.
* `clients_per_edge = num-selected-clients / num-edges`.

Example:

```text
num-selected-clients = 10
num-edges = 2
clients_per_edge = 5
```

Therefore:

```text
edge 0: clients 0, 1, 2, 3, 4
edge 1: clients 5, 6, 7, 8, 9
```

## Running

It is recommended to use `python -m flwr` to avoid errors caused by a virtual environment pointing to another Python installation.

```powershell
python -m pip install -U pip
python -m pip install -e .
python -m pip install -U "flwr[simulation]" torch torchvision datasets "flwr-datasets[vision]" matplotlib
python -m flwr federation simulation-config --num-supernodes 10 local
python -m flwr run . local --stream
```

## Cleaning Logs and Processes on Windows

```powershell
taskkill /F /IM python.exe
taskkill /F /IM ray.exe
taskkill /F /IM flower-supernode.exe
taskkill /F /IM flower-superlink.exe
Remove-Item -Recurse -Force logs
```

## Generated Files

```text
logs/fl_timing_exec_001.csv
logs/fl_timing_exec_002.csv
logs/fl_timing_exec_003.csv
logs/compiled_with_warmup.csv
logs/model_parameters_for_spn.csv
```

## Most Important File for Filling the SPN Model

Use this file:

```text
logs/model_parameters_for_spn.csv
```

It contains only the parameters that should be inserted into the model:

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

The `ECS` parameter is calculated as a calibrated residual:

```text
ECS = average_reconstructed_hfl_round_time_with_warmup
      - (SC + T + EPPE + AGE + EC + AGC)
```

This avoids directly using the raw `envia_para_clientes` value from Flower/Ray, which may not represent exactly the same concept as in the SPN model.

## Log Interpretation

The main stages are:

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

The reconstructed total used for comparison with the SPN follows:

```text
seleciona_clientes
+ envia_para_clientes
+ max_by_edge(
    max_by_client(treinamento_por_cliente + enviar_pesos_para_edge)
    + agregar_no_edge
    + enviar_cloud_por_edge
  )
+ agregacao_de_edges_na_cloud
```

The `tempo_wallclock_medio_strategy_start_com_warmup` value is only a Flower/Ray diagnostic metric. It includes runtime overheads, scheduling, loading, and other costs that should not be directly summed into the SPN parameters.
