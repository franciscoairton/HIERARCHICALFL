# Quickstart PyTorch HFL

This project runs a Flower/PyTorch simulation adapted for Hierarchical Federated Learning (HFL), following the communication flow:

```text
cloud -> clients -> edges -> cloud
```

The implementation extends the traditional Flower federated learning workflow by adding an intermediate edge layer between the clients and the cloud server. In this version, clients do not send their updates directly to the cloud for global aggregation. Instead, each client sends its trained weights to an assigned edge server. Each edge aggregates the updates from its clients and then forwards a single aggregated update to the cloud.

The general workflow is:

1. The cloud initializes the global model.
2. The cloud selects the clients for the current round.
3. The cloud sends the model to the selected clients.
4. Each client trains the model locally.
5. Each client sends its trained weights to the assigned edge.
6. Each edge waits for the updates from its group of clients.
7. Each edge aggregates the client updates into a single edge update.
8. Each edge sends its aggregated update to the cloud.
9. The cloud waits for all edge updates.
10. The cloud aggregates the edge updates and completes the round.

## Main Difference from Traditional Flower FedAvg

In the traditional Flower FedAvg workflow, the cloud receives and aggregates the updates directly from all selected clients.

In this adapted HFL version, aggregation happens in two levels:

```text
clients -> edge aggregation -> cloud aggregation
```

This means that the cloud no longer performs FedAvg directly over all individual client updates. First, each edge aggregates the updates from its assigned clients. Then, the cloud aggregates the models received from the edges.

This behavior makes the simulation closer to an HFL scenario, where edge servers act as intermediate aggregation points between clients and the central server.

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

* `num-supernodes`: total number of clients available in the Flower simulation.
* `num-selected-clients`: number of clients selected per training round.
* `num-evaluate-clients`: number of clients used for federated evaluation.
* `num-edges`: number of logical edge servers.
* `local-epochs`: number of local training epochs per client.
* `learning-rate`: learning rate used during local training.
* `batch-size`: batch size used by the clients.
* `log-dir`: directory where the timing logs are stored.

The value of `num-selected-clients` must be divisible by `num-edges`.

```text
clients_per_edge = num-selected-clients / num-edges
```

Example:

```text
num-selected-clients = 10
num-edges = 2
clients_per_edge = 5
```

Therefore, the clients are distributed across the edges as follows:

```text
edge 0: clients 0, 1, 2, 3, 4
edge 1: clients 5, 6, 7, 8, 9
```

## Running the Project

It is recommended to use `python -m flwr` to avoid errors caused by a virtual environment pointing to a different Python installation.

```powershell
python -m pip install -U pip
python -m pip install -e .
python -m pip install -U "flwr[simulation]" torch torchvision datasets "flwr-datasets[vision]" matplotlib
python -m flwr federation simulation-config --num-supernodes 10 local
python -m flwr run . local --stream
```

The number passed to `--num-supernodes` should match the `num-supernodes` value defined in `pyproject.toml`.

## Cleaning Logs and Processes on Windows

```powershell
taskkill /F /IM python.exe
taskkill /F /IM ray.exe
taskkill /F /IM flower-supernode.exe
taskkill /F /IM flower-superlink.exe
Remove-Item -Recurse -Force logs
```

## Generated Files

After running the simulation, the project generates timing logs and compiled summaries:

```text
logs/fl_timing_exec_001.csv
logs/fl_timing_exec_002.csv
logs/fl_timing_exec_003.csv
logs/compiled_with_warmup.csv
logs/model_parameters_for_spn.csv
```

The files `fl_timing_exec_001.csv`, `fl_timing_exec_002.csv`, and `fl_timing_exec_003.csv` store the timing events for each execution.

The file `compiled_with_warmup.csv` contains the compiled timing summary across executions.

The file `model_parameters_for_spn.csv` is generated as an additional output for users who want to reuse the measured times in analytical models.

## Timing Logs

The main stages recorded in the logs are:

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

These stages make it possible to analyze the time spent in the main parts of the HFL workflow, including client training, client-to-edge communication, edge aggregation, edge-to-cloud communication, and cloud aggregation.

## Interpreting the HFL Timing Flow

The reconstructed HFL round time follows the hierarchical execution logic:

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

This calculation considers that clients assigned to the same edge may execute in parallel or in overlapping waves. Therefore, the total time does not simply sum the training time of all clients sequentially.

The `tempo_wallclock_medio_strategy_start_com_warmup` value is kept as a diagnostic metric from Flower/Ray. It may include runtime overheads, scheduling, loading, and other internal costs. For this reason, it should be interpreted as an execution-level diagnostic value rather than as a direct sum of the individual HFL stages.

## Notes

This version is intended to provide a practical HFL adaptation of Flower/PyTorch. It can be used to study hierarchical aggregation, compare traditional FL and HFL workflows, and collect timing information from each stage of the federated learning process.
