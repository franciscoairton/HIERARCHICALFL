"""Flower/PyTorch app with explicit hierarchical federated learning (HFL).

This version implements the logical SPN flow:

    cloud -> clients -> edges -> cloud

For each round:
1. the cloud selects clients;
2. the cloud sends the current model to the selected clients;
3. each client trains locally and sends its weights to one logical edge;
4. each edge waits for exactly CLIENTS/EDGES clients;
5. each edge aggregates its client weights into one edge update;
6. each edge sends one update to the cloud;
7. the cloud waits for all EDGES updates and aggregates them into the new
   global model.

Flower/Ray still transports replies through the ServerApp process, but the
aggregation semantics are hierarchical: client updates are first aggregated per
edge, then the edge updates are aggregated at the cloud.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Any

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import Net, load_centralized_dataset, test
from pytorchexample.timing_utils import (
    Timer,
    clear_execution_logs,
    compile_summary,
    get_log_dir,
    write_log,
)

app = ServerApp()


def _plain_value(value: Any) -> Any:
    """Return a plain Python value from Flower metric-like values."""
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value


def _metric_value(reply: Any, key: str, default: Any = None) -> Any:
    """Read a metric from a Flower reply in a defensive way."""
    try:
        metrics = reply.content["metrics"]
        return _plain_value(metrics[key])
    except Exception:
        pass
    try:
        metrics = reply.content.get("metrics")
        return _plain_value(metrics[key])
    except Exception:
        return default


def _reply_partition_id(reply: Any, fallback: int) -> int:
    """Return client partition id from reply metrics or metadata fallback."""
    value = _metric_value(reply, "client_partition_id", None)
    if value is not None:
        try:
            return int(value)
        except Exception:
            pass
    for attr in ("src_node_id", "node_id"):
        try:
            value = getattr(reply.metadata, attr)
            return int(value)
        except Exception:
            pass
    return fallback


def _reply_num_examples(reply: Any) -> int:
    value = _metric_value(reply, "num-examples", 0)
    try:
        return int(value)
    except Exception:
        return 0


def _reply_arrays(reply: Any) -> ArrayRecord:
    try:
        return reply.content["arrays"]
    except Exception as exc:
        raise RuntimeError("Reply does not contain model arrays") from exc


def _weighted_average_state_dicts(state_dicts: list[dict], weights: list[int]) -> dict:
    """Weighted average of PyTorch state dicts.

    Parameters are averaged using the number of examples as weights. Floating
    tensors are averaged directly. Non-floating tensors, if any, are copied from
    the first state dict.
    """
    if not state_dicts:
        raise ValueError("state_dicts cannot be empty")

    total_weight = float(sum(weights))
    if total_weight <= 0:
        weights = [1 for _ in state_dicts]
        total_weight = float(len(state_dicts))

    averaged = {}
    first = state_dicts[0]
    for name, tensor in first.items():
        if torch.is_floating_point(tensor):
            acc = torch.zeros_like(tensor, dtype=tensor.dtype)
            for state, weight in zip(state_dicts, weights):
                acc = acc + state[name].to(dtype=tensor.dtype) * (float(weight) / total_weight)
            averaged[name] = acc
        else:
            averaged[name] = tensor.clone()
    return averaged


def _aggregate_replies_to_arrayrecord(replies: list[Any]) -> tuple[ArrayRecord, int]:
    """Aggregate client replies into one ArrayRecord and return examples."""
    state_dicts = []
    weights = []
    for reply in replies:
        state_dicts.append(_reply_arrays(reply).to_torch_state_dict())
        weights.append(_reply_num_examples(reply))
    total_examples = sum(weights)
    return ArrayRecord(_weighted_average_state_dicts(state_dicts, weights)), total_examples


def _aggregate_edge_updates(edge_updates: list[dict]) -> ArrayRecord:
    """Aggregate edge updates into the cloud model."""
    state_dicts = [edge["arrays"].to_torch_state_dict() for edge in edge_updates]
    weights = [int(edge["num_examples"]) for edge in edge_updates]
    return ArrayRecord(_weighted_average_state_dicts(state_dicts, weights))


class FaithfulHFLFedAvg(FedAvg):
    """Strategy that follows the SPN HFL process: clients -> edge -> cloud."""

    def __init__(
        self,
        *,
        execution_id: int,
        run_id: str,
        log_dir: str,
        num_edges: int,
        num_selected_clients: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.execution_id = execution_id
        self.run_id = run_id
        self.log_dir = log_dir
        self.num_edges = num_edges
        self.num_selected_clients = num_selected_clients
        self.clients_per_edge = num_selected_clients // num_edges

    def configure_train(self, server_round, arrays, config, grid):  # type: ignore[override]
        # Cloud selects clients.
        with Timer() as timer:
            messages = list(super().configure_train(server_round, arrays, config, grid))
        write_log(
            execution_id=self.execution_id,
            run_id=self.run_id,
            server_round=server_round,
            node_id="cloud",
            stage="seleciona_clientes",
            elapsed_sec=timer.elapsed,
            details=(
                f"num_clientes_selecionados={len(messages)}; "
                f"num_edges={self.num_edges}; clients_per_edge={self.clients_per_edge}"
            ),
            log_dir=self.log_dir,
        )

        # Cloud -> clients. The actual network dispatch is performed by Flower
        # after this method returns. This timed block corresponds to the model
        # transition that prepares/sends the global model to selected clients.
        with Timer() as timer:
            for idx, msg in enumerate(messages):
                edge_id = min(idx // self.clients_per_edge, self.num_edges - 1)
                try:
                    msg.content["config"]["server_round"] = server_round
                    msg.content["config"]["execution_id"] = self.execution_id
                    msg.content["config"]["run_id"] = self.run_id
                    msg.content["config"]["edge_id"] = edge_id
                    msg.content["config"]["num_edges"] = self.num_edges
                    msg.content["config"]["clients_per_edge"] = self.clients_per_edge
                except Exception:
                    pass
        write_log(
            execution_id=self.execution_id,
            run_id=self.run_id,
            server_round=server_round,
            node_id="cloud",
            stage="envia_para_clientes",
            elapsed_sec=timer.elapsed,
            details=(
                "cloud prepara/dissemina o modelo para os clientes selecionados; "
                "nao representa o wall-clock completo da rodada"
            ),
            log_dir=self.log_dir,
        )
        return messages

    def aggregate_train(self, server_round, replies):  # type: ignore[override]
        """Aggregate as HFL: clients -> edges -> cloud."""
        replies_list = list(replies)
        if len(replies_list) != self.num_selected_clients:
            raise RuntimeError(
                f"A cloud recebeu {len(replies_list)} respostas de clientes, "
                f"mas esperava {self.num_selected_clients}."
            )

        # Group clients by logical edge. The partition_id is used because the
        # order of replies can vary in Ray/Flower.
        replies_by_edge: dict[int, list[Any]] = defaultdict(list)
        for idx, reply in enumerate(replies_list):
            partition_id = _reply_partition_id(reply, fallback=idx)
            edge_id = min(partition_id // self.clients_per_edge, self.num_edges - 1)
            replies_by_edge[edge_id].append(reply)

        edge_updates: list[dict] = []
        for edge_id in range(self.num_edges):
            edge_replies = replies_by_edge.get(edge_id, [])

            # Barrier at the edge: CLIENTS/EDGES client tokens must be present.
            if len(edge_replies) != self.clients_per_edge:
                distribution = {eid: len(vals) for eid, vals in sorted(replies_by_edge.items())}
                raise RuntimeError(
                    f"Edge {edge_id} recebeu {len(edge_replies)} clientes, "
                    f"mas esperava {self.clients_per_edge}. Distribuicao recebida: {distribution}."
                )

            write_log(
                execution_id=self.execution_id,
                run_id=self.run_id,
                server_round=server_round,
                node_id=f"edge_{edge_id}",
                stage="edge_recebe_clientes",
                elapsed_sec=0.0,
                details=(
                    f"edge_id={edge_id}; barreira do edge satisfeita; "
                    f"clientes_recebidos={len(edge_replies)}; esperados={self.clients_per_edge}"
                ),
                log_dir=self.log_dir,
            )

            # Edge aggregation: CLIENTS/EDGES client updates -> one edge update.
            with Timer() as timer:
                edge_arrays, num_examples = _aggregate_replies_to_arrayrecord(edge_replies)
            write_log(
                execution_id=self.execution_id,
                run_id=self.run_id,
                server_round=server_round,
                node_id=f"edge_{edge_id}",
                stage="agregar_no_edge",
                elapsed_sec=timer.elapsed,
                details=(
                    f"edge_id={edge_id}; agrega {len(edge_replies)} clientes em um unico update de edge; "
                    f"num_examples={num_examples}"
                ),
                log_dir=self.log_dir,
            )

            # Edge -> cloud: each edge sends one aggregated update.
            with Timer() as timer:
                edge_update = {
                    "edge_id": edge_id,
                    "arrays": edge_arrays,
                    "num_examples": num_examples,
                }
                edge_updates.append(edge_update)
            write_log(
                execution_id=self.execution_id,
                run_id=self.run_id,
                server_round=server_round,
                node_id=f"edge_{edge_id}",
                stage="enviar_cloud_por_edge",
                elapsed_sec=timer.elapsed,
                details=f"edge_id={edge_id}; envia um update agregado do edge para a cloud",
                log_dir=self.log_dir,
            )

        # Barrier at the cloud: EDGES edge updates must arrive.
        if len(edge_updates) != self.num_edges:
            raise RuntimeError(f"Cloud recebeu {len(edge_updates)} edges, mas esperava {self.num_edges}.")
        write_log(
            execution_id=self.execution_id,
            run_id=self.run_id,
            server_round=server_round,
            node_id="cloud",
            stage="cloud_recebe_edges",
            elapsed_sec=0.0,
            details=f"barreira da cloud satisfeita; edges_recebidos={len(edge_updates)}; esperados={self.num_edges}",
            log_dir=self.log_dir,
        )

        # Cloud aggregation: EDGES edge updates -> one global model.
        with Timer() as timer:
            cloud_arrays = _aggregate_edge_updates(edge_updates)
            metrics = MetricRecord(
                {
                    "num-client-replies": len(replies_list),
                    "num-edge-updates": len(edge_updates),
                    "num-examples": sum(int(edge["num_examples"]) for edge in edge_updates),
                }
            )
            result = (cloud_arrays, metrics)
        write_log(
            execution_id=self.execution_id,
            run_id=self.run_id,
            server_round=server_round,
            node_id="cloud",
            stage="agregacao_de_edges_na_cloud",
            elapsed_sec=timer.elapsed,
            details=(
                f"cloud agrega {self.num_edges} updates de edge depois de todos chegarem; "
                f"num_respostas_clientes={len(replies_list)}; agregacao_hierarquica_real=true"
            ),
            log_dir=self.log_dir,
        )
        return result

    def start(self, *args, **kwargs):  # type: ignore[override]
        with Timer() as timer:
            result = super().start(*args, **kwargs)
        write_log(
            execution_id=self.execution_id,
            run_id=self.run_id,
            server_round=0,
            node_id="cloud",
            stage="tempo_total_execucao",
            elapsed_sec=timer.elapsed,
            details="tempo wall-clock total de strategy.start; diagnostico, nao usar como soma de parametros SPN",
            log_dir=self.log_dir,
        )
        return result

    def _train_round(self, server_round, arrays, train_config, grid):  # type: ignore[override]
        start = time.perf_counter()
        result = super()._train_round(server_round, arrays, train_config, grid)
        elapsed = time.perf_counter() - start
        write_log(
            execution_id=self.execution_id,
            run_id=self.run_id,
            server_round=server_round,
            node_id="cloud",
            stage="ciclo_hfl_wallclock",
            elapsed_sec=elapsed,
            details=(
                "tempo wall-clock completo da rodada Flower; inclui transporte, clientes, edges logicos, "
                "retorno e cloud. Mantido apenas para diagnostico, nao somar com etapas HFL."
            ),
            log_dir=self.log_dir,
        )
        return result


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds: int = int(context.run_config["num-server-rounds"])
    num_executions: int = int(context.run_config.get("num-executions", 1))
    warmup_rounds: int = int(context.run_config.get("warmup-rounds", 1))
    num_supernodes: int = int(context.run_config.get("num-supernodes", 10))
    num_selected_clients: int = int(context.run_config.get("num-selected-clients", 10))
    num_evaluate_clients: int = int(context.run_config.get("num-evaluate-clients", 0))
    num_edges: int = int(context.run_config.get("num-edges", 1))
    lr: float = float(context.run_config["learning-rate"])

    if num_supernodes <= 0:
        raise ValueError("num-supernodes deve ser maior que zero")
    if num_selected_clients <= 0:
        raise ValueError("num-selected-clients deve ser maior que zero")
    if num_evaluate_clients < 0:
        raise ValueError("num-evaluate-clients nao pode ser negativo")
    if num_edges <= 0:
        raise ValueError("num-edges deve ser maior que zero")
    if num_selected_clients > num_supernodes:
        raise ValueError("num-selected-clients nao pode ser maior que num-supernodes")
    if num_evaluate_clients > num_supernodes:
        raise ValueError("num-evaluate-clients nao pode ser maior que num-supernodes")
    if num_selected_clients % num_edges != 0:
        raise ValueError("num-selected-clients deve ser divisivel por num-edges")

    fraction_train = num_selected_clients / num_supernodes
    fraction_evaluate = 0.0 if num_evaluate_clients == 0 else num_evaluate_clients / num_supernodes
    log_dir: str = str(context.run_config.get("log-dir", "logs"))
    clients_per_edge = num_selected_clients // num_edges
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    get_log_dir(log_dir).mkdir(parents=True, exist_ok=True)
    for execution_id in range(1, num_executions + 1):
        clear_execution_logs(execution_id=execution_id, log_dir=log_dir)

        with Timer() as timer:
            global_model = Net()
            arrays = ArrayRecord(global_model.state_dict())
        write_log(
            execution_id=execution_id,
            run_id=run_id,
            server_round=0,
            node_id="cloud",
            stage="inicializa_modelo",
            elapsed_sec=timer.elapsed,
            details=(
                "Net()+ArrayRecord(global_model.state_dict()); "
                f"num_supernodes={num_supernodes}; num_selected_clients={num_selected_clients}; "
                f"num_edges={num_edges}; clients_per_edge={clients_per_edge}; "
                f"num_evaluate_clients={num_evaluate_clients}"
            ),
            log_dir=log_dir,
        )

        strategy = FaithfulHFLFedAvg(
            execution_id=execution_id,
            run_id=run_id,
            log_dir=log_dir,
            num_edges=num_edges,
            num_selected_clients=num_selected_clients,
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=num_selected_clients,
            min_evaluate_nodes=num_evaluate_clients,
            min_available_nodes=num_supernodes,
        )

        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord(
                {
                    "lr": lr,
                    "execution_id": execution_id,
                    "run_id": run_id,
                    "server_round": 0,
                    "num_edges": num_edges,
                    "clients_per_edge": clients_per_edge,
                }
            ),
            num_rounds=num_rounds,
            evaluate_fn=global_evaluate,
        )

        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, f"final_hfl_model_exec_{execution_id:03d}.pt")

    compile_summary(
        log_dir=log_dir,
        execution_ids=list(range(1, num_executions + 1)),
        warmup_rounds=warmup_rounds,
        run_id=run_id,
    )


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
