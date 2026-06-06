"""pytorchexample: A Flower / PyTorch HFL app."""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn
from pytorchexample.timing_utils import Timer, write_log

app = ClientApp()


def _get_config_value(config, key, default):
    try:
        return config[key]
    except Exception:
        return default


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data and send the update to its logical edge."""

    cfg = msg.content["config"]
    execution_id = int(_get_config_value(cfg, "execution_id", 1))
    run_id = str(_get_config_value(cfg, "run_id", ""))
    server_round = int(_get_config_value(cfg, "server_round", 0))
    log_dir = str(context.run_config.get("log-dir", "logs"))

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = context.run_config["batch-size"]

    # Edge assignment is computed from the actual client/partition id.
    # This is more robust than relying only on the message order produced by Flower.
    num_edges = int(context.run_config.get("num-edges", _get_config_value(cfg, "num_edges", 1)))
    clients_per_edge = int(context.run_config.get("clients-per-edge", _get_config_value(cfg, "clients_per_edge", 1)))
    if clients_per_edge <= 0:
        clients_per_edge = max(1, num_partitions // max(1, num_edges))
    edge_id = min(partition_id // clients_per_edge, max(0, num_edges - 1))

    with Timer() as timer:
        model = Net()
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    write_log(
        execution_id=execution_id,
        run_id=run_id,
        server_round=server_round,
        node_id=partition_id,
        stage="cliente_inicializa_modelo_recebido",
        elapsed_sec=timer.elapsed,
        details=f"edge_id={edge_id}; num_partitions={num_partitions}",
        log_dir=log_dir,
    )

    with Timer() as timer:
        trainloader, _ = load_data(partition_id, num_partitions, batch_size)
    write_log(
        execution_id=execution_id,
        run_id=run_id,
        server_round=server_round,
        node_id=partition_id,
        stage="cliente_carrega_dados",
        elapsed_sec=timer.elapsed,
        details=f"edge_id={edge_id}; batch_size={batch_size}; exemplos={len(trainloader.dataset)}",
        log_dir=log_dir,
    )

    with Timer() as timer:
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )
    write_log(
        execution_id=execution_id,
        run_id=run_id,
        server_round=server_round,
        node_id=partition_id,
        stage="treinamento_por_cliente",
        elapsed_sec=timer.elapsed,
        details=(
            f"edge_id={edge_id}; local_epochs={context.run_config['local-epochs']}; "
            f"lr={msg.content['config']['lr']}; train_loss={train_loss}"
        ),
        log_dir=log_dir,
    )

    # HFL step: the client prepares/sends its local weights to the assigned edge.
    # In this Flower simulation the edge is logical; the message still returns to
    # the server process, where edge aggregation is emulated by grouping replies.
    with Timer() as timer:
        model_record = ArrayRecord(model.state_dict())
        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
            "edge_id": edge_id,
            "clients_per_edge": clients_per_edge,
            "client_partition_id": partition_id,
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        reply = Message(content=content, reply_to=msg)
    write_log(
        execution_id=execution_id,
        run_id=run_id,
        server_round=server_round,
        node_id=partition_id,
        stage="enviar_pesos_para_edge",
        elapsed_sec=timer.elapsed,
        details=f"edge_id={edge_id}; prepara ArrayRecord(model.state_dict()) para o edge logico",
        log_dir=log_dir,
    )
    return reply


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
