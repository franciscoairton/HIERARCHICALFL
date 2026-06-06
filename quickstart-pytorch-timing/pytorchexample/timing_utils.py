"""Utilities for timing logs used by the Flower HFL example.

This version avoids concurrent CSV appends. In Flower simulations, several Ray
client processes can write at the same time. Each timing event is first written
to an individual CSV file under logs/_events/exec_XXX/. At the end, the server
merges all events into one readable CSV per execution and creates a compiled
summary.
"""

from __future__ import annotations

import csv
import os
import shutil
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

FIELDNAMES = [
    "timestamp",
    "run_id",
    "execution_id",
    "server_round",
    "node_id",
    "stage",
    "elapsed_sec",
    "details",
]

# Server-side stages used to estimate the total round time.
# envia_para_clientes includes the whole client cycle observed by Flower/Ray:
# send, client execution/training, client reply, edge/cloud logical processing.
TOTAL_SERVER_STAGES = {
    "seleciona_clientes",
    "envia_para_clientes",
    "agregacao_de_edges_na_cloud",
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def get_log_dir(default: str = "logs") -> Path:
    log_dir = Path(os.environ.get("FL_TIMING_LOG_DIR", default))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def exec_log_path(execution_id: int, log_dir: str = "logs") -> Path:
    return get_log_dir(log_dir) / f"hfl_timing_exec_{execution_id:03d}.csv"


def event_dir_path(execution_id: int, log_dir: str = "logs") -> Path:
    return get_log_dir(log_dir) / "_events" / f"exec_{execution_id:03d}"


def clear_execution_logs(*, execution_id: int, log_dir: str = "logs") -> None:
    """Remove previous logs for one execution.

    On Windows, Ray/Python can keep file handles open briefly. If deletion is
    blocked, the old event folder is renamed and a new clean folder is created.
    """
    csv_path = exec_log_path(execution_id, log_dir)
    if csv_path.exists():
        try:
            csv_path.unlink()
        except OSError:
            archived_csv = csv_path.with_name(f"{csv_path.stem}_old_{int(time.time())}{csv_path.suffix}")
            try:
                csv_path.rename(archived_csv)
            except OSError:
                pass

    ev_dir = event_dir_path(execution_id, log_dir)
    ev_dir.parent.mkdir(parents=True, exist_ok=True)
    if ev_dir.exists():
        removed = False
        for _ in range(5):
            try:
                shutil.rmtree(ev_dir)
                removed = True
                break
            except (PermissionError, OSError):
                time.sleep(0.5)
        if not removed:
            archived = ev_dir.with_name(f"{ev_dir.name}_old_{int(time.time())}")
            try:
                ev_dir.rename(archived)
            except OSError:
                pass

    ev_dir.mkdir(parents=True, exist_ok=True)


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _row_sort_key(row: dict):
    ts = parse_iso(row.get("timestamp", "")) or datetime.min
    server_round = safe_int(row.get("server_round", "0")) or 0
    execution_id = safe_int(row.get("execution_id", "0")) or 0
    return (execution_id, server_round, ts, row.get("stage", ""), str(row.get("node_id", "")))


def write_log(
    *,
    execution_id: int,
    run_id: str = "",
    server_round: int,
    node_id: str | int,
    stage: str,
    elapsed_sec: float,
    details: str = "",
    log_dir: str = "logs",
) -> None:
    """Write one timing event using one file per event."""
    ev_dir = event_dir_path(execution_id, log_dir)
    ev_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": now_iso(),
        "run_id": run_id,
        "execution_id": execution_id,
        "server_round": server_round,
        "node_id": node_id,
        "stage": stage,
        "elapsed_sec": f"{elapsed_sec:.9f}",
        "details": details,
    }
    filename = f"{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex}.csv"
    path = ev_dir / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=_row_sort_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def materialize_execution_logs(*, log_dir: str, execution_ids: Iterable[int], run_id: str = "") -> None:
    """Merge per-event files into one readable CSV per execution."""
    for execution_id in execution_ids:
        ev_dir = event_dir_path(execution_id, log_dir)
        rows: list[dict] = []
        if ev_dir.exists():
            for path in ev_dir.glob("*.csv"):
                rows.extend(_read_rows(path))
        old_rows = _read_rows(exec_log_path(execution_id, log_dir))
        if old_rows and not rows:
            rows.extend(old_rows)
        if run_id:
            rows = [r for r in rows if r.get("run_id", "") == run_id]
        if rows:
            _write_rows(exec_log_path(execution_id, log_dir), rows)


def iter_log_rows(log_dir: str, execution_ids: Iterable[int], run_id: str = "") -> Iterable[dict]:
    for execution_id in execution_ids:
        path = exec_log_path(execution_id, log_dir)
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or row.get("stage") is None:
                    continue
                row.setdefault("execution_id", str(execution_id))
                if run_id and row.get("run_id", "") != run_id:
                    continue
                yield row


def ensure_estimated_send_rows(*, log_dir: str, execution_ids: Iterable[int]) -> None:
    """Insert envia_para_clientes rows when Flower internals did not expose them.

    The value is estimated using server timestamps:

        envia_para_clientes ~= inicio_agregacao_cloud - fim_selecao

    This stage includes sending the model, client training, sending updates to
    logical edges, edge aggregation, edge-to-cloud transfer, and Flower/Ray
    overhead. It should be used for total round time instead of summing all
    client/edge rows.
    """
    for execution_id in execution_ids:
        path = exec_log_path(execution_id, log_dir)
        rows = _read_rows(path)
        if not rows:
            continue

        changed = False
        by_round: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            server_round = safe_int(row.get("server_round", "0"))
            if server_round is None or server_round <= 0:
                continue
            by_round[server_round].append(row)

        for server_round, round_rows in by_round.items():
            stages = {row.get("stage", "") for row in round_rows}
            if "envia_para_clientes" in stages:
                continue

            selection_rows = [r for r in round_rows if r.get("stage") == "seleciona_clientes"]
            aggregation_rows = [r for r in round_rows if r.get("stage") == "agregacao_de_edges_na_cloud"]
            if not selection_rows or not aggregation_rows:
                continue

            selection = selection_rows[0]
            aggregation = aggregation_rows[-1]
            t_selection_end = parse_iso(selection.get("timestamp", ""))
            t_aggregation_end = parse_iso(aggregation.get("timestamp", ""))
            aggregation_elapsed = safe_float(aggregation.get("elapsed_sec", "")) or 0.0
            if t_selection_end is None or t_aggregation_end is None:
                continue

            estimated = (t_aggregation_end - t_selection_end).total_seconds() - aggregation_elapsed
            if estimated < 0:
                estimated = 0.0

            rows.append(
                {
                    "timestamp": aggregation.get("timestamp", now_iso()),
                    "execution_id": str(execution_id),
                    "server_round": str(server_round),
                    "node_id": "server",
                    "stage": "envia_para_clientes",
                    "elapsed_sec": f"{estimated:.9f}",
                    "details": (
                        "estimado_por_timestamps; inclui envio, treino nos clientes, envio para edges, "
                        "agregacao nos edges, envio para cloud e overhead Flower/Ray"
                    ),
                }
            )
            changed = True

        if changed:
            _write_rows(path, rows)


def _t_critical_95_two_tailed(df: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return table.get(df, 1.96)


def mean_std_ci95(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, mean, mean
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std_dev = variance ** 0.5
    margin = _t_critical_95_two_tailed(n - 1) * (std_dev / (n ** 0.5))
    return mean, std_dev, mean - margin, mean + margin



def compile_summary(
    *,
    log_dir: str,
    execution_ids: list[int],
    warmup_rounds: int,
    run_id: str = "",
) -> None:
    """Generate compiled HFL summary following the SPN process exactly.

    The reconstructed total follows:

        cloud selection
        + cloud -> clients
        + max over edges(
            max over clients in edge(training + client->edge)
            + edge aggregation
            + edge->cloud
          )
        + cloud aggregation over all edges

    This avoids double counting the Flower wall-clock round, which already
    contains all client/edge/cloud work.
    """
    materialize_execution_logs(log_dir=log_dir, execution_ids=execution_ids, run_id=run_id)
    rows = list(iter_log_rows(log_dir, execution_ids, run_id=run_id))
    if not rows:
        return

    old_without = get_log_dir(log_dir) / "compiled_without_warmup.csv"
    if old_without.exists():
        try:
            old_without.unlink()
        except OSError:
            pass

    _write_hfl_summary_file(
        rows=rows,
        log_dir=log_dir,
        output_name="compiled_with_warmup.csv",
        execution_ids=execution_ids,
        warmup_rounds=warmup_rounds,
    )


def _extract_edge_id(details: str, node_id: str = "") -> int:
    import re
    match = re.search(r"edge_id=(\d+)", details or "")
    if match:
        return int(match.group(1))
    match = re.search(r"edge_(\d+)", node_id or "")
    if match:
        return int(match.group(1))
    return 0



MODEL_PARAM_FIELDNAMES = [
    "parameter",
    "acronym",
    "value_sec",
    "count",
    "source",
    "details",
]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_model_parameters_file(
    *,
    log_dir: str,
    stage_values: dict[str, list[float]],
    reconstructed_round_mean: float,
) -> None:
    """Write only the SPN/HFL parameters that should be used in the model.

    The ECS parameter is calibrated from the reconstructed HFL round mean so
    that the mean SPN round time follows the same logical equation used in the
    summary:

        round = SC + ECS + max_edge(max_client(T + EPPE) + AGE + EC) + AGC

    Using average parameters for the symmetric validation scenario, this becomes:

        round ~= SC + ECS + T + EPPE + AGE + EC + AGC

    Therefore:

        ECS = reconstructed_round_mean - (SC + T + EPPE + AGE + EC + AGC)

    This avoids using the raw `envia_para_clientes` stage as ECS, because that
    stage is a Flower/Ray wall-clock envelope and would double count client/edge
    work already represented by T, EPPE, AGE, EC, and AGC in the SPN.
    """
    im = _mean(stage_values.get("inicializa_modelo", []))
    sc = _mean(stage_values.get("seleciona_clientes", []))
    t = _mean(stage_values.get("treinamento_por_cliente", []))
    eppe = _mean(stage_values.get("enviar_pesos_para_edge", []))
    age = _mean(stage_values.get("agregar_no_edge", []))
    ec = _mean(stage_values.get("enviar_cloud_por_edge", []))
    agc = _mean(stage_values.get("agregacao_de_edges_na_cloud", []))

    ecs = reconstructed_round_mean - (sc + t + eppe + age + ec + agc)

    rows = [
        {
            "parameter": "Model Initialization",
            "acronym": "IM",
            "value_sec": f"{im:.9f}",
            "count": len(stage_values.get("inicializa_modelo", [])),
            "source": "inicializa_modelo",
            "details": "mean time to initialize the cloud/global model",
        },
        {
            "parameter": "Client Selection",
            "acronym": "SC",
            "value_sec": f"{sc:.9f}",
            "count": len(stage_values.get("seleciona_clientes", [])),
            "source": "seleciona_clientes",
            "details": "mean time to select clients in the cloud",
        },
        {
            "parameter": "Send to Clients / Synchronization Residual",
            "acronym": "ECS",
            "value_sec": f"{ecs:.9f}",
            "count": len(stage_values.get("envia_para_clientes", [])),
            "source": "calibrated_residual",
            "details": (
                "calibrated as reconstructed_round_mean - (SC + T + EPPE + AGE + EC + AGC); "
                "do not use raw envia_para_clientes here to avoid double counting"
            ),
        },
        {
            "parameter": "Local Training",
            "acronym": "T",
            "value_sec": f"{t:.9f}",
            "count": len(stage_values.get("treinamento_por_cliente", [])),
            "source": "treinamento_por_cliente",
            "details": "mean local training time per client",
        },
        {
            "parameter": "Send Weights to Edge",
            "acronym": "EPPE",
            "value_sec": f"{eppe:.9f}",
            "count": len(stage_values.get("enviar_pesos_para_edge", [])),
            "source": "enviar_pesos_para_edge",
            "details": "mean time for a client update to be prepared/sent to its logical edge",
        },
        {
            "parameter": "Aggregate at Edge",
            "acronym": "AGE",
            "value_sec": f"{age:.9f}",
            "count": len(stage_values.get("agregar_no_edge", [])),
            "source": "agregar_no_edge",
            "details": "mean aggregation time at each edge after receiving its clients",
        },
        {
            "parameter": "Send Edge Update to Cloud",
            "acronym": "EC",
            "value_sec": f"{ec:.9f}",
            "count": len(stage_values.get("enviar_cloud_por_edge", [])),
            "source": "enviar_cloud_por_edge",
            "details": "mean time for an edge update to be prepared/sent to the cloud",
        },
        {
            "parameter": "Aggregate Edges at Cloud",
            "acronym": "AGC",
            "value_sec": f"{agc:.9f}",
            "count": len(stage_values.get("agregacao_de_edges_na_cloud", [])),
            "source": "agregacao_de_edges_na_cloud",
            "details": "mean cloud aggregation time after all edges arrive",
        },
        {
            "parameter": "Target HFL Round Time",
            "acronym": "TARGET_ROUND",
            "value_sec": f"{reconstructed_round_mean:.9f}",
            "count": "",
            "source": "tempo_total_medio_por_rodada_hfl_reconstruido_com_warmup",
            "details": "validation target per round; the calibrated parameters above reproduce this mean round time",
        },
    ]

    out_path = get_log_dir(log_dir) / "model_parameters_for_spn.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MODEL_PARAM_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_hfl_summary_file(
    *,
    rows: list[dict],
    log_dir: str,
    output_name: str,
    execution_ids: list[int],
    warmup_rounds: int,
) -> None:
    """Write HFL summary with reconstructed SPN-like totals."""
    stage_values: dict[str, list[float]] = defaultdict(list)
    wallclock_total_by_execution: dict[int, float] = {}
    wallclock_round_values: list[float] = []

    # Per-round structures
    selection: dict[tuple[int, int], float] = defaultdict(float)
    send_clients: dict[tuple[int, int], float] = defaultdict(float)
    cloud_agg: dict[tuple[int, int], float] = defaultdict(float)
    client_train: dict[tuple[int, int, str, int], float] = defaultdict(float)  # exec, round, node, edge
    client_to_edge: dict[tuple[int, int, str, int], float] = defaultdict(float)
    edge_agg: dict[tuple[int, int, int], float] = defaultdict(float)
    edge_to_cloud: dict[tuple[int, int, int], float] = defaultdict(float)

    for row in rows:
        elapsed = safe_float(row.get("elapsed_sec", ""))
        if elapsed is None:
            continue

        stage = row.get("stage", "")
        execution_id = safe_int(row.get("execution_id", "0"))
        server_round = safe_int(row.get("server_round", "0"))
        node_id = str(row.get("node_id", ""))
        details = row.get("details", "")
        if execution_id is None or server_round is None:
            continue

        stage_values[stage].append(elapsed)

        if stage == "tempo_total_execucao":
            wallclock_total_by_execution[execution_id] = elapsed
            continue
        if stage == "ciclo_hfl_wallclock" and server_round > 0:
            wallclock_round_values.append(elapsed)
            continue
        if server_round <= 0:
            continue

        key = (execution_id, server_round)
        if stage == "seleciona_clientes":
            selection[key] += elapsed
        elif stage == "envia_para_clientes":
            send_clients[key] += elapsed
        elif stage == "agregacao_de_edges_na_cloud":
            cloud_agg[key] += elapsed
        elif stage == "treinamento_por_cliente":
            edge_id = _extract_edge_id(details, node_id)
            client_train[(execution_id, server_round, node_id, edge_id)] += elapsed
        elif stage == "enviar_pesos_para_edge":
            edge_id = _extract_edge_id(details, node_id)
            client_to_edge[(execution_id, server_round, node_id, edge_id)] += elapsed
        elif stage == "agregar_no_edge":
            edge_id = _extract_edge_id(details, node_id)
            edge_agg[(execution_id, server_round, edge_id)] += elapsed
        elif stage == "enviar_cloud_por_edge":
            edge_id = _extract_edge_id(details, node_id)
            edge_to_cloud[(execution_id, server_round, edge_id)] += elapsed

    all_round_keys = sorted(set(selection) | set(send_clients) | set(cloud_agg))
    round_totals: dict[tuple[int, int], float] = {}

    for key in all_round_keys:
        execution_id, server_round = key

        # Build edge -> list(client path durations)
        clients_by_edge: dict[int, list[float]] = defaultdict(list)
        client_keys = set()
        for ck in client_train:
            if ck[0] == execution_id and ck[1] == server_round:
                client_keys.add(ck)
        for ck in client_to_edge:
            if ck[0] == execution_id and ck[1] == server_round:
                client_keys.add(ck)

        for ck in client_keys:
            _, _, node, edge_id = ck
            train_time = client_train.get((execution_id, server_round, node, edge_id), 0.0)
            to_edge_time = client_to_edge.get((execution_id, server_round, node, edge_id), 0.0)
            clients_by_edge[edge_id].append(train_time + to_edge_time)

        edge_ids = set(clients_by_edge.keys())
        for ek in edge_agg:
            if ek[0] == execution_id and ek[1] == server_round:
                edge_ids.add(ek[2])
        for ek in edge_to_cloud:
            if ek[0] == execution_id and ek[1] == server_round:
                edge_ids.add(ek[2])

        edge_times = []
        for edge_id in edge_ids:
            client_wait = max(clients_by_edge.get(edge_id, [0.0]))
            eagg = edge_agg.get((execution_id, server_round, edge_id), 0.0)
            etoc = edge_to_cloud.get((execution_id, server_round, edge_id), 0.0)
            edge_times.append(client_wait + eagg + etoc)

        cloud_wait_edges = max(edge_times) if edge_times else 0.0
        total = selection.get(key, 0.0) + send_clients.get(key, 0.0) + cloud_wait_edges + cloud_agg.get(key, 0.0)
        if total > 0:
            round_totals[key] = total

    round_total_values = list(round_totals.values())
    round_mean = sum(round_total_values) / len(round_total_values) if round_total_values else 0.0

    # Write a compact file with only the SPN parameters to use in the HFL model.
    _write_model_parameters_file(
        log_dir=log_dir,
        stage_values=stage_values,
        reconstructed_round_mean=round_mean,
    )

    total_by_execution: dict[int, float] = defaultdict(float)
    for (execution_id, _server_round), value in round_totals.items():
        total_by_execution[execution_id] += value
    execution_total_values = [total_by_execution[eid] for eid in execution_ids if total_by_execution.get(eid, 0) > 0]
    execution_total_mean, execution_total_std, execution_total_ci_low, execution_total_ci_high = mean_std_ci95(execution_total_values)

    wallclock_values = [wallclock_total_by_execution[eid] for eid in execution_ids if eid in wallclock_total_by_execution]
    wallclock_mean, wallclock_std, wallclock_ci_low, wallclock_ci_high = mean_std_ci95(wallclock_values)
    wallclock_round_mean = sum(wallclock_round_values) / len(wallclock_round_values) if wallclock_round_values else 0.0

    out_path = get_log_dir(log_dir) / output_name
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "summary_type",
            "stage",
            "mean_elapsed_sec",
            "count",
            "std_dev_sec",
            "ci95_low_sec",
            "ci95_high_sec",
            "warmup_rounds_removed",
            "details",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow(
            {
                "summary_type": "total",
                "stage": "tempo_total_medio_por_execucao_hfl_reconstruido_com_warmup",
                "mean_elapsed_sec": f"{execution_total_mean:.9f}",
                "count": len(execution_total_values),
                "std_dev_sec": f"{execution_total_std:.9f}",
                "ci95_low_sec": f"{execution_total_ci_low:.9f}",
                "ci95_high_sec": f"{execution_total_ci_high:.9f}",
                "warmup_rounds_removed": 0,
                "details": (
                    "total reconstruido seguindo o SPN HFL: seleciona_clientes + envia_para_clientes + "
                    "max_por_edge(max_por_cliente(treinamento_por_cliente + enviar_pesos_para_edge) + "
                    "agregar_no_edge + enviar_cloud_por_edge) + agregacao_de_edges_na_cloud; "
                    "IC 95% calculado sobre os totais das N execucoes"
                ),
            }
        )
        writer.writerow(
            {
                "summary_type": "total",
                "stage": "tempo_total_medio_por_rodada_hfl_reconstruido_com_warmup",
                "mean_elapsed_sec": f"{round_mean:.9f}",
                "count": len(round_total_values),
                "std_dev_sec": "",
                "ci95_low_sec": "",
                "ci95_high_sec": "",
                "warmup_rounds_removed": 0,
                "details": "media das rodadas reconstruidas com a barreira clientes->edge e a barreira edges->cloud",
            }
        )
        if wallclock_values:
            writer.writerow(
                {
                    "summary_type": "total",
                    "stage": "tempo_wallclock_medio_strategy_start_com_warmup",
                    "mean_elapsed_sec": f"{wallclock_mean:.9f}",
                    "count": len(wallclock_values),
                    "std_dev_sec": f"{wallclock_std:.9f}",
                    "ci95_low_sec": f"{wallclock_ci_low:.9f}",
                    "ci95_high_sec": f"{wallclock_ci_high:.9f}",
                    "warmup_rounds_removed": 0,
                    "details": "tempo wall-clock real medido ao redor de strategy.start; diagnostico, nao usar como soma de parametros SPN",
                }
            )
        if wallclock_round_values:
            writer.writerow(
                {
                    "summary_type": "total",
                    "stage": "tempo_wallclock_medio_por_rodada_com_warmup",
                    "mean_elapsed_sec": f"{wallclock_round_mean:.9f}",
                    "count": len(wallclock_round_values),
                    "std_dev_sec": "",
                    "ci95_low_sec": "",
                    "ci95_high_sec": "",
                    "warmup_rounds_removed": 0,
                    "details": "tempo wall-clock real da rodada Flower; inclui todas as etapas e overheads; nao somar com os tempos reconstruidos",
                }
            )

        for stage in sorted(stage_values):
            values = stage_values[stage]
            mean = sum(values) / len(values)
            writer.writerow(
                {
                    "summary_type": "stage",
                    "stage": stage,
                    "mean_elapsed_sec": f"{mean:.9f}",
                    "count": len(values),
                    "std_dev_sec": "",
                    "ci95_low_sec": "",
                    "ci95_high_sec": "",
                    "warmup_rounds_removed": 0,
                    "details": "media aritmetica dos registros da etapa; warmup incluido",
                }
            )
