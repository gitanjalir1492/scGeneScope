import csv
from pathlib import Path

from scipy.stats import spearmanr


SEARCH_DIR = Path(__file__).resolve().parent.parent

MASTER_RESULTS = (
    SEARCH_DIR
    / "results"
    / "master_results.csv"
)

PROXY_RESULTS = (
    SEARCH_DIR
    / "results"
    / "proxy_results.csv"
)


def read_csv(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def to_float(value):
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def main():
    master_rows = read_csv(
        MASTER_RESULTS
    )

    proxy_rows = read_csv(
        PROXY_RESULTS
    )

    full_by_id = {
        row["experiment_id"]: row
        for row in master_rows
        if (
            row.get("status") == "completed"
            and row.get("search_method") == "random"
        )
    }

    # Keep only the latest completed proxy for each source config.
    latest_proxy = {}

    for row in proxy_rows:
        if row.get("status") != "completed":
            continue

        source_id = row.get(
            "source_experiment_id"
        )

        if not source_id:
            continue

        latest_proxy[source_id] = row

    joined = []

    for source_id, proxy in latest_proxy.items():
        full = full_by_id.get(source_id)

        if full is None:
            continue

        proxy_acc = to_float(
            proxy.get("val_accuracy")
        )

        proxy_f1 = to_float(
            proxy.get("val_f1")
        )

        full_acc = to_float(
            full.get("val_accuracy")
        )

        full_f1 = to_float(
            full.get("val_f1")
        )

        if proxy_acc is None or full_acc is None:
            continue

        joined.append(
            {
                "experiment_id": source_id,
                "proxy_acc": proxy_acc,
                "full_acc": full_acc,
                "acc_delta":
                    proxy_acc - full_acc,
                "proxy_f1": proxy_f1,
                "full_f1": full_f1,
            }
        )

    joined.sort(
        key=lambda row: row["experiment_id"]
    )

    if not joined:
        print(
            "No completed proxy/full pairs found."
        )
        return

    print(
        "\nProxy vs full validation results\n"
    )

    header = (
        f"{'ID':<12}"
        f"{'Proxy Acc':>12}"
        f"{'Full Acc':>12}"
        f"{'Delta':>12}"
        f"{'Proxy F1':>12}"
        f"{'Full F1':>12}"
    )

    print(header)
    print("-" * len(header))

    for row in joined:
        proxy_f1 = (
            ""
            if row["proxy_f1"] is None
            else f"{row['proxy_f1']:.4f}"
        )

        full_f1 = (
            ""
            if row["full_f1"] is None
            else f"{row['full_f1']:.4f}"
        )

        print(
            f"{row['experiment_id']:<12}"
            f"{row['proxy_acc']:>12.4f}"
            f"{row['full_acc']:>12.4f}"
            f"{row['acc_delta']:>12.4f}"
            f"{proxy_f1:>12}"
            f"{full_f1:>12}"
        )

    print(
        f"\nMatched pairs: {len(joined)}"
    )

    mae = (
        sum(
            abs(row["acc_delta"])
            for row in joined
        )
        / len(joined)
    )

    print(
        "Mean absolute accuracy error: "
        f"{mae:.4f}"
    )

    proxy_accs = [
        row["proxy_acc"]
        for row in joined
    ]

    full_accs = [
        row["full_acc"]
        for row in joined
    ]

    result = spearmanr(
        proxy_accs,
        full_accs,
    )

    print(
        "Spearman rank correlation: "
        f"{result.statistic:.4f}"
    )

    print(
        "Spearman p-value: "
        f"{result.pvalue:.6f}"
    )

    proxy_rank = sorted(
        joined,
        key=lambda row: row["proxy_acc"],
        reverse=True,
    )

    full_rank = sorted(
        joined,
        key=lambda row: row["full_acc"],
        reverse=True,
    )

    print("\nProxy ranking:")

    for i, row in enumerate(
        proxy_rank,
        start=1,
    ):
        print(
            f"{i}. "
            f"{row['experiment_id']} "
            f"({row['proxy_acc']:.4f})"
        )

    print("\nFull ranking:")

    for i, row in enumerate(
        full_rank,
        start=1,
    ):
        print(
            f"{i}. "
            f"{row['experiment_id']} "
            f"({row['full_acc']:.4f})"
        )


if __name__ == "__main__":
    main()
