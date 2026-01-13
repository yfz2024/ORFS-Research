import argparse
import pandas as pd
import matplotlib.pyplot as plt


def extract_table(file_content: str) -> pd.DataFrame:
    lines = file_content.splitlines()

    start_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("| result_dump") and "total_wirelength" in line:
            start_idx = i + 2
            break

    if start_idx == -1:
        return pd.DataFrame()

    rows = []
    for line in lines[start_idx:]:
        line = line.strip()
        if not line or line.startswith("#"):
            break
        if not (line.startswith("|") and line.endswith("|")):
            continue

        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) != 4:
            continue

        try:
            rows.append({
                "result_dump": int(parts[0]),
                "clock_period": float(parts[2]) if parts[2] != "N/A" else None,
                "total_wirelength": float(parts[3].replace("um", "")) if parts[3] != "N/A" else None
            })
        except ValueError:
            continue

    return pd.DataFrame(rows).dropna()


def build_envelope(df: pd.DataFrame, mode: str):
    df = df.sort_values("result_dump").reset_index(drop=True)

    if mode == "DWL":
        df["metric"] = df["total_wirelength"]
        ylabel = "Total Wirelength (um)"
    elif mode == "ECP":
        df["metric"] = df["clock_period"]
        ylabel = "Clock Period"
    else:
        df["metric"] = df["clock_period"] + df["total_wirelength"]
        ylabel = "Clock Period + Total Wirelength"

    df["envelope"] = df["metric"].cummin()
    return df, ylabel


def extract_envelope_breakpoints(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["envelope"].diff().fillna(0) < 0
    mask.iloc[0] = True
    return df[mask]


def plot_envelope(df_breakpoints: pd.DataFrame, ylabel: str, filename: str, mode: str, total_iters: int):
    plt.figure(figsize=(12, 6))

    x_vals = df_breakpoints["result_dump"].tolist()
    y_vals = df_breakpoints["envelope"].tolist()
    # extend to total iterations
    if x_vals[-1] < total_iters:
        x_vals.append(total_iters)
        y_vals.append(y_vals[-1])

    plt.step(x_vals, y_vals, where="post", linewidth=2, color="blue", label="cumulative best")
    plt.scatter(df_breakpoints["result_dump"], df_breakpoints["envelope"], color="red", label="best updates")

    # annotate distinct best values close to points
    seen = set()
    for _, row in df_breakpoints.iterrows():
        val = row["envelope"]
        if val in seen:
            continue
        seen.add(val)
        label = f"best={val:.2f}" if val == df_breakpoints["envelope"].iloc[-1] else f"{val:.2f}"
        plt.annotate(
            label,
            (row["result_dump"], val),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
            color="red",
        )

    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(f"{mode} Envelope vs Iteration\n{filename}")
    plt.grid(True, linestyle="--", linewidth=0.5)

    if total_iters <= 50:
        plt.xticks(range(1, total_iters + 1))

    plt.tight_layout()
    out = f"{filename.rsplit('.',1)[0]}_{mode}_envelope.png"
    plt.savefig(out)
    print(f"Saved plot to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("-o", "--objective", required=True,
                        choices=["DWL", "ECP", "COMBO"])
    args = parser.parse_args()

    with open(args.filename, "r", encoding="utf-8") as f:
        content = f.read()

    df_full = extract_table(content)
    if df_full.empty:
        print("No valid data found in table.")
        return
    # Collapse multiple entries per iteration to their best (min) values
    df_full = (
        df_full.groupby("result_dump", as_index=False)
        .agg({"clock_period": "min", "total_wirelength": "min"})
        .sort_values("result_dump")
    )
    total_iters = int(df_full["result_dump"].max())

    df_env, ylabel = build_envelope(df_full, args.objective)
    df_bp = extract_envelope_breakpoints(df_env)
    plot_envelope(df_bp, ylabel, args.filename, args.objective, total_iters)


if __name__ == "__main__":
    main()

# python3 print.py ./output_results/orfo-textgrad-i75-p25.md -o DWL
