import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import os

def extract_table(file_content: str) -> pd.DataFrame:
    """
    解析 Markdown 表格，提取 result_dump, ECP, total_wirelength
    """
    lines = file_content.splitlines()
    rows = []
    
    in_table = False
    header_found = False
    
    # 正则表达式：匹配数字（支持浮点数）
    num_re = re.compile(r'([\d.]+)')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 1. 识别表头 (兼容新旧格式)
        if '| result_dump |' in line and '| base |' in line:
            header_found = True
            continue
        
        # 2. 识别分隔行 (支持 :---: 格式)
        if header_found and re.match(r'^\|[\s\-:|]+\|$', line):
            in_table = True
            continue

        # 3. 解析数据行
        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')]
            # 过滤空字符串
            parts = [p for p in parts if p]

            # 需要至少4列: result_dump, base, ecp, dwl
            if len(parts) >= 4:
                try:
                    # 辅助解析函数
                    def parse_val(s):
                        if not s or "N/A" in s:
                            return None
                        m = num_re.search(s)
                        return float(m.group(1)) if m else None

                    # 提取数据 (列索引根据新表格格式调整)
                    # | result_dump | base | ecp | dwl | cts_wl |
                    r_dump = int(parts[0])
                    # base = int(parts[1]) # 暂不需要
                    ecp = parse_val(parts[2])
                    dwl = parse_val(parts[3])

                    # 只有当关键数据有效时才添加
                    rows.append({
                        "result_dump": r_dump,
                        "ECP": ecp,
                        "total_wirelength": dwl
                    })
                except (ValueError, IndexError):
                    continue

    return pd.DataFrame(rows)


def prepare_data(df: pd.DataFrame, mode: str):
    """
    根据目标计算 metric，并计算 Envelope (累积最小值)
    """
    # 1. 过滤掉所需列为 NaN 的行
    if mode == "DWL":
        df = df.dropna(subset=["total_wirelength"]).copy()
        df["metric"] = df["total_wirelength"]
        ylabel = "Total Wirelength (um)"
    elif mode == "ECP":
        df = df.dropna(subset=["ECP"]).copy()
        df["metric"] = df["ECP"]
        ylabel = "Effective Clock Period"
    else: # COMBO
        df = df.dropna(subset=["ECP", "total_wirelength"]).copy()
        df["metric"] = df["ECP"] + df["total_wirelength"]
        ylabel = "ECP + Total Wirelength"

    # 2. 按 result_dump 分组，找出每次迭代中的最优值
    # 注意：这里先计算 metric 再取 min，确保数据真实存在，而不是 ECP_min + DWL_min 拼凑出的假数据
    df_best = df.groupby("result_dump")["metric"].min().reset_index()
    
    # 3. 排序并计算累积最小值 (Envelope)
    df_best = df_best.sort_values("result_dump").reset_index(drop=True)
    df_best["envelope"] = df_best["metric"].cummin()
    
    return df_best, ylabel


def extract_envelope_breakpoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取 Envelope 下降的关键点（拐点）
    """
    # 找出 envelope 值发生变化的行，或者是第一行
    mask = df["envelope"].diff().fillna(0) < 0
    mask.iloc[0] = True
    return df[mask]


def plot_envelope(df_breakpoints: pd.DataFrame, ylabel: str, filename: str, mode: str, total_iters: int):
    plt.figure(figsize=(12, 6))

    x_vals = df_breakpoints["result_dump"].tolist()
    y_vals = df_breakpoints["envelope"].tolist()
    
    # 延伸绘图线到最后一次迭代
    if x_vals[-1] < total_iters:
        x_vals.append(total_iters)
        y_vals.append(y_vals[-1])

    # 绘制阶梯图
    plt.step(x_vals, y_vals, where="post", linewidth=2, color="blue", label="Cumulative Best")
    
    # 标记突破点
    plt.scatter(df_breakpoints["result_dump"], df_breakpoints["envelope"], color="red", label="New Best Found", zorder=5)

    # 在点旁边标注数值 (避免重叠)
    seen_vals = set()
    for _, row in df_breakpoints.iterrows():
        val = row["envelope"]
        # 只标注明显不同的值，防止过于密集
        if val in seen_vals:
            continue
        seen_vals.add(val)
        
        label_text = f"{val:.2f}"
        plt.annotate(
            label_text,
            (row["result_dump"], val),
            textcoords="offset points",
            xytext=(0, -15), # 放在点下方
            ha="center",
            fontsize=9,
            color="darkred",
            fontweight='bold'
        )

    plt.xlabel("Iteration (Result Dump)")
    plt.ylabel(ylabel)
    plt.title(f"{mode} Optimization Envelope\nSource: {filename}")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # 如果迭代次数不多，强制显示整数刻度
    if total_iters <= 25:
        plt.xticks(range(1, total_iters + 1))
    
    plt.tight_layout()
    
    # 生成输出文件名
    base_name = os.path.splitext(filename)[0] 
    out = f"{base_name}_{mode}_envelope.png"
    plt.savefig(out)
    print(f"Saved plot to {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot optimization envelope from log metrics.")
    parser.add_argument("filename", help="Path to markdown log file")
    parser.add_argument("-o", "--objective", required=True,
                        choices=["DWL", "ECP", "COMBO"],
                        help="Optimization objective")
    args = parser.parse_args()

    try:
        with open(args.filename, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found.")
        sys.exit(1)

    # 1. 提取所有原始数据
    df_raw = extract_table(content)
    if df_raw.empty:
        print("No valid data found in table. Check file format.")
        return

    # 获取最大迭代次数用于绘图边界
    total_iters = int(df_raw["result_dump"].max())

    # 2. 处理数据（计算 Metric -> GroupBy -> CumMin）
    df_env, ylabel = prepare_data(df_raw, args.objective)
    
    if df_env.empty:
        print(f"No valid data for objective {args.objective} (all N/A?).")
        return

    # 3. 提取拐点并绘图
    df_bp = extract_envelope_breakpoints(df_env)
    plot_envelope(df_bp, ylabel, args.filename, args.objective, total_iters)


if __name__ == "__main__":
    main()
