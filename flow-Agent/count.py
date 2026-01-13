#!/usr/bin/env python3
"""
分析 log_metrics Markdown 文件
找出最优目标值及对应的另一列（支持 DWL/ECP/COMBO）
统计 N/A 出现次数
统计每个 result_dump 中 N/A 出现次数
"""

import re
import sys
import argparse
from collections import defaultdict


def parse_markdown_table(file_path):
    """
    解析 Markdown 表格文件
    
    返回:
        records: 所有记录的列表 [{result_dump, base, clock_period, total_wirelength}, ...]
    """
    records = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找表格内容
    in_table = False
    header_found = False
    
    for line in lines:
        line = line.strip()
        
        # 跳过空行
        if not line:
            continue
        
        # 检测表头
        if '| result_dump |' in line and '| base |' in line:
            header_found = True
            continue
        
        # 跳过分隔行
        if header_found and re.match(r'^\|[\s\-|]+\|$', line):
            in_table = True
            continue
        
        # 解析数据行
        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')]
            # 过滤空字符串
            parts = [p for p in parts if p]
            
            if len(parts) >= 4:
                try:
                    result_dump = int(parts[0])
                    base = int(parts[1])
                    clock_period = float(parts[2])
                    
                    # 处理 total_wirelength
                    wirelength_str = parts[3]
                    if wirelength_str == 'N/A':
                        total_wirelength = None
                    else:
                        # 提取数字部分 (例如: "107593 um" -> 107593)
                        match = re.search(r'([\d.]+)', wirelength_str)
                        if match:
                            total_wirelength = float(match.group(1))
                        else:
                            total_wirelength = None
                    
                    records.append({
                        'result_dump': result_dump,
                        'base': base,
                        'clock_period': clock_period,
                        'total_wirelength': total_wirelength
                    })
                except (ValueError, IndexError) as e:
                    # 跳过无法解析的行
                    continue
    
    return records


def analyze_records(records, objective: str):
    """
    分析记录
    
    返回:
        min_info: 最优目标的信息（根据 objective）
        na_count: N/A 总次数
        na_per_dump: 每个 result_dump 的 N/A 次数
    """
    obj = objective.upper()

    # 目标列/伴随列选择
    def metric(rec):
        if obj == "DWL":
            return rec['total_wirelength']
        if obj == "ECP":
            return rec['clock_period']
        if obj == "COMBO":
            wl = rec['total_wirelength']
            cp = rec['clock_period']
            if wl is None or cp is None:
                return None
            return wl + cp
        return None

    valid_records = []
    for rec in records:
        m = metric(rec)
        if m is not None:
            rec = rec.copy()
            rec['metric'] = m
            valid_records.append(rec)
    
    min_info = None
    if valid_records:
        min_record = min(valid_records, key=lambda x: x['metric'])
        min_info = {
            'result_dump': min_record['result_dump'],
            'base': min_record['base'],
            'clock_period': min_record['clock_period'],
            'total_wirelength': min_record['total_wirelength'],
            'metric': min_record['metric'],
        }
    
    # 统计 N/A 总次数
    na_count = sum(1 for r in records if r['total_wirelength'] is None or r['clock_period'] is None)
    
    # 统计每个 result_dump 的 N/A 次数
    na_per_dump = defaultdict(int)
    for record in records:
        if record['total_wirelength'] is None or record['clock_period'] is None:
            na_per_dump[record['result_dump']] += 1
    
    return min_info, na_count, na_per_dump


def print_analysis(min_info, na_count, na_per_dump, records, objective: str):
    """
    打印分析结果
    """
    print("=" * 80)
    print("日志指标分析结果")
    print("=" * 80)
    
    # 最优目标
    print(f"\n【最优 {objective.upper()}】")
    if min_info:
        print(f"  Result Dump:      {min_info['result_dump']}")
        print(f"  Base:             {min_info['base']}")
        print(f"  Clock Period:     {min_info['clock_period']}")
        print(f"  Total Wirelength: {min_info['total_wirelength']}")
        print(f"  Metric({objective.upper()}): {min_info['metric']:.4f}")
    else:
        print("  没有找到有效的目标数据")
    
    # N/A 统计
    print(f"\n【N/A 统计】")
    print(f"  总记录数:         {len(records)}")
    print(f"  N/A 总次数:       {na_count}")
    print(f"  N/A 比例:         {na_count/len(records)*100:.2f}%")
    
    # 每个 result_dump 的 N/A 统计
    print(f"\n【各 Result Dump 的 N/A 统计】")
    if na_per_dump:
        # 找出 N/A 次数最多的
        max_na_dump = max(na_per_dump.items(), key=lambda x: x[1])
        print(f"  N/A 次数最多的 Result Dump: {max_na_dump[0]} (共 {max_na_dump[1]} 次)")
        
        # 显示前 10 个 N/A 最多的
        print(f"\n  前 10 个 N/A 最多的 Result Dump:")
        sorted_dumps = sorted(na_per_dump.items(), key=lambda x: x[1], reverse=True)[:10]
        for dump_id, count in sorted_dumps:
            print(f"    Result Dump {dump_id:3d}: {count:2d} 次 N/A")
    else:
        print("  所有记录都有有效的 wirelength 数据")
    
    # 完全有效的 result_dump
    print(f"\n【完全有效的 Result Dump】")
    all_dumps = set(r['result_dump'] for r in records)
    dumps_with_na = set(na_per_dump.keys())
    valid_dumps = all_dumps - dumps_with_na
    
    if valid_dumps:
        print(f"  无 N/A 的 Result Dump 数量: {len(valid_dumps)}")
        print(f"  Result Dump IDs: {sorted(valid_dumps)}")
    else:
        print("  没有完全无 N/A 的 Result Dump")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='分析 log_metrics Markdown 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python count.py log_metrics_dual_1208_100.md
  python count.py --detailed log_metrics_orfo-textgrad_1221_100.md
        '''
    )
    parser.add_argument('file', help='Markdown 文件路径')
    parser.add_argument('--detailed', action='store_true', 
                       help='显示详细的每个 result_dump 的 N/A 次数')
    parser.add_argument('-o', '--objective', required=True,
                       choices=['DWL', 'ECP', 'COMBO'],
                       help='选择分析的目标：DWL(总线长)、ECP(时钟周期)、COMBO(两者之和)')
    
    args = parser.parse_args()
    
    try:
        # 解析文件
        print(f"正在读取文件: {args.file}")
        records = parse_markdown_table(args.file)
        
        if not records:
            print("错误: 未找到有效的数据记录")
            sys.exit(1)
        
        print(f"成功解析 {len(records)} 条记录\n")
        
        # 分析数据
        min_info, na_count, na_per_dump = analyze_records(records, args.objective)
        
        # 打印结果
        print_analysis(min_info, na_count, na_per_dump, records, args.objective)
        
        # 详细模式：显示所有 result_dump 的 N/A 统计
        if args.detailed and na_per_dump:
            print("\n【详细 N/A 统计（所有 Result Dump）】")
            for dump_id in sorted(na_per_dump.keys()):
                count = na_per_dump[dump_id]
                print(f"  Result Dump {dump_id:3d}: {count:2d} 次 N/A")
    
    except FileNotFoundError:
        print(f"错误: 文件 '{args.file}' 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('-h')
    
    main()

# python3 count.py --detailed ./output_results/orfo-textgrad-i75-p25.md -o DWL
# python3 count.py --detailed ./output_results/orfo-textgrad-i75-p25.md -o ECP
# python3 count.py --detailed ./output_results/orfo-textgrad-i75-p25.md -o COMBO

