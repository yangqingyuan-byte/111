import json
import os
import sys
import csv
from datetime import datetime
from tabulate import tabulate

def load_logs(log_file):
    """加载 JSONL 格式的日志文件"""
    data = []
    if not os.path.exists(log_file):
        print(f"错误: 找不到日志文件 {log_file}")
        return []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data

def get_choice(options, prompt):
    """通用的交互式选择函数"""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print(f"  0. 全部 (或跳过筛选)")
    
    while True:
        try:
            choice = int(input("\n请输入数字编号: "))
            if choice == 0:
                return None
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print("编号超出范围，请重新输入。")
        except ValueError:
            print("请输入有效的数字编号。")

def format_params(row):
    """格式化超参数组合"""
    exclude = {'data_path', 'model', 'model_id', 'test_mse', 'test_mae', 'timestamp'}
    params = [f"{k}={v}" for k, v in row.items() if k not in exclude]
    return "\n".join(params)

def get_exportable_params(row):
    """提取适合导出的参数组合"""
    exclude = {'data_path', 'model', 'model_id', 'test_mse', 'test_mae', 'timestamp'}
    return {k: v for k, v in row.items() if k not in exclude}

def get_model_name(log):
    """获取模型名称，优先使用 model_id，如果不存在则使用 model 字段"""
    return log.get('model_id') or log.get('model', 'Unknown')

def sanitize_name(name):
    return str(name).replace('/', '_').replace(' ', '_')

def export_best_configs(best_items, selected_data, selected_model, metric_name):
    """导出最佳参数组合到 JSON/CSV"""
    if not best_items:
        print("\n没有可导出的结果。")
        return

    export_dir = "/root/0/T3Time/Results/best_config_exports"
    os.makedirs(export_dir, exist_ok=True)

    data_name = sanitize_name(selected_data or "all_data")
    model_name = sanitize_name(selected_model or "all_models")
    metric_tag = metric_name.lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{data_name}_{model_name}_{metric_tag}_{ts}"
    json_path = os.path.join(export_dir, f"{base_name}.json")
    csv_path = os.path.join(export_dir, f"{base_name}.csv")

    export_rows = []
    for item in best_items:
        source = item['source_log']
        export_rows.append({
            'pred_len': item['pred_len'],
            'metric': metric_name,
            'selected_model': get_model_name(source),
            'test_mse': source['test_mse'],
            'test_mae': source['test_mae'],
            'data_path': source.get('data_path'),
            'params': get_exportable_params(source),
        })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(export_rows, f, ensure_ascii=False, indent=2)

    flat_rows = []
    all_param_keys = sorted({
        key
        for row in export_rows
        for key in row['params'].keys()
    })
    for row in export_rows:
        flat_row = {
            'pred_len': row['pred_len'],
            'metric': row['metric'],
            'selected_model': row['selected_model'],
            'test_mse': row['test_mse'],
            'test_mae': row['test_mae'],
            'data_path': row['data_path'],
        }
        for key in all_param_keys:
            flat_row[key] = row['params'].get(key, '')
        flat_rows.append(flat_row)

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['pred_len', 'metric', 'selected_model', 'test_mse', 'test_mae', 'data_path'] + all_param_keys
        )
        writer.writeheader()
        writer.writerows(flat_rows)

    print("\n已导出最佳参数组合:")
    print(f"  JSON: {json_path}")
    print(f"  CSV : {csv_path}")

def maybe_export_best_configs(mse_best_by_pred_len, mae_best_by_pred_len, selected_data, selected_model):
    """交互式导出最佳参数组合"""
    print("\n是否导出最佳参数组合？")
    print("  1. 导出按 MSE 选出的最佳组合")
    print("  2. 导出按 MAE 选出的最佳组合")
    print("  3. 两者都导出")
    print("  0. 不导出")
    choice = input("\n请输入编号 (默认 0): ").strip()

    if choice == '1':
        export_best_configs(mse_best_by_pred_len, selected_data, selected_model, "MSE")
    elif choice == '2':
        export_best_configs(mae_best_by_pred_len, selected_data, selected_model, "MAE")
    elif choice == '3':
        export_best_configs(mse_best_by_pred_len, selected_data, selected_model, "MSE")
        export_best_configs(mae_best_by_pred_len, selected_data, selected_model, "MAE")

def maybe_export_single_best(mse_sorted, mae_sorted, selected_data, selected_model):
    """交互式导出当前筛选条件下的最佳单条组合"""
    print("\n是否导出当前筛选条件下的最佳参数组合？")
    print("  1. 导出按 MSE 排序的第 1 名")
    print("  2. 导出按 MAE 排序的第 1 名")
    print("  3. 两者都导出")
    print("  0. 不导出")
    choice = input("\n请输入编号 (默认 0): ").strip()

    mse_items = []
    mae_items = []
    if mse_sorted:
        mse_items.append({
            'pred_len': mse_sorted[0].get('pred_len', 'N/A'),
            'mse': mse_sorted[0]['test_mse'],
            'mae': mse_sorted[0]['test_mae'],
            'model': get_model_name(mse_sorted[0]),
            'params': format_params(mse_sorted[0]),
            'source_log': mse_sorted[0],
        })
    if mae_sorted:
        mae_items.append({
            'pred_len': mae_sorted[0].get('pred_len', 'N/A'),
            'mse': mae_sorted[0]['test_mse'],
            'mae': mae_sorted[0]['test_mae'],
            'model': get_model_name(mae_sorted[0]),
            'params': format_params(mae_sorted[0]),
            'source_log': mae_sorted[0],
        })

    if choice == '1':
        export_best_configs(mse_items, selected_data, selected_model, "MSE")
    elif choice == '2':
        export_best_configs(mae_items, selected_data, selected_model, "MAE")
    elif choice == '3':
        export_best_configs(mse_items, selected_data, selected_model, "MSE")
        export_best_configs(mae_items, selected_data, selected_model, "MAE")

def analyze_all_pred_lens(logs, selected_data, selected_model, pred_lens):
    """统计所有预测长度的最好结果并求均值"""
    print(f"\n" + "="*80)
    print(f" 正在分析: 数据集={selected_data or '全部'}, 模型={selected_model or '全部'}")
    print(f" 统计所有预测长度: {pred_lens}")
    print("="*80)
    
    # 按 MSE 排序的最好结果
    mse_best_by_pred_len = []
    mse_values = []
    
    # 按 MAE 排序的最好结果
    mae_best_by_pred_len = []
    mae_values = []
    
    for pred_len in pred_lens:
        # 筛选当前 pred_len 的数据
        pred_len_logs = [log for log in logs if log.get('pred_len') == pred_len]
        
        if not pred_len_logs:
            continue
        
        # 找出 MSE 最好的结果
        mse_best = min(pred_len_logs, key=lambda x: x['test_mse'])
        mse_best_model = get_model_name(mse_best)
        mse_best_by_pred_len.append({
            'pred_len': pred_len,
            'mse': mse_best['test_mse'],
            'mae': mse_best['test_mae'],
            'model': mse_best_model,
            'params': format_params(mse_best),
            'source_log': mse_best
        })
        mse_values.append(mse_best['test_mse'])
        
        # 找出 MAE 最好的结果
        mae_best = min(pred_len_logs, key=lambda x: x['test_mae'])
        mae_best_model = get_model_name(mae_best)
        mae_best_by_pred_len.append({
            'pred_len': pred_len,
            'mse': mae_best['test_mse'],
            'mae': mae_best['test_mae'],
            'model': mae_best_model,
            'params': format_params(mae_best),
            'source_log': mae_best
        })
        mae_values.append(mae_best['test_mae'])
    
    # 计算均值
    mse_mean = sum(mse_values) / len(mse_values) if mse_values else 0
    mae_mean = sum(mae_values) / len(mae_values) if mae_values else 0
    
    # 显示按 MSE 排序的最好结果
    print(f"\n>>> 按 Test MSE 排序 - 每个预测长度的最好结果:")
    mse_table = []
    for item in mse_best_by_pred_len:
        mse_table.append([
            str(item['pred_len']),
            item.get('model', 'Unknown'),
            f"{item['mse']:.6f}",
            f"{item['mae']:.6f}",
            item['params']
        ])
    print(tabulate(mse_table, headers=["pred_len", "Model", "MSE (最好)", "MAE", "Parameters"], tablefmt="grid"))
    print(f"\n所有预测长度 MSE 最好结果的平均值: {mse_mean:.6f}")
    mse_mae_mean = sum(item['mae'] for item in mse_best_by_pred_len) / len(mse_best_by_pred_len) if mse_best_by_pred_len else 0
    print(f"所有预测长度对应 MAE 的平均值: {mse_mae_mean:.6f}")
    
    # 显示按 MAE 排序的最好结果
    print(f"\n>>> 按 Test MAE 排序 - 每个预测长度的最好结果:")
    mae_table = []
    for item in mae_best_by_pred_len:
        mae_table.append([
            str(item['pred_len']),
            item.get('model', 'Unknown'),
            f"{item['mse']:.6f}",
            f"{item['mae']:.6f}",
            item['params']
        ])
    print(tabulate(mae_table, headers=["pred_len", "Model", "MSE", "MAE (最好)", "Parameters"], tablefmt="grid"))
    print(f"\n所有预测长度 MAE 最好结果的平均值: {mae_mean:.6f}")
    mae_mse_mean = sum(item['mse'] for item in mae_best_by_pred_len) / len(mae_best_by_pred_len) if mae_best_by_pred_len else 0
    print(f"所有预测长度对应 MSE 的平均值: {mae_mse_mean:.6f}")
    maybe_export_best_configs(mse_best_by_pred_len, mae_best_by_pred_len, selected_data, selected_model)

def main():
    log_file = "/root/0/T3Time/experiment_results.log"
    logs = load_logs(log_file)
    
    if not logs:
        print("日志文件为空或不存在。")
        return

    # 1. 选择 Data Path
    data_paths = sorted(list(set(log['data_path'] for log in logs)))
    selected_data = get_choice(data_paths, "请选择要查看的数据集 (Data Path):")
    
    current_logs = logs
    if selected_data:
        current_logs = [log for log in current_logs if log['data_path'] == selected_data]

    # 2. 选择 Model
    # 优先使用 model_id，如果不存在则使用 model 字段
    models = sorted(list(set(get_model_name(log) for log in current_logs)))
    selected_model = get_choice(models, "请选择要查看的模型 (Model/Model_ID):")
    
    if selected_model:
        # 匹配 model_id 或 model 字段
        current_logs = [log for log in current_logs if get_model_name(log) == selected_model]

    # 2.5 询问是否统计所有预测长度
    pred_lens = sorted(list(set(log.get('pred_len') for log in current_logs if 'pred_len' in log and log.get('pred_len') is not None)))
    if pred_lens:
        print("\n请选择分析模式:")
        print("  1. 统计所有预测长度 (pred_len) 的最好结果并求均值")
        print("  2. 选择特定预测长度 (pred_len) 进行详细分析")
        analysis_mode = input("\n请输入编号 (默认 2): ").strip()
        
        if analysis_mode == '1':
            # 统计所有预测长度的最好结果
            analyze_all_pred_lens(current_logs, selected_data, selected_model, pred_lens)
            return
        
        # 3. 选择预测步长（pred_len）
        selected_pred_len_str = None
        pred_len_strs = [str(pl) for pl in pred_lens]
        selected_pred_len_str = get_choice(pred_len_strs, "请选择预测步长 (pred_len):")
        if selected_pred_len_str:
            selected_pred_len = int(selected_pred_len_str) if selected_pred_len_str.isdigit() else selected_pred_len_str
            current_logs = [log for log in current_logs if log.get('pred_len') == selected_pred_len]
    else:
        print("\n提示: 当前筛选结果中没有 pred_len 字段，跳过 pred_len 筛选。")

    # 4. 选择 Channel
    selected_channel_str = None
    # 检查是否有 channel 字段，如果没有则跳过
    channels = sorted(list(set(log.get('channel') for log in current_logs if 'channel' in log and log.get('channel') is not None)))
    if channels:
        # 将 channel 值转换为字符串以便显示
        channel_strs = [str(ch) for ch in channels]
        selected_channel_str = get_choice(channel_strs, "请选择要查看的 Channel:")
        
        if selected_channel_str:
            selected_channel = int(selected_channel_str) if selected_channel_str.isdigit() else selected_channel_str
            current_logs = [log for log in current_logs if log.get('channel') == selected_channel]
    else:
        print("\n提示: 当前筛选结果中没有 channel 字段，跳过 channel 筛选。")

    if not current_logs:
        print("\n没有符合筛选条件的记录。")
        return

    # 4. 选择最好还是最坏
    print("\n请选择查看类型:")
    print("  1. 最好 (Test MSE 从小到大)")
    print("  2. 最坏 (Test MSE 从大到小)")
    mode = input("\n请输入编号 (默认 1): ").strip()
    
    is_ascending = mode != '2'
    mode_str = "最好" if is_ascending else "最坏"

    # 5. 排序并展示
    # 同时展示按 MSE 排序和按 MAE 排序的前 10 个结果
    print(f"\n" + "="*80)
    pred_len_info = f", pred_len={selected_pred_len_str}" if selected_pred_len_str else ""
    channel_info = f", Channel={selected_channel_str}" if selected_channel_str else ""
    print(f" 正在分析: 数据集={selected_data or '全部'}, 模型={selected_model or '全部'}{pred_len_info}{channel_info} ({mode_str} 10 个指标)")
    print("="*80)

    # 按 MSE 排序
    mse_sorted = sorted(current_logs, key=lambda x: x['test_mse'], reverse=not is_ascending)[:10]
    mse_table = []
    for row in mse_sorted:
        # 显示时优先使用 model_id，如果不存在则使用 model
        model_display = get_model_name(row)
        mse_table.append([
            model_display,
            row['data_path'],
            f"{row['test_mse']:.6f}",
            f"{row['test_mae']:.6f}",
            format_params(row)
        ])
    
    print(f"\n>>> 按 Test MSE 排序的 {mode_str} 结果:")
    print(tabulate(mse_table, headers=["Model", "Data", "MSE", "MAE", "Parameters"], tablefmt="grid"))

    # 按 MAE 排序
    mae_sorted = sorted(current_logs, key=lambda x: x['test_mae'], reverse=not is_ascending)[:10]
    mae_table = []
    for row in mae_sorted:
        # 显示时优先使用 model_id，如果不存在则使用 model
        model_display = get_model_name(row)
        mae_table.append([
            model_display,
            row['data_path'],
            f"{row['test_mse']:.6f}",
            f"{row['test_mae']:.6f}",
            format_params(row)
        ])
    
    print(f"\n>>> 按 Test MAE 排序的 {mode_str} 结果:")
    print(tabulate(mae_table, headers=["Model", "Data", "MSE", "MAE", "Parameters"], tablefmt="grid"))
    maybe_export_single_best(mse_sorted, mae_sorted, selected_data, selected_model)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(0)
