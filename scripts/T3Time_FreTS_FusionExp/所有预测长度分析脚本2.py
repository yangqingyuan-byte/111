#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索 T3Time_FreTS_FusionExp 模型的所有种子的参数寻优实验结果
按预测长度（96, 192, 336, 720）分别分析
支持分析所有种子或指定种子的实验结果
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_hyperopt_results(result_file=None, seed=None, model_id=None, model=None, 
                          model_id_prefix=None, data_path=None):
    """
    加载参数寻优实验结果
    
    Args:
        result_file: 结果文件路径，默认为 experiment_results.log
        seed: 随机种子，如果为 None 则加载所有种子的结果
        model_id: 精确匹配的model_id（优先级最高）
        model: 精确匹配的model名称
        model_id_prefix: 模型ID前缀（如果model_id和model都为None时使用）
        data_path: 数据集名称（例如 'ETTh1'）。如果为 None 则不过滤数据集。
    """
    if result_file is None:
        result_file = os.path.join(project_root, "experiment_results.log")
    
    results = []
    
    if not os.path.exists(result_file):
        print(f"❌ 结果文件不存在: {result_file}")
        return results
    
    with open(result_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                
                # 模型匹配逻辑（优先级：model_id > model > model_id_prefix）
                matched = False
                if model_id is not None:
                    # 精确匹配model_id
                    if data.get('model_id') == model_id:
                        matched = True
                elif model is not None:
                    # 精确匹配model
                    if data.get('model') == model:
                        matched = True
                elif model_id_prefix is not None:
                    # 前缀匹配model_id
                    if data.get('model_id', '').startswith(model_id_prefix):
                        matched = True
                else:
                    # 如果没有指定任何匹配条件，匹配所有
                    matched = True
                
                if not matched:
                    continue

                # 如果指定了数据集，则只保留该数据集的结果
                if data_path is not None:
                    # 部分日志可能使用 'data' 或 'data_path' 作为键，这里统一兼容
                    log_data_path = data.get('data_path', data.get('data'))
                    if log_data_path != data_path:
                        continue

                # 如果指定了 seed，则只加载该 seed 的结果；否则加载所有 seed
                if seed is None or data.get('seed') == seed:
                    results.append(data)
            except json.JSONDecodeError as e:
                continue
            except Exception as e:
                continue
    
    return results

def find_best_params_by_pred_len(results, pred_lens=[96, 192, 336, 720]):
    """按预测长度分组，找出每个预测长度的最佳参数组合"""
    if not results:
        return {}
    
    results_by_pred_len = {}
    
    for pred_len in pred_lens:
        # 筛选该预测长度的结果
        pred_results = [r for r in results if r.get('pred_len') == pred_len]
        
        if not pred_results:
            results_by_pred_len[pred_len] = {
                'best_mse': None,
                'best_mae': None,
                'sorted_results_mse': [],
                'sorted_results_mae': [],
                'param_avg': {},
                'count': 0
            }
            continue
        
        # 按 MSE 排序
        sorted_results_mse = sorted(pred_results, key=lambda x: x.get('test_mse', float('inf')))
        best_mse = sorted_results_mse[0] if sorted_results_mse else None
        
        # 按 MAE 排序
        sorted_results_mae = sorted(pred_results, key=lambda x: x.get('test_mae', float('inf')))
        best_mae = sorted_results_mae[0] if sorted_results_mae else None
        
        # 统计每个参数组合的MSE和MAE
        param_stats_mse = defaultdict(list)
        param_stats_mae = defaultdict(list)
        for r in pred_results:
            param_key = (r.get('channel'), r.get('dropout_n'), r.get('head'))
            param_stats_mse[param_key].append(r.get('test_mse', float('inf')))
            param_stats_mae[param_key].append(r.get('test_mae', float('inf')))
        
        # 计算每个参数组合的平均 MSE 和 MAE
        param_avg = {}
        for param_key in param_stats_mse.keys():
            mse_list = param_stats_mse[param_key]
            mae_list = param_stats_mae[param_key]
            param_avg[param_key] = {
                'mse_mean': sum(mse_list) / len(mse_list),
                'mse_min': min(mse_list),
                'mse_max': max(mse_list),
                'mae_mean': sum(mae_list) / len(mae_list),
                'mae_min': min(mae_list),
                'mae_max': max(mae_list),
                'count': len(mse_list)
            }
        
        results_by_pred_len[pred_len] = {
            'best_mse': best_mse,
            'best_mae': best_mae,
            'sorted_results_mse': sorted_results_mse,
            'sorted_results_mae': sorted_results_mae,
            'param_avg': param_avg,
            'count': len(pred_results)
        }
    
    return results_by_pred_len

def get_available_models(result_file=None):
    """
    从日志文件中提取所有可用的model_id和model
    
    Args:
        result_file: 结果文件路径，默认为 experiment_results.log
    
    Returns:
        tuple: (model_ids列表, models列表)，都按字母顺序排序
    """
    if result_file is None:
        result_file = os.path.join(project_root, "experiment_results.log")
    
    model_ids = set()
    models = set()
    
    if not os.path.exists(result_file):
        return [], []
    
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                # 提取model_id
                model_id = data.get('model_id')
                if model_id:
                    model_ids.add(model_id)
                # 提取model
                model = data.get('model')
                if model:
                    models.add(model)
            except (json.JSONDecodeError, Exception):
                continue
    
    return sorted(list(model_ids)), sorted(list(models))

def get_available_datasets(result_file=None, model_id=None, model=None, model_id_prefix=None):
    """
    从日志文件中提取所有可用的数据集
    
    Args:
        result_file: 结果文件路径，默认为 experiment_results.log
        model_id: 精确匹配的model_id
        model: 精确匹配的model名称
        model_id_prefix: 模型ID前缀
    
    Returns:
        list: 可用数据集列表，按字母顺序排序
    """
    if result_file is None:
        result_file = os.path.join(project_root, "experiment_results.log")
    
    datasets = set()
    
    if not os.path.exists(result_file):
        return []
    
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                
                # 模型匹配逻辑
                matched = False
                if model_id is not None:
                    if data.get('model_id') == model_id:
                        matched = True
                elif model is not None:
                    if data.get('model') == model:
                        matched = True
                elif model_id_prefix is not None:
                    if data.get('model_id', '').startswith(model_id_prefix):
                        matched = True
                else:
                    matched = True
                
                if not matched:
                    continue
                
                # 提取数据集名称
                data_path = data.get('data_path', data.get('data'))
                if data_path:
                    datasets.add(data_path)
            except (json.JSONDecodeError, Exception):
                continue
    
    return sorted(list(datasets))

def interactive_select_model(result_file=None):
    """
    交互式选择模型（model_id或model）
    
    Args:
        result_file: 结果文件路径
    
    Returns:
        tuple: (model_id, model, model_id_prefix)，如果用户取消则返回 (None, None, None)
    """
    model_ids, models = get_available_models(result_file)
    
    if not model_ids and not models:
        print("\n❌ 未找到任何可用的模型")
        return None, None, None
    
    print("\n" + "="*80)
    print("🔍 请选择要检索的模型")
    print("="*80)
    print("\n选择方式:")
    print("  [1] 通过 model_id 检索")
    print("  [2] 通过 model 名称检索")
    print("  [3] 通过 model_id 前缀检索")
    print("  [4] 手动输入 model_id 或 model")
    print("  [0] 取消")
    print("-"*80)
    
    while True:
        try:
            choice = input("\n请选择方式 (0-4): ").strip()
            
            if choice == '0':
                print("已取消选择")
                return None, None, None
            
            elif choice == '1':
                # 通过model_id检索
                if not model_ids:
                    print("❌ 未找到任何可用的 model_id")
                    continue
                
                print("\n可用的 model_id:")
                print("-"*80)
                for idx, mid in enumerate(model_ids, 1):
                    print(f"  [{idx}] {mid}")
                print("  [0] 返回")
                print("-"*80)
                
                sub_choice = input("\n请输入选项编号 (0-{}): ".format(len(model_ids))).strip()
                if sub_choice == '0':
                    continue
                
                try:
                    sub_choice_num = int(sub_choice)
                    if 1 <= sub_choice_num <= len(model_ids):
                        selected = model_ids[sub_choice_num - 1]
                        print(f"\n✓ 已选择 model_id: {selected}")
                        return selected, None, None
                    else:
                        print(f"❌ 无效的选项")
                except ValueError:
                    print("❌ 请输入有效的数字")
            
            elif choice == '2':
                # 通过model检索
                if not models:
                    print("❌ 未找到任何可用的 model")
                    continue
                
                print("\n可用的 model:")
                print("-"*80)
                for idx, m in enumerate(models, 1):
                    print(f"  [{idx}] {m}")
                print("  [0] 返回")
                print("-"*80)
                
                sub_choice = input("\n请输入选项编号 (0-{}): ".format(len(models))).strip()
                if sub_choice == '0':
                    continue
                
                try:
                    sub_choice_num = int(sub_choice)
                    if 1 <= sub_choice_num <= len(models):
                        selected = models[sub_choice_num - 1]
                        print(f"\n✓ 已选择 model: {selected}")
                        return None, selected, None
                    else:
                        print(f"❌ 无效的选项")
                except ValueError:
                    print("❌ 请输入有效的数字")
            
            elif choice == '3':
                # 通过model_id前缀检索
                if not model_ids:
                    print("❌ 未找到任何可用的 model_id")
                    continue
                
                print("\n可用的 model_id 前缀（基于现有model_id）:")
                print("-"*80)
                # 提取前缀
                prefixes = set()
                for mid in model_ids:
                    parts = mid.split('_')
                    for i in range(1, len(parts) + 1):
                        prefixes.add('_'.join(parts[:i]))
                
                prefixes = sorted(list(prefixes))
                for idx, prefix in enumerate(prefixes, 1):
                    print(f"  [{idx}] {prefix}")
                print("  [0] 返回")
                print("-"*80)
                
                sub_choice = input("\n请输入选项编号 (0-{}): ".format(len(prefixes))).strip()
                if sub_choice == '0':
                    continue
                
                try:
                    sub_choice_num = int(sub_choice)
                    if 1 <= sub_choice_num <= len(prefixes):
                        selected = prefixes[sub_choice_num - 1]
                        print(f"\n✓ 已选择 model_id 前缀: {selected}")
                        return None, None, selected
                    else:
                        print(f"❌ 无效的选项")
                except ValueError:
                    print("❌ 请输入有效的数字")
            
            elif choice == '4':
                # 手动输入
                print("\n请输入 model_id 或 model 名称（留空则检索所有）:")
                user_input = input("> ").strip()
                
                if not user_input:
                    print("\n✓ 将检索所有模型")
                    return None, None, None
                
                # 检查是model_id还是model
                if user_input in model_ids:
                    print(f"\n✓ 已选择 model_id: {user_input}")
                    return user_input, None, None
                elif user_input in models:
                    print(f"\n✓ 已选择 model: {user_input}")
                    return None, user_input, None
                else:
                    # 尝试作为前缀
                    matching_ids = [mid for mid in model_ids if mid.startswith(user_input)]
                    matching_models = [m for m in models if m.startswith(user_input)]
                    
                    if matching_ids:
                        print(f"\n找到匹配的 model_id: {matching_ids[0]}")
                        print(f"✓ 已选择 model_id: {matching_ids[0]}")
                        return matching_ids[0], None, None
                    elif matching_models:
                        print(f"\n找到匹配的 model: {matching_models[0]}")
                        print(f"✓ 已选择 model: {matching_models[0]}")
                        return None, matching_models[0], None
                    else:
                        print(f"\n⚠ 未找到精确匹配，将作为前缀使用: {user_input}")
                        print(f"✓ 已选择 model_id 前缀: {user_input}")
                        return None, None, user_input
            
            else:
                print("❌ 无效的选项，请输入 0-4 之间的数字")
        
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n已取消选择")
            return None, None, None

def interactive_select_dataset(result_file=None, model_id=None, model=None, model_id_prefix=None):
    """
    交互式选择数据集
    
    Args:
        result_file: 结果文件路径
        model_id: 精确匹配的model_id
        model: 精确匹配的model名称
        model_id_prefix: 模型ID前缀
    
    Returns:
        str: 用户选择的数据集名称，如果用户取消则返回 None
    """
    datasets = get_available_datasets(result_file, model_id, model, model_id_prefix)
    
    if not datasets:
        print("\n❌ 未找到任何可用的数据集")
        return None
    
    print("\n" + "="*80)
    print("📊 请选择要分析的数据集")
    print("="*80)
    print("\n可用的数据集:")
    print("-"*80)
    
    for idx, dataset in enumerate(datasets, 1):
        print(f"  [{idx}] {dataset}")
    
    print(f"  [0] 取消")
    print("-"*80)
    
    while True:
        try:
            choice = input("\n请输入选项编号 (0-{}): ".format(len(datasets))).strip()
            
            if choice == '0':
                print("已取消选择")
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(datasets):
                selected = datasets[choice_num - 1]
                print(f"\n✓ 已选择数据集: {selected}")
                return selected
            else:
                print(f"❌ 无效的选项，请输入 0-{len(datasets)} 之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n已取消选择")
            return None

def get_seed_statistics(results):
    """统计所有结果的种子分布"""
    seed_counts = defaultdict(int)
    seed_by_pred_len = defaultdict(lambda: defaultdict(int))
    
    for r in results:
        seed = r.get('seed', 'Unknown')
        pred_len = r.get('pred_len', 'Unknown')
        seed_counts[seed] += 1
        seed_by_pred_len[pred_len][seed] += 1
    
    return seed_counts, seed_by_pred_len

def print_results_by_pred_len(results_by_pred_len, pred_lens=[96, 192, 336, 720], all_results=None, data_path=None):
    """按预测长度打印结果"""
    print("="*80)
    print("T3Time_FreTS_Gated_Qwen 参数寻优结果分析（所有种子）")
    if data_path:
        print(f"数据集: {data_path}")
    print("按预测长度分别分析: {}".format(", ".join(map(str, pred_lens))))
    print("="*80)
    
    # 统计总结果数和种子分布
    total_results = sum(data['count'] for data in results_by_pred_len.values())
    
    if all_results:
        seed_counts, seed_by_pred_len_stats = get_seed_statistics(all_results)
        print(f"\n找到 {total_results} 条实验结果")
        print(f"涉及 {len(seed_counts)} 个不同的种子: {sorted(seed_counts.keys())}")
        print("\n种子分布统计:")
        print(f"{'Seed':<10} {'总实验数':<12}")
        print("-"*25)
        for seed in sorted(seed_counts.keys()):
            print(f"{seed:<10} {seed_counts[seed]:<12}")
    else:
        print(f"\n找到 {total_results} 条实验结果\n")
    
    # 对每个预测长度分别分析
    for pred_len in pred_lens:
        data = results_by_pred_len.get(pred_len, {})
        best_mse = data.get('best_mse')
        best_mae = data.get('best_mae')
        sorted_results_mse = data.get('sorted_results_mse', [])
        sorted_results_mae = data.get('sorted_results_mae', [])
        param_avg = data.get('param_avg', {})
        count = data.get('count', 0)
        
        if not best_mse or not best_mae:
            print("\n" + "="*80)
            print(f"预测长度 {pred_len}: 未找到实验结果")
            print("="*80)
            continue
        
        print("\n" + "="*80)
        print(f"📊 预测长度 {pred_len} (共 {count} 条实验结果)")
        print("="*80)
        
        # 打印该预测长度的最佳结果
        print_single_pred_len_results(best_mse, best_mae, sorted_results_mse, sorted_results_mae, param_avg, pred_len)

def print_single_pred_len_results(best_mse, best_mae, sorted_results_mse, sorted_results_mae, param_avg, pred_len):
    """打印单个预测长度的结果"""
    
    # 最小MSE最佳结果（显示所有详细参数）
    print("\n" + "="*80)
    print(f"🏆 预测长度 {pred_len} - 最小 MSE 参数组合（完整参数）")
    print("="*80)
    print("【架构参数】")
    print(f"  Channel:        {best_mse.get('channel', 'N/A')}")
    print(f"  Head:           {best_mse.get('head', 'N/A')}")
    print(f"  E_Layer:        {best_mse.get('e_layer', 'N/A')}")
    print(f"  D_Layer:        {best_mse.get('d_layer', 'N/A')}")
    print("")
    print("【训练参数】")
    print(f"  Learning_Rate:  {best_mse.get('learning_rate', 'N/A')}")
    print(f"  Weight_Decay:   {best_mse.get('weight_decay', 'N/A')}")
    print(f"  Dropout:        {best_mse.get('dropout_n', 'N/A')}")
    print(f"  Batch_Size:     {best_mse.get('batch_size', 'N/A')}")
    print(f"  Loss_Function:  {best_mse.get('loss_fn', 'N/A')}")
    print(f"  LR_Adjust:      {best_mse.get('lradj', 'N/A')}")
    print("")
    print("【数据参数】")
    print(f"  Data_Path:      {best_mse.get('data_path', 'N/A')}")
    print(f"  Seq_Len:        {best_mse.get('seq_len', 'N/A')}")
    print(f"  Pred_Len:       {best_mse.get('pred_len', 'N/A')}")
    print(f"  Embed_Version:  {best_mse.get('embed_version', 'N/A')}")
    print("")
    print("【训练配置】")
    print(f"  Epochs:         {best_mse.get('epochs', 'N/A')}")
    print(f"  Patience:       {best_mse.get('patience', 'N/A')}")
    print(f"  Seed:           {best_mse.get('seed', 'N/A')} ⭐")
    print("")
    print("【结果指标】")
    print(f"  Test MSE:       {best_mse.get('test_mse', 'N/A'):.6f}")
    print(f"  Test MAE:       {best_mse.get('test_mae', 'N/A'):.6f}")
    print("")
    print("【其他信息】")
    print(f"  Model_ID:       {best_mse.get('model_id', 'N/A')}")
    print(f"  Timestamp:      {best_mse.get('timestamp', 'N/A')}")
    
    # 最小MAE最佳结果（显示所有详细参数）
    print("\n" + "="*80)
    print(f"🏆 预测长度 {pred_len} - 最小 MAE 参数组合（完整参数）")
    print("="*80)
    print("【架构参数】")
    print(f"  Channel:        {best_mae.get('channel', 'N/A')}")
    print(f"  Head:           {best_mae.get('head', 'N/A')}")
    print(f"  E_Layer:        {best_mae.get('e_layer', 'N/A')}")
    print(f"  D_Layer:        {best_mae.get('d_layer', 'N/A')}")
    print("")
    print("【训练参数】")
    print(f"  Learning_Rate:  {best_mae.get('learning_rate', 'N/A')}")
    print(f"  Weight_Decay:   {best_mae.get('weight_decay', 'N/A')}")
    print(f"  Dropout:        {best_mae.get('dropout_n', 'N/A')}")
    print(f"  Batch_Size:     {best_mae.get('batch_size', 'N/A')}")
    print(f"  Loss_Function:  {best_mae.get('loss_fn', 'N/A')}")
    print(f"  LR_Adjust:      {best_mae.get('lradj', 'N/A')}")
    print("")
    print("【数据参数】")
    print(f"  Data_Path:      {best_mae.get('data_path', 'N/A')}")
    print(f"  Seq_Len:        {best_mae.get('seq_len', 'N/A')}")
    print(f"  Pred_Len:       {best_mae.get('pred_len', 'N/A')}")
    print(f"  Embed_Version:  {best_mae.get('embed_version', 'N/A')}")
    print("")
    print("【训练配置】")
    print(f"  Epochs:         {best_mae.get('epochs', 'N/A')}")
    print(f"  Patience:       {best_mae.get('patience', 'N/A')}")
    print(f"  Seed:           {best_mae.get('seed', 'N/A')} ⭐")
    print("")
    print("【结果指标】")
    print(f"  Test MSE:       {best_mae.get('test_mse', 'N/A'):.6f}")
    print(f"  Test MAE:       {best_mae.get('test_mae', 'N/A'):.6f}")
    print("")
    print("【其他信息】")
    print(f"  Model_ID:       {best_mae.get('model_id', 'N/A')}")
    print(f"  Timestamp:      {best_mae.get('timestamp', 'N/A')}")
    
    # 添加命令行参数格式，方便直接使用
    print("\n" + "="*80)
    print(f"📋 预测长度 {pred_len} - 最佳 MSE 参数组合（命令行格式）")
    print("="*80)
    print("python train_frets_gated_qwen.py \\")
    print(f"    --data_path {best_mse.get('data_path', 'ETTh1')} \\")
    print(f"    --seq_len {best_mse.get('seq_len', 96)} \\")
    print(f"    --pred_len {best_mse.get('pred_len', 96)} \\")
    print(f"    --channel {best_mse.get('channel', 'N/A')} \\")
    print(f"    --head {best_mse.get('head', 'N/A')} \\")
    print(f"    --e_layer {best_mse.get('e_layer', 1)} \\")
    print(f"    --d_layer {best_mse.get('d_layer', 1)} \\")
    print(f"    --learning_rate {best_mse.get('learning_rate', 'N/A')} \\")
    print(f"    --weight_decay {best_mse.get('weight_decay', 'N/A')} \\")
    print(f"    --dropout_n {best_mse.get('dropout_n', 'N/A')} \\")
    print(f"    --batch_size {best_mse.get('batch_size', 'N/A')} \\")
    print(f"    --loss_fn {best_mse.get('loss_fn', 'N/A')} \\")
    print(f"    --lradj {best_mse.get('lradj', 'type1')} \\")
    print(f"    --embed_version {best_mse.get('embed_version', 'qwen3_0.6b')} \\")
    print(f"    --epochs {best_mse.get('epochs', 100)} \\")
    print(f"    --es_patience {best_mse.get('patience', 10)} \\")
    print(f"    --seed {best_mse.get('seed', 2088)}")
    
    print("\n" + "="*80)
    print(f"📋 预测长度 {pred_len} - 最佳 MAE 参数组合（命令行格式）")
    print("="*80)
    print("python train_frets_gated_qwen.py \\")
    print(f"    --data_path {best_mae.get('data_path', 'ETTh1')} \\")
    print(f"    --seq_len {best_mae.get('seq_len', 96)} \\")
    print(f"    --pred_len {best_mae.get('pred_len', 96)} \\")
    print(f"    --channel {best_mae.get('channel', 'N/A')} \\")
    print(f"    --head {best_mae.get('head', 'N/A')} \\")
    print(f"    --e_layer {best_mae.get('e_layer', 1)} \\")
    print(f"    --d_layer {best_mae.get('d_layer', 1)} \\")
    print(f"    --learning_rate {best_mae.get('learning_rate', 'N/A')} \\")
    print(f"    --weight_decay {best_mae.get('weight_decay', 'N/A')} \\")
    print(f"    --dropout_n {best_mae.get('dropout_n', 'N/A')} \\")
    print(f"    --batch_size {best_mae.get('batch_size', 'N/A')} \\")
    print(f"    --loss_fn {best_mae.get('loss_fn', 'N/A')} \\")
    print(f"    --lradj {best_mae.get('lradj', 'type1')} \\")
    print(f"    --embed_version {best_mae.get('embed_version', 'qwen3_0.6b')} \\")
    print(f"    --epochs {best_mae.get('epochs', 100)} \\")
    print(f"    --es_patience {best_mae.get('patience', 10)} \\")
    print(f"    --seed {best_mae.get('seed', 2088)}")
    
    # Top 10 最佳结果（按MSE）
    print("\n" + "="*80)
    print(f"预测长度 {pred_len} - Top 10 最佳配置（按 MSE 排序）")
    print("="*80)
    print(f"{'Rank':<6} {'Channel':<10} {'Dropout':<10} {'Head':<8} {'MSE':<15} {'MAE':<15}")
    print("-"*80)
    
    for i, r in enumerate(sorted_results_mse[:10], 1):
        print(f"{i:<6} {r.get('channel', 'N/A'):<10} {r.get('dropout_n', 'N/A'):<10.1f} "
              f"{r.get('head', 'N/A'):<8} {r.get('test_mse', 'N/A'):<15.6f} {r.get('test_mae', 'N/A'):<15.6f}")
    
    # Top 10 最佳结果（按MAE）
    print("\n" + "="*80)
    print(f"预测长度 {pred_len} - Top 10 最佳配置（按 MAE 排序）")
    print("="*80)
    print(f"{'Rank':<6} {'Channel':<10} {'Dropout':<10} {'Head':<8} {'MSE':<15} {'MAE':<15}")
    print("-"*80)
    
    for i, r in enumerate(sorted_results_mae[:10], 1):
        print(f"{i:<6} {r.get('channel', 'N/A'):<10} {r.get('dropout_n', 'N/A'):<10.1f} "
              f"{r.get('head', 'N/A'):<8} {r.get('test_mse', 'N/A'):<15.6f} {r.get('test_mae', 'N/A'):<15.6f}")
    
    # 参数统计分析（按MSE）
    print("\n" + "="*80)
    print(f"预测长度 {pred_len} - 参数统计分析（按平均 MSE 排序）")
    print("="*80)
    print(f"{'Channel':<10} {'Dropout':<10} {'Head':<8} {'平均MSE':<15} {'最小MSE':<15} {'最大MSE':<15} {'次数':<8}")
    print("-"*80)
    
    sorted_params_mse = sorted(param_avg.items(), key=lambda x: x[1]['mse_mean'])
    for (channel, dropout, head), stats in sorted_params_mse[:20]:  # 显示前20个
        print(f"{channel:<10} {dropout:<10.1f} {head:<8} "
              f"{stats['mse_mean']:<15.6f} {stats['mse_min']:<15.6f} {stats['mse_max']:<15.6f} {stats['count']:<8}")
    
    # 参数统计分析（按MAE）
    print("\n" + "="*80)
    print(f"预测长度 {pred_len} - 参数统计分析（按平均 MAE 排序）")
    print("="*80)
    print(f"{'Channel':<10} {'Dropout':<10} {'Head':<8} {'平均MAE':<15} {'最小MAE':<15} {'最大MAE':<15} {'次数':<8}")
    print("-"*80)
    
    sorted_params_mae = sorted(param_avg.items(), key=lambda x: x[1]['mae_mean'])
    for (channel, dropout, head), stats in sorted_params_mae[:20]:  # 显示前20个
        print(f"{channel:<10} {dropout:<10.1f} {head:<8} "
              f"{stats['mae_mean']:<15.6f} {stats['mae_min']:<15.6f} {stats['mae_max']:<15.6f} {stats['count']:<8}")
    
    # 各参数维度分析
    print("\n" + "="*80)
    print(f"预测长度 {pred_len} - 各参数维度分析（MSE）")
    print("="*80)
    
    # Channel 分析（MSE）
    channel_stats_mse = defaultdict(list)
    channel_stats_mae = defaultdict(list)
    for r in sorted_results_mse:
        channel_stats_mse[r.get('channel')].append(r.get('test_mse', float('inf')))
        channel_stats_mae[r.get('channel')].append(r.get('test_mae', float('inf')))
    
    print("\n[1] Channel 参数分析（MSE）:")
    print(f"{'Channel':<10} {'平均MSE':<15} {'最小MSE':<15} {'最大MSE':<15} {'次数':<8}")
    print("-"*80)
    for channel in sorted(channel_stats_mse.keys()):
        mse_list = channel_stats_mse[channel]
        print(f"{channel:<10} {sum(mse_list)/len(mse_list):<15.6f} "
              f"{min(mse_list):<15.6f} {max(mse_list):<15.6f} {len(mse_list):<8}")
    
    # Dropout 分析（MSE）
    dropout_stats_mse = defaultdict(list)
    dropout_stats_mae = defaultdict(list)
    for r in sorted_results_mse:
        dropout_stats_mse[r.get('dropout_n')].append(r.get('test_mse', float('inf')))
        dropout_stats_mae[r.get('dropout_n')].append(r.get('test_mae', float('inf')))
    
    print("\n[2] Dropout 参数分析（MSE）:")
    print(f"{'Dropout':<10} {'平均MSE':<15} {'最小MSE':<15} {'最大MSE':<15} {'次数':<8}")
    print("-"*80)
    for dropout in sorted(dropout_stats_mse.keys()):
        mse_list = dropout_stats_mse[dropout]
        print(f"{dropout:<10.1f} {sum(mse_list)/len(mse_list):<15.6f} "
              f"{min(mse_list):<15.6f} {max(mse_list):<15.6f} {len(mse_list):<8}")
    
    # Head 分析（MSE）
    head_stats_mse = defaultdict(list)
    head_stats_mae = defaultdict(list)
    for r in sorted_results_mse:
        head_stats_mse[r.get('head')].append(r.get('test_mse', float('inf')))
        head_stats_mae[r.get('head')].append(r.get('test_mae', float('inf')))
    
    print("\n[3] Head 参数分析（MSE）:")
    print(f"{'Head':<10} {'平均MSE':<15} {'最小MSE':<15} {'最大MSE':<15} {'次数':<8}")
    print("-"*80)
    for head in sorted(head_stats_mse.keys()):
        mse_list = head_stats_mse[head]
        print(f"{head:<10} {sum(mse_list)/len(mse_list):<15.6f} "
              f"{min(mse_list):<15.6f} {max(mse_list):<15.6f} {len(mse_list):<8}")
    
    # 各参数维度分析（MAE）
    print("\n" + "="*80)
    print(f"预测长度 {pred_len} - 各参数维度分析（MAE）")
    print("="*80)
    
    print("\n[1] Channel 参数分析（MAE）:")
    print(f"{'Channel':<10} {'平均MAE':<15} {'最小MAE':<15} {'最大MAE':<15} {'次数':<8}")
    print("-"*80)
    for channel in sorted(channel_stats_mae.keys()):
        mae_list = channel_stats_mae[channel]
        print(f"{channel:<10} {sum(mae_list)/len(mae_list):<15.6f} "
              f"{min(mae_list):<15.6f} {max(mae_list):<15.6f} {len(mae_list):<8}")
    
    print("\n[2] Dropout 参数分析（MAE）:")
    print(f"{'Dropout':<10} {'平均MAE':<15} {'最小MAE':<15} {'最大MAE':<15} {'次数':<8}")
    print("-"*80)
    for dropout in sorted(dropout_stats_mae.keys()):
        mae_list = dropout_stats_mae[dropout]
        print(f"{dropout:<10.1f} {sum(mae_list)/len(mae_list):<15.6f} "
              f"{min(mae_list):<15.6f} {max(mae_list):<15.6f} {len(mae_list):<8}")
    
    print("\n[3] Head 参数分析（MAE）:")
    print(f"{'Head':<10} {'平均MAE':<15} {'最小MAE':<15} {'最大MAE':<15} {'次数':<8}")
    print("-"*80)
    for head in sorted(head_stats_mae.keys()):
        mae_list = head_stats_mae[head]
        print(f"{head:<10} {sum(mae_list)/len(mae_list):<15.6f} "
              f"{min(mae_list):<15.6f} {max(mae_list):<15.6f} {len(mae_list):<8}")
    
    # 最佳参数组合（按平均MSE）
    if param_avg:
        best_param_avg_mse = min(param_avg.items(), key=lambda x: x[1]['mse_mean'])
        (best_channel_mse, best_dropout_mse, best_head_mse), best_stats_mse = best_param_avg_mse
        
        print("\n" + "="*80)
        print(f"预测长度 {pred_len} - 🏆 最佳参数组合（按平均 MSE）")
        print("="*80)
        print(f"Channel:     {best_channel_mse}")
        print(f"Dropout:     {best_dropout_mse}")
        print(f"Head:        {best_head_mse}")
        print(f"平均 MSE:    {best_stats_mse['mse_mean']:.6f}")
        print(f"最小 MSE:    {best_stats_mse['mse_min']:.6f}")
        print(f"最大 MSE:    {best_stats_mse['mse_max']:.6f}")
        print(f"平均 MAE:    {best_stats_mse['mae_mean']:.6f}")
        print(f"最小 MAE:    {best_stats_mse['mae_min']:.6f}")
        print(f"最大 MAE:    {best_stats_mse['mae_max']:.6f}")
        print(f"实验次数:    {best_stats_mse['count']}")
        
        # 最佳参数组合（按平均MAE）
        best_param_avg_mae = min(param_avg.items(), key=lambda x: x[1]['mae_mean'])
        (best_channel_mae, best_dropout_mae, best_head_mae), best_stats_mae = best_param_avg_mae
        
        print("\n" + "="*80)
        print(f"预测长度 {pred_len} - 🏆 最佳参数组合（按平均 MAE）")
        print("="*80)
        print(f"Channel:     {best_channel_mae}")
        print(f"Dropout:     {best_dropout_mae}")
        print(f"Head:        {best_head_mae}")
        print(f"平均 MSE:    {best_stats_mae['mse_mean']:.6f}")
        print(f"最小 MSE:    {best_stats_mae['mse_min']:.6f}")
        print(f"最大 MSE:    {best_stats_mae['mse_max']:.6f}")
        print(f"平均 MAE:    {best_stats_mae['mae_mean']:.6f}")
        print(f"最小 MAE:    {best_stats_mae['mae_min']:.6f}")
        print(f"最大 MAE:    {best_stats_mae['mae_max']:.6f}")
        print(f"实验次数:    {best_stats_mae['count']}")

def print_summary_table(results_by_pred_len, pred_lens=[96, 192, 336, 720], data_path=None, 
                        model_id=None, model=None, model_id_prefix=None):
    """打印所有预测长度的汇总表格"""
    print("\n" + "="*80)
    print("📊 所有预测长度的最佳结果汇总（跨所有种子）")
    if data_path:
        print(f"数据集: {data_path}")
    
    # 显示模型信息
    model_info = []
    if model_id:
        model_info.append(f"model_id: {model_id}")
    if model:
        model_info.append(f"model: {model}")
    if model_id_prefix:
        model_info.append(f"model_id_prefix: {model_id_prefix}")
    if model_info:
        print(f"模型: {', '.join(model_info)}")
    
    print("="*80)
    
    # MSE 汇总（添加综合均值）
    print("\n【最小 MSE 汇总】")
    print(f"{'Pred_Len':<12} {'Seed':<8} {'Channel':<10} {'Dropout':<10} {'Head':<8} {'LR':<12} {'WD':<12} {'BS':<8} {'MSE':<15} {'MAE':<15}")
    print("-"*110)
    
    mse_values = []
    mae_values = []
    
    for pred_len in pred_lens:
        data = results_by_pred_len.get(pred_len, {})
        best_mse = data.get('best_mse')
        if best_mse:
            seed = best_mse.get('seed', 'N/A')
            mse_val = best_mse.get('test_mse')
            mae_val = best_mse.get('test_mae')
            
            if mse_val is not None:
                mse_values.append(mse_val)
            if mae_val is not None:
                mae_values.append(mae_val)
            
            print(f"{pred_len:<12} {seed:<8} {best_mse.get('channel', 'N/A'):<10} "
                  f"{best_mse.get('dropout_n', 'N/A'):<10.1f} {best_mse.get('head', 'N/A'):<8} "
                  f"{best_mse.get('learning_rate', 'N/A'):<12} {best_mse.get('weight_decay', 'N/A'):<12} "
                  f"{best_mse.get('batch_size', 'N/A'):<8} "
                  f"{mse_val:<15.6f} {mae_val:<15.6f}")
        else:
            print(f"{pred_len:<12} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<12} {'N/A':<12} {'N/A':<8} {'N/A':<15} {'N/A':<15}")
    
    # 显示综合均值
    if mse_values and mae_values:
        mse_avg = sum(mse_values) / len(mse_values)
        mae_avg = sum(mae_values) / len(mae_values)
        print("-"*110)
        print(f"{'综合均值':<12} {'':<8} {'':<10} {'':<10} {'':<8} {'':<12} {'':<12} {'':<8} "
              f"{mse_avg:<15.6f} {mae_avg:<15.6f}")
    
    # MAE 汇总（添加综合均值）
    print("\n【最小 MAE 汇总】")
    print(f"{'Pred_Len':<12} {'Seed':<8} {'Channel':<10} {'Dropout':<10} {'Head':<8} {'LR':<12} {'WD':<12} {'BS':<8} {'MSE':<15} {'MAE':<15}")
    print("-"*110)
    
    mse_values_mae = []
    mae_values_mae = []
    
    for pred_len in pred_lens:
        data = results_by_pred_len.get(pred_len, {})
        best_mae = data.get('best_mae')
        if best_mae:
            seed = best_mae.get('seed', 'N/A')
            mse_val = best_mae.get('test_mse')
            mae_val = best_mae.get('test_mae')
            
            if mse_val is not None:
                mse_values_mae.append(mse_val)
            if mae_val is not None:
                mae_values_mae.append(mae_val)
            
            print(f"{pred_len:<12} {seed:<8} {best_mae.get('channel', 'N/A'):<10} "
                  f"{best_mae.get('dropout_n', 'N/A'):<10.1f} {best_mae.get('head', 'N/A'):<8} "
                  f"{best_mae.get('learning_rate', 'N/A'):<12} {best_mae.get('weight_decay', 'N/A'):<12} "
                  f"{best_mae.get('batch_size', 'N/A'):<8} "
                  f"{mse_val:<15.6f} {mae_val:<15.6f}")
        else:
            print(f"{pred_len:<12} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<12} {'N/A':<12} {'N/A':<8} {'N/A':<15} {'N/A':<15}")
    
    # 显示综合均值
    if mse_values_mae and mae_values_mae:
        mse_avg_mae = sum(mse_values_mae) / len(mse_values_mae)
        mae_avg_mae = sum(mae_values_mae) / len(mae_values_mae)
        print("-"*110)
        print(f"{'综合均值':<12} {'':<8} {'':<10} {'':<10} {'':<8} {'':<12} {'':<12} {'':<8} "
              f"{mse_avg_mae:<15.6f} {mae_avg_mae:<15.6f}")

def export_best_configs_to_json(results_by_pred_len, pred_lens=[96, 192, 336, 720], data_path=None, output_file=None):
    """
    导出最佳MSE和最佳MAE的配置到JSON文件
    
    Args:
        results_by_pred_len: 按预测长度分组的结果
        pred_lens: 预测长度列表
        data_path: 数据集名称
        output_file: 输出文件路径，如果为None则自动生成
    """
    if output_file is None:
        if data_path:
            output_file = f"best_configs_{data_path}.json"
        else:
            output_file = "best_configs.json"
    
    best_configs = {
        "dataset": data_path,
        "export_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_mse_configs": [],
        "best_mae_configs": []
    }
    
    # 收集最佳MSE配置
    for pred_len in pred_lens:
        data = results_by_pred_len.get(pred_len, {})
        best_mse = data.get('best_mse')
        if best_mse:
            config = {
                "pred_len": pred_len,
                "seed": best_mse.get('seed'),
                "channel": best_mse.get('channel'),
                "dropout_n": best_mse.get('dropout_n'),
                "head": best_mse.get('head'),
                "e_layer": best_mse.get('e_layer'),
                "d_layer": best_mse.get('d_layer'),
                "learning_rate": best_mse.get('learning_rate'),
                "weight_decay": best_mse.get('weight_decay'),
                "batch_size": best_mse.get('batch_size'),
                "seq_len": best_mse.get('seq_len'),
                "loss_fn": best_mse.get('loss_fn'),
                "lradj": best_mse.get('lradj'),
                "embed_version": best_mse.get('embed_version'),
                "epochs": best_mse.get('epochs'),
                "patience": best_mse.get('patience'),
                "test_mse": best_mse.get('test_mse'),
                "test_mae": best_mse.get('test_mae'),
                "timestamp": best_mse.get('timestamp'),
                "model_id": best_mse.get('model_id')
            }
            best_configs["best_mse_configs"].append(config)
    
    # 收集最佳MAE配置
    for pred_len in pred_lens:
        data = results_by_pred_len.get(pred_len, {})
        best_mae = data.get('best_mae')
        if best_mae:
            config = {
                "pred_len": pred_len,
                "seed": best_mae.get('seed'),
                "channel": best_mae.get('channel'),
                "dropout_n": best_mae.get('dropout_n'),
                "head": best_mae.get('head'),
                "e_layer": best_mae.get('e_layer'),
                "d_layer": best_mae.get('d_layer'),
                "learning_rate": best_mae.get('learning_rate'),
                "weight_decay": best_mae.get('weight_decay'),
                "batch_size": best_mae.get('batch_size'),
                "seq_len": best_mae.get('seq_len'),
                "loss_fn": best_mae.get('loss_fn'),
                "lradj": best_mae.get('lradj'),
                "embed_version": best_mae.get('embed_version'),
                "epochs": best_mae.get('epochs'),
                "patience": best_mae.get('patience'),
                "test_mse": best_mae.get('test_mse'),
                "test_mae": best_mae.get('test_mae'),
                "timestamp": best_mae.get('timestamp'),
                "model_id": best_mae.get('model_id')
            }
            best_configs["best_mae_configs"].append(config)
    
    # 保存到JSON文件
    output_path = os.path.join(project_root, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(best_configs, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print(f"✅ 最佳配置已导出到: {output_path}")
    print(f"   - 最佳MSE配置数量: {len(best_configs['best_mse_configs'])}")
    print(f"   - 最佳MAE配置数量: {len(best_configs['best_mae_configs'])}")
    print("="*80)
    
    return output_path

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='检索 T3Time_FreTS_FusionExp 模型的所有种子的参数寻优实验结果（按预测长度分别分析）'
    )
    parser.add_argument('--result_file', type=str, default=None,
                        help='结果文件路径（默认: experiment_results.log）')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子（默认: None，分析所有种子）')
    parser.add_argument('--model_id', type=str, default=None,
                        help='精确匹配的model_id')
    parser.add_argument('--model', type=str, default=None,
                        help='精确匹配的model名称')
    parser.add_argument('--model_id_prefix', type=str, default=None,
                        help='模型ID前缀（如果未指定model_id和model，且未交互选择，则使用默认值）')
    parser.add_argument('--pred_lens', type=int, nargs='+',
                        default=[96, 192, 336, 720],
                        help='要分析的预测长度列表（默认: 96 192 336 720）')
    parser.add_argument('--data_path', type=str, default=None,
                        help='数据集名称（默认: None，将显示交互式菜单选择；例如: ETTh1, ETTh2, ETTm1, ETTm2）')
    
    args = parser.parse_args()
    
    # 如果未指定模型，则显示交互式菜单
    model_id = args.model_id
    model = args.model
    model_id_prefix = args.model_id_prefix
    
    if model_id is None and model is None and model_id_prefix is None:
        model_id, model, model_id_prefix = interactive_select_model(result_file=args.result_file)
        if model_id is None and model is None and model_id_prefix is None:
            print("\n程序已退出")
            return
    
    # 如果仍未指定，使用默认前缀
    if model_id is None and model is None and model_id_prefix is None:
        model_id_prefix = 'T3Time_FreTS_Gated_Qwen_Hyperopt'
        print(f"\n使用默认 model_id 前缀: {model_id_prefix}")
    
    # 如果未指定数据集，则显示交互式菜单
    data_path = args.data_path
    if data_path is None:
        data_path = interactive_select_dataset(
            result_file=args.result_file,
            model_id=model_id,
            model=model,
            model_id_prefix=model_id_prefix
        )
        if data_path is None:
            print("\n程序已退出")
            return
    
    results = load_hyperopt_results(
        result_file=args.result_file,
        seed=args.seed,
        model_id=model_id,
        model=model,
        model_id_prefix=model_id_prefix,
        data_path=data_path,
    )
    
    if not results:
        model_desc = ""
        if model_id:
            model_desc = f"model_id={model_id}"
        elif model:
            model_desc = f"model={model}"
        elif model_id_prefix:
            model_desc = f"model_id_prefix={model_id_prefix}"
        else:
            model_desc = "所有模型"
        
        if args.seed is None:
            print(f"\n❌ 未找到 {model_desc} 的任何实验结果")
        else:
            print(f"\n❌ 未找到 {model_desc}, seed={args.seed} 的实验结果")
        print("请检查日志文件或运行相应的实验脚本")
        return
    
    # 按预测长度分组分析
    results_by_pred_len = find_best_params_by_pred_len(results, args.pred_lens)
    
    # 打印汇总表格
    print_summary_table(results_by_pred_len, args.pred_lens, data_path=data_path,
                       model_id=model_id, model=model, model_id_prefix=model_id_prefix)
    
    # 导出最佳配置到JSON文件
    export_best_configs_to_json(results_by_pred_len, args.pred_lens, data_path=data_path)
    
    # 打印每个预测长度的详细结果（传入所有结果用于种子统计）
    # print_results_by_pred_len(results_by_pred_len, args.pred_lens, all_results=results, data_path=data_path)
    
    # print("\n" + "="*80)
    # print("分析完成！")
    # print("="*80)

if __name__ == "__main__":
    main()
