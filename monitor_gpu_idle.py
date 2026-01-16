#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 空闲监控脚本 - 当所有 GPU 使用率为 0% 时发送微信通知
"""
import subprocess
import time
import sys
import os
from datetime import datetime
from notify_wechat import WeChatNotifier

def get_gpu_utilization():
    """
    获取所有 GPU 的使用率
    
    Returns:
        list: 每个 GPU 的使用率列表，例如 [0, 5, 0, 0] 表示 4 个 GPU 的使用率
    """
    try:
        # 使用 nvidia-smi 查询 GPU 使用率
        # --query-gpu=utilization.gpu 查询 GPU 使用率
        # --format=csv,noheader,nounits 输出格式为 CSV，无表头，无单位
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"❌ nvidia-smi 执行失败: {result.stderr}")
            return None
        
        # 解析输出，每行一个 GPU 的使用率
        utilizations = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                try:
                    util = int(line.strip())
                    utilizations.append(util)
                except ValueError:
                    print(f"⚠️ 无法解析 GPU 使用率: {line}")
                    return None
        
        return utilizations
    
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi 执行超时")
        return None
    except FileNotFoundError:
        print("❌ 未找到 nvidia-smi 命令，请确保已安装 NVIDIA 驱动")
        return None
    except Exception as e:
        print(f"❌ 获取 GPU 使用率时出错: {e}")
        return None

def check_all_gpus_idle(utilizations, threshold=0):
    """
    检查是否所有 GPU 都空闲（使用率 <= threshold）
    
    Args:
        utilizations: GPU 使用率列表
        threshold: 使用率阈值，默认 0
    
    Returns:
        bool: 如果所有 GPU 使用率都 <= threshold，返回 True
    """
    if not utilizations:
        return False
    
    return all(util <= threshold for util in utilizations)

def format_gpu_status(utilizations):
    """
    格式化 GPU 状态信息
    
    Args:
        utilizations: GPU 使用率列表
    
    Returns:
        str: 格式化的状态字符串
    """
    if not utilizations:
        return "无法获取 GPU 状态"
    
    status_lines = []
    for i, util in enumerate(utilizations):
        status = "🟢 空闲" if util == 0 else f"🟡 {util}%"
        status_lines.append(f"GPU {i}: {status}")
    
    return "\n".join(status_lines)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU 空闲监控脚本')
    parser.add_argument('--interval', type=int, default=60,
                       help='检查间隔（秒），默认 60 秒')
    parser.add_argument('--threshold', type=int, default=0,
                       help='使用率阈值（%），默认 0，即完全空闲')
    parser.add_argument('--check-count', type=int, default=3,
                       help='连续检查次数，默认 3 次（避免误报）')
    parser.add_argument('--method', choices=['serverchan', 'qywx'], default='serverchan',
                       help='通知方式: serverchan (虾推啥) 或 qywx (企业微信)')
    
    # 通知参数（从环境变量读取）
    parser.add_argument('--sendkey', help='虾推啥 token (或设置环境变量 SENDKEY)')
    parser.add_argument('--corpid', help='企业微信 CorpID (或设置环境变量 QYWX_CORPID)')
    parser.add_argument('--corpsecret', help='企业微信 CorpSecret (或设置环境变量 QYWX_CORPSECRET)')
    parser.add_argument('--agentid', help='企业微信 AgentID (或设置环境变量 QYWX_AGENTID)')
    
    args = parser.parse_args()
    
    # 初始化通知器
    try:
        if args.method == 'serverchan':
            sendkey = args.sendkey or os.getenv('SENDKEY')
            if not sendkey:
                print("❌ 错误: 虾推啥方式需要提供 --sendkey 或设置环境变量 SENDKEY")
                return 1
            notifier = WeChatNotifier(method='serverchan', sendkey=sendkey)
        else:
            corpid = args.corpid or os.getenv('QYWX_CORPID')
            corpsecret = args.corpsecret or os.getenv('QYWX_CORPSECRET')
            agentid = args.agentid or os.getenv('QYWX_AGENTID')
            if not all([corpid, corpsecret, agentid]):
                print("❌ 错误: 企业微信方式需要提供 corpid, corpsecret, agentid")
                return 1
            notifier = WeChatNotifier(method='qywx', corpid=corpid, corpsecret=corpsecret, agentid=agentid)
    except Exception as e:
        print(f"❌ 初始化通知器失败: {e}")
        return 1
    
    print("=" * 50)
    print("🚀 GPU 空闲监控脚本启动")
    print("=" * 50)
    print(f"检查间隔: {args.interval} 秒")
    print(f"使用率阈值: {args.threshold}%")
    print(f"连续检查次数: {args.check_count} 次")
    print(f"通知方式: {args.method}")
    print("=" * 50)
    print("按 Ctrl+C 退出")
    print()
    
    idle_count = 0  # 连续空闲次数
    last_notify_time = None  # 上次通知时间
    notify_cooldown = 3600  # 通知冷却时间（秒），避免频繁通知
    
    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 获取 GPU 使用率
            utilizations = get_gpu_utilization()
            
            if utilizations is None:
                print(f"[{timestamp}] ⚠️ 无法获取 GPU 使用率，等待 {args.interval} 秒后重试...")
                time.sleep(args.interval)
                continue
            
            # 检查是否所有 GPU 都空闲
            all_idle = check_all_gpus_idle(utilizations, args.threshold)
            
            # 显示当前状态
            status = format_gpu_status(utilizations)
            print(f"[{timestamp}]")
            print(status)
            
            if all_idle:
                idle_count += 1
                print(f"✅ 所有 GPU 空闲 (连续 {idle_count}/{args.check_count} 次)")
                
                # 如果连续空闲达到指定次数，且距离上次通知超过冷却时间
                if idle_count >= args.check_count:
                    current_time = time.time()
                    if last_notify_time is None or (current_time - last_notify_time) >= notify_cooldown:
                        # 发送通知
                        title = "🎉 所有 GPU 已空闲"
                        body = f"""所有 GPU 使用率已降至 {args.threshold}% 以下

📊 GPU 状态:
{status}

⏰ 检测时间: {timestamp}
🔢 连续空闲次数: {idle_count} 次
⏱️ 检查间隔: {args.interval} 秒

所有训练任务可能已完成，请检查实验状态。
"""
                        
                        success, msg = notifier.send(title, body)
                        if success:
                            print(f"✅ 微信通知已发送: {msg}")
                            last_notify_time = current_time
                            idle_count = 0  # 重置计数，避免重复通知
                        else:
                            print(f"❌ 微信通知发送失败: {msg}")
                    else:
                        remaining_cooldown = int(notify_cooldown - (current_time - last_notify_time))
                        print(f"⏳ 通知冷却中，还需等待 {remaining_cooldown} 秒")
            else:
                idle_count = 0  # 重置计数
                max_util = max(utilizations) if utilizations else 0
                print(f"🔄 GPU 正在使用中 (最高使用率: {max_util}%)")
            
            print(f"下次检查: {args.interval} 秒后\n")
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 50)
        print("👋 监控脚本已停止")
        print("=" * 50)
        return 0
    except Exception as e:
        print(f"\n❌ 监控脚本异常: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
