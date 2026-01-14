#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 空闲监控脚本

功能：
- 每隔一段时间检查一次所有 GPU 的利用率（依赖 nvidia-smi）
- 如果连续 N 次检测到「所有 GPU 利用率都为 0%」，则通过微信通知脚本 notify_wechat.py 提醒
- 适合挂在 screen / tmux 里后台运行
"""

import argparse
import subprocess
import time
from datetime import datetime
import os

from notify_wechat import WeChatNotifier


def get_gpu_utils():
    """
    调用 nvidia-smi 获取所有 GPU 的利用率（百分比整数）。
    返回值示例：[0, 0, 35, 100]
    """
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        lines = output.decode("utf-8").strip().splitlines()
        utils = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                utils.append(int(line))
            except ValueError:
                # 非法行，忽略
                continue
        return utils
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now()}] 获取 GPU 利用率失败: {e.output.decode('utf-8', errors='ignore')}")
    except FileNotFoundError:
        print(f"[{datetime.now()}] 未找到 nvidia-smi，请确认已安装 NVIDIA 驱动和 CUDA 工具链。")
    except Exception as e:
        print(f"[{datetime.now()}] 获取 GPU 利用率异常: {e}")
    return []


def build_notifier(method: str) -> WeChatNotifier:
    """
    根据 method 和环境变量构建 WeChatNotifier。
    为了简单起见，这里只从环境变量读取配置：
      - serverchan: SENDKEY
      - qywx: QYWX_CORPID, QYWX_CORPSECRET, QYWX_AGENTID
    """
    if method == "serverchan":
        sendkey = os.getenv("SENDKEY")
        if not sendkey:
            raise RuntimeError("serverchan 模式需要环境变量 SENDKEY，请先在运行 screen 前导出该变量。")
        return WeChatNotifier(method="serverchan", sendkey=sendkey)
    else:
        corpid = os.getenv("QYWX_CORPID")
        corpsecret = os.getenv("QYWX_CORPSECRET")
        agentid = os.getenv("QYWX_AGENTID")
        if not all([corpid, corpsecret, agentid]):
            raise RuntimeError(
                "qywx 模式需要环境变量 QYWX_CORPID / QYWX_CORPSECRET / QYWX_AGENTID，"
                "请先在运行 screen 前导出这些变量。"
            )
        return WeChatNotifier(
            method="qywx",
            corpid=corpid,
            corpsecret=corpsecret,
            agentid=agentid,
        )


def send_notify(method: str, title: str, body: str):
    try:
        notifier = build_notifier(method)
        ok, msg = notifier.send(title, body)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if ok:
            print(f"[{ts}] ✅ 通知发送成功: {msg}")
        else:
            print(f"[{ts}] ❌ 通知发送失败: {msg}")
    except Exception as e:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] ❌ 构建/发送通知异常: {e}")


def main():
    parser = argparse.ArgumentParser(description="监控 GPU 利用率，连续若干次空闲后发送微信通知")
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="检测间隔秒数，默认 60 秒",
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        default=5,
        help="连续多少次所有 GPU 利用率为 0% 视为任务结束，默认 5 次",
    )
    parser.add_argument(
        "--method",
        choices=["serverchan", "qywx"],
        default="serverchan",
        help="通知方式：serverchan(虾推啥/xtuis) 或 qywx(企业微信)，默认 serverchan",
    )
    args = parser.parse_args()

    interval = max(5, args.interval)  # 间隔至少 5 秒，避免刷太快
    consecutive_target = max(1, args.consecutive)

    zero_streak = 0

    print(
        f"[{datetime.now()}] 启动 GPU 监控：间隔 {interval}s，"
        f"连续 {consecutive_target} 次所有 GPU 利用率为 0% 将发送通知并退出..."
    )

    try:
        while True:
            utils = get_gpu_utils()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if not utils:
                print(f"[{ts}] 未获取到 GPU 利用率，这次检测忽略。")
                time.sleep(interval)
                continue

            all_zero = all(u == 0 for u in utils)
            utils_str = ", ".join(f"{i}:{u}%" for i, u in enumerate(utils))

            if all_zero:
                zero_streak += 1
                print(
                    f"[{ts}] 当前所有 GPU 利用率为 0%（{utils_str}），"
                    f"连续空闲计数：{zero_streak}/{consecutive_target}"
                )
            else:
                if zero_streak > 0:
                    print(
                        f"[{ts}] 检测到 GPU 再次使用（{utils_str}），连续空闲计数清零。"
                    )
                else:
                    print(f"[{ts}] 当前 GPU 利用率：{utils_str}")
                zero_streak = 0

            if zero_streak >= consecutive_target:
                title = "GPU 任务似乎已经全部结束"
                body = (
                    f"连续 {consecutive_target} 次检测到所有 GPU 利用率为 0%。\n"
                    f"最后一次检测时间：{ts}\n"
                    f"各卡利用率：{utils_str}"
                )
                print(f"[{ts}] 条件满足，发送通知并退出监控脚本。")
                send_notify(args.method, title, body)
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] 收到中断信号，退出监控。")


if __name__ == "__main__":
    main()

