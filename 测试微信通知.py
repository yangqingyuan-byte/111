#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微信通知测试脚本

直接读取 .gpu_monitor_config 中的加密 token，发送一条测试消息。
"""
import base64
import getpass
import os
import socket
import sys
from datetime import datetime

from notify_wechat import WeChatNotifier

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ModuleNotFoundError as exc:
    missing_module = exc.name or "cryptography"
    print(
        "❌ 缺少依赖模块: "
        f"{missing_module}\n"
        "请先安装:\n"
        "  conda activate TimeCMA_Qwen3\n"
        "  pip install cryptography"
    )
    sys.exit(1)


def generate_key_from_password(password: bytes, salt: bytes) -> bytes:
    """从密码生成加密密钥。"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    return base64.urlsafe_b64encode(kdf.derive(password))


def decrypt_token(config_file: str = ".gpu_monitor_config") -> str:
    """从加密配置文件读取并解密 token。"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"配置文件不存在: {config_file}\n"
            "请先运行: python3 setup_gpu_monitor.py --token YOUR_TOKEN"
        )

    with open(config_file, "rb") as f:
        data = f.read()

    parts = data.split(b"\n", 1)
    if len(parts) != 2:
        raise ValueError("配置文件格式错误")

    salt, encrypted_token = parts
    password = f"{socket.gethostname()}_{getpass.getuser()}_gpu_monitor_2026"
    key = generate_key_from_password(password.encode("utf-8"), salt)
    fernet = Fernet(key)

    try:
        decrypted_token = fernet.decrypt(encrypted_token)
        return decrypted_token.decode("utf-8")
    except Exception as exc:
        raise ValueError(f"解密失败: {exc}\n可能是配置文件损坏或在不同机器上运行") from exc


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="微信通知测试脚本")
    parser.add_argument("--config", default=".gpu_monitor_config", help="加密配置文件路径")
    parser.add_argument("--title", default="T3Time 微信通知测试", help="通知标题")
    parser.add_argument(
        "--body",
        default=None,
        help="通知内容；不传则自动生成测试文案",
    )
    args = parser.parse_args()

    try:
        print("🔐 正在读取加密 token...")
        token = decrypt_token(args.config)
        print("✅ Token 读取成功")
    except Exception as exc:
        print(f"❌ 读取 token 失败: {exc}")
        return 1

    body = args.body or (
        "这是一条来自 T3Time 的微信测试通知。\n\n"
        f"发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"主机名: {socket.gethostname()}\n"
        "如果你能收到这条消息，说明通知链路是通的。"
    )

    try:
        notifier = WeChatNotifier(method="serverchan", sendkey=token)
        success, msg = notifier.send(args.title, body)
    except Exception as exc:
        print(f"❌ 初始化或发送失败: {exc}")
        return 1

    if success:
        print(f"✅ 微信通知发送成功: {msg}")
        return 0

    print(f"❌ 微信通知发送失败: {msg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
