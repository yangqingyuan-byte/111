#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 监控配置脚本 - 加密保存 token
"""
import base64
import os
import getpass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

def generate_key_from_password(password: bytes, salt: bytes) -> bytes:
    """从密码生成加密密钥"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return base64.urlsafe_b64encode(kdf.derive(password))

def encrypt_token(token: str, password: str = None) -> tuple:
    """
    加密 token
    
    Returns:
        tuple: (加密后的 token, salt)
    """
    if password is None:
        # 使用默认密码（基于主机名和用户名，增加安全性）
        import socket
        hostname = socket.gethostname()
        username = getpass.getuser()
        password = f"{hostname}_{username}_gpu_monitor_2026"
    
    password_bytes = password.encode('utf-8')
    salt = os.urandom(16)  # 生成随机 salt
    
    key = generate_key_from_password(password_bytes, salt)
    fernet = Fernet(key)
    
    encrypted_token = fernet.encrypt(token.encode('utf-8'))
    
    return encrypted_token, salt

def save_encrypted_token(token: str, config_file: str = '.gpu_monitor_config'):
    """保存加密的 token 到文件"""
    encrypted_token, salt = encrypt_token(token)
    
    # 将 salt 和加密的 token 一起保存
    with open(config_file, 'wb') as f:
        f.write(salt + b'\n' + encrypted_token)
    
    # 设置文件权限，只有所有者可读
    os.chmod(config_file, 0o600)
    
    print(f"✅ Token 已加密保存到: {config_file}")
    print(f"✅ 文件权限已设置为 600 (仅所有者可读)")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU 监控配置脚本 - 加密保存 token')
    parser.add_argument('--token', help='虾推啥 token (如果不提供，将从环境变量 SENDKEY 读取)')
    parser.add_argument('--config', default='.gpu_monitor_config',
                       help='配置文件路径，默认 .gpu_monitor_config')
    
    args = parser.parse_args()
    
    # 获取 token
    token = args.token or os.getenv('SENDKEY')
    
    if not token:
        print("❌ 错误: 请提供 token (通过 --token 参数或设置环境变量 SENDKEY)")
        print("\n使用方法:")
        print("  python3 setup_gpu_monitor.py --token YOUR_TOKEN")
        print("  或")
        print("  export SENDKEY=YOUR_TOKEN")
        print("  python3 setup_gpu_monitor.py")
        return 1
    
    # 保存加密的 token
    save_encrypted_token(token, args.config)
    
    print("\n✅ 配置完成！现在可以运行监控脚本:")
    print("  python3 monitor_gpu_idle_auto.py")
    
    return 0

if __name__ == '__main__':
    exit(main())
