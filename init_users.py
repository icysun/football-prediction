import sqlite3
import hashlib
from datetime import datetime, timedelta

def create_users_table():
    """创建用户管理相关数据库表"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    # 创建用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            membership_level INTEGER NOT NULL DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME,
            is_active INTEGER DEFAULT 1,
            last_login DATETIME
        )
    ''')
    
    # 创建会员等级表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS membership_levels (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            price REAL DEFAULT 0,
            duration_days INTEGER DEFAULT 30
        )
    ''')
    
    # 插入会员等级数据
    membership_levels = [
        (1, '初级会员', '只能查看老段推荐和一致看好', 0, 30),
        (2, '中级会员', '可查看ML和DL预测，不含AI分析', 99, 30),
        (3, '高级会员', '可查看全部预测数据包括AI分析', 199, 30)
    ]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO membership_levels (id, name, description, price, duration_days)
        VALUES (?, ?, ?, ?, ?)
    ''', membership_levels)
    
    conn.commit()
    conn.close()
    print("✅ 用户管理数据库表创建成功")

def hash_password(password):
    """密码哈希函数"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_test_users():
    """创建测试用户"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    # 创建测试用户
    test_users = [
        ('admin', '123456', 'admin@example.com', 3),  # 高级会员
        ('vip_user', '123456', 'vip@example.com', 2),   # 中级会员
        ('basic_user', '123456', 'basic@example.com', 1)  # 初级会员
    ]
    
    for username, password, email, level in test_users:
        password_hash = hash_password(password)
        expires_at = datetime.now() + timedelta(days=365)  # 一年有效期
        
        cursor.execute('''
            INSERT OR REPLACE INTO users (username, password_hash, email, membership_level, expires_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, password_hash, email, level, expires_at))
    
    conn.commit()
    conn.close()
    print("✅ 测试用户创建成功")
    print("测试账号：")
    print("  高级会员: admin / 123456")
    print("  中级会员: vip_user / 123456")
    print("  初级会员: basic_user / 123456")

if __name__ == '__main__':
    create_users_table()
    create_test_users() 