import sqlite3
from datetime import datetime

def create_betting_tables():
    """创建模拟下注相关数据库表"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    # 创建用户钱包表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wallets (
            user_id INTEGER PRIMARY KEY,
            balance REAL DEFAULT 10000.0,
            total_bet REAL DEFAULT 0.0,
            total_win REAL DEFAULT 0.0,
            total_loss REAL DEFAULT 0.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # 创建投注记录表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS betting_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            match_id TEXT NOT NULL,
            real_match_id TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            league TEXT,
            match_date TEXT,
            match_time TEXT,
            bet_type TEXT NOT NULL,
            bet_option TEXT NOT NULL,
            bet_amount REAL NOT NULL,
            odds REAL NOT NULL,
            potential_win REAL NOT NULL,
            actual_result TEXT,
            bet_result TEXT,
            actual_win REAL DEFAULT 0.0,
            bet_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            settled_time DATETIME,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # 创建下注类型配置表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bet_types (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type_code TEXT UNIQUE NOT NULL,
            type_name TEXT NOT NULL,
            description TEXT,
            min_amount REAL DEFAULT 10.0,
            max_amount REAL DEFAULT 50000.0,
            is_active INTEGER DEFAULT 1
        )
    ''')
    
    # 插入下注类型数据
    bet_types_data = [
        ('HAD', '胜平负', '预测比赛90分钟内的胜平负结果', 10.0, 50000.0, 1),
        ('HHAD', '让球胜平负', '基于让球数的胜平负预测', 10.0, 50000.0, 1),
        ('TTG', '总进球数', '预测比赛总进球数范围', 10.0, 30000.0, 1),
        ('HF', '半全场', '预测半场和全场结果组合', 20.0, 20000.0, 1)
    ]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO bet_types (type_code, type_name, description, min_amount, max_amount, is_active)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', bet_types_data)
    
    # 为现有用户初始化钱包
    cursor.execute('''
        INSERT OR IGNORE INTO wallets (user_id, balance)
        SELECT id, 10000.0 FROM users
    ''')
    
    conn.commit()
    conn.close()
    print("✅ 模拟下注数据库表创建成功")

def init_user_wallets():
    """初始化用户钱包，给所有用户1万虚拟币"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    # 获取所有用户
    users = cursor.execute("SELECT id, username FROM users").fetchall()
    
    for user_id, username in users:
        cursor.execute('''
            INSERT OR REPLACE INTO wallets (user_id, balance)
            VALUES (?, 10000.0)
        ''', (user_id,))
        print(f"✅ 为用户 {username} 初始化钱包: 10,000 虚拟币")
    
    conn.commit()
    conn.close()
    print("✅ 用户钱包初始化完成")

if __name__ == '__main__':
    create_betting_tables()
    init_user_wallets() 