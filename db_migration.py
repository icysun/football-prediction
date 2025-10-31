import sqlite3
import hashlib
from datetime import datetime, timedelta

def migrate_database():
    """迁移数据库到新的表结构"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    print("开始数据库迁移...")
    
    # 1. 创建新的user_membership表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_membership (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            level INTEGER NOT NULL DEFAULT 1,
            level_name TEXT NOT NULL DEFAULT 'Basic',
            permissions TEXT NOT NULL DEFAULT 'basic_predictions',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # 2. 创建新的wallets表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wallets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            balance REAL DEFAULT 1000.0,
            total_bet REAL DEFAULT 0.0,
            total_win REAL DEFAULT 0.0,
            total_loss REAL DEFAULT 0.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # 3. 检查是否需要迁移数据
    existing_users = cursor.execute("SELECT id, username, membership_level FROM users").fetchall()
    
    if existing_users:
        print(f"发现 {len(existing_users)} 个用户，开始迁移...")
        
        # 为每个用户创建会员信息
        level_names = {1: 'Basic', 2: 'VIP', 3: 'Premium'}
        permissions = {1: 'basic_predictions', 2: 'advanced_analysis', 3: 'full_access'}
        
        for user_id, username, membership_level in existing_users:
            # 迁移会员信息
            cursor.execute('''
                INSERT OR REPLACE INTO user_membership (user_id, level, level_name, permissions)
                VALUES (?, ?, ?, ?)
            ''', (user_id, membership_level, 
                  level_names.get(membership_level, 'Basic'),
                  permissions.get(membership_level, 'basic_predictions')))
            
            # 检查是否已有钱包记录（从user_wallets表）
            wallet_data = cursor.execute(
                "SELECT balance, total_bet, total_win, total_loss FROM user_wallets WHERE user_id = ?", 
                (user_id,)
            ).fetchone()
            
            if wallet_data:
                # 迁移钱包数据
                cursor.execute('''
                    INSERT OR REPLACE INTO wallets (user_id, balance, total_bet, total_win, total_loss)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, wallet_data[0], wallet_data[1], wallet_data[2], wallet_data[3]))
            else:
                # 创建默认钱包
                cursor.execute('''
                    INSERT OR REPLACE INTO wallets (user_id, balance)
                    VALUES (?, ?)
                ''', (user_id, 1000.0))
            
            print(f"✅ 用户 {username} (ID:{user_id}) 迁移完成")
    
    # 4. 更新users表结构，修改密码字段名
    try:
        # 检查是否有password_hash字段
        columns = cursor.execute("PRAGMA table_info(users)").fetchall()
        has_password_hash = any(col[1] == 'password_hash' for col in columns)
        has_password = any(col[1] == 'password' for col in columns)
        
        if has_password and not has_password_hash:
            # 添加新的password字段（如果不存在）
            cursor.execute("ALTER TABLE users ADD COLUMN password TEXT")
            # 复制数据
            cursor.execute("UPDATE users SET password = password_hash")
            print("✅ 密码字段迁移完成")
            
    except Exception as e:
        print(f"密码字段迁移跳过: {e}")
    
    # 5. 确保betting_records表引用正确
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS betting_records_new (
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
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # 检查是否需要迁移betting_records
    try:
        existing_bets = cursor.execute("SELECT COUNT(*) FROM betting_records").fetchone()[0]
        if existing_bets > 0:
            cursor.execute('''
                INSERT INTO betting_records_new 
                SELECT * FROM betting_records
            ''')
            cursor.execute("DROP TABLE betting_records")
            cursor.execute("ALTER TABLE betting_records_new RENAME TO betting_records")
            print(f"✅ 迁移了 {existing_bets} 条投注记录")
    except:
        # 如果旧表不存在，重命名新表
        cursor.execute("ALTER TABLE betting_records_new RENAME TO betting_records")
        print("✅ 创建了新的投注记录表")
    
    conn.commit()
    conn.close()
    print("✅ 数据库迁移完成！")

def create_admin_user():
    """创建管理员用户"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    # 检查admin用户是否存在
    admin_user = cursor.execute("SELECT id FROM users WHERE username = 'admin'").fetchone()
    
    if not admin_user:
        # 创建admin用户
        password_hash = hashlib.sha256('admin123'.encode()).hexdigest()
        cursor.execute('''
            INSERT INTO users (username, password, email, membership_level, expires_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('admin', password_hash, 'admin@example.com', 3, 
              datetime.now() + timedelta(days=3650), 1))  # 10年有效期
        
        admin_id = cursor.lastrowid
        
        # 创建admin会员信息
        cursor.execute('''
            INSERT INTO user_membership (user_id, level, level_name, permissions)
            VALUES (?, ?, ?, ?)
        ''', (admin_id, 3, 'Premium', 'full_access'))
        
        # 创建admin钱包
        cursor.execute('''
            INSERT INTO wallets (user_id, balance)
            VALUES (?, ?)
        ''', (admin_id, 100000.0))
        
        print("✅ 管理员账户创建成功: admin / admin123")
    else:
        print("✅ 管理员账户已存在")
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    migrate_database()
    create_admin_user()
    print("\n🎉 数据库迁移和初始化完成！")
    print("现在可以启动应用了：python app.py") 