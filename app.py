# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify, make_response, session, redirect, url_for, flash, request
from datetime import datetime, timedelta
import pandas as pd
import json
import os
import logging
import sqlite3
import hashlib
from functools import wraps
import threading
import time
from football_data_crawler import (
    fetch_match_data, 
    fetch_historical_match_results,
    comprehensive_match_analysis,
    get_comprehensive_match_analysis
)
from football_ml_predictor import FootballPredictor
from football_dl_predictor import FootballDLPredictor
from logging.handlers import RotatingFileHandler

# 配置日志
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # 用于session加密

# 配置Session
app.config.update(
    SESSION_COOKIE_SECURE=False,  # 允许HTTP连接
    SESSION_COOKIE_HTTPONLY=True,  # 防止XSS攻击
    SESSION_COOKIE_SAMESITE='Lax',  # CSRF保护
    PERMANENT_SESSION_LIFETIME=3600  # session有效期1小时
)

# --- 优化后的日志配置 ---
def setup_logging():
    """配置应用的日志记录器，实现日志轮转和UTF-8编码"""
    # 如果已经有handlers了，就不要重复添加，防止日志重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 文件处理器 - 实现日志轮转
    # 每个日志文件最大1MB，最多保留5个备份
    handler = RotatingFileHandler(
        'prediction_app.log', 
        maxBytes=1*1024*1024, 
        backupCount=5,
        encoding='utf-8'  # 解决中文乱码问题
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 给logger添加处理器
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

# 数据库文件
DB_FILE = 'predictions.db'

# 初始化预测器
ml_predictor = None
dl_predictor = None

def init_predictors():
    """初始化预测模型，包含错误处理"""
    global ml_predictor, dl_predictor
    
    try:
        ml_predictor = FootballPredictor()
        if os.path.exists('football_models.pkl'):
            ml_predictor.load_models('football_models.pkl')
            logger.info("ML模型加载成功")
        else:
            logger.error("ML模型文件不存在")
            ml_predictor = None
    except Exception as e:
        logger.error("ML model loading failed: " + str(e))
        ml_predictor = None

    try:
        dl_predictor = FootballDLPredictor()
        if os.path.exists('football_dl_model.pth'):
            dl_predictor.load_model('football_dl_model.pth')
            logger.info("DL模型加载成功")
        else:
            logger.error("DL模型文件不存在")
            dl_predictor = None
    except Exception as e:
        logger.error("DL model loading failed: " + str(e))
        dl_predictor = None

# ---- 新增数据库辅助函数 ----
def get_db_connection():
    """创建并返回一个数据库连接，并设置 row_factory 以便将行作为字典处理"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # 这使得查询结果可以像字典一样访问列
    return conn

def dict_from_row(row):
    """将 sqlite3.Row 对象转换为标准的 Python 字典"""
    if row is None:
        return None
    return dict(zip(row.keys(), row))

def get_last_update_time():
    """获取最后一次成功更新的时间"""
    conn = get_db_connection()
    try:
        # 检查配置表是否存在，不存在则创建
        conn.execute('''
            CREATE TABLE IF NOT EXISTS app_config (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        last_update = conn.execute("SELECT value FROM app_config WHERE key = 'last_successful_update'").fetchone()
        conn.close()
        if last_update:
            return datetime.fromisoformat(last_update[0])
        return None
    except Exception as e:
        logger.error(f"获取最后更新时间失败: {e}")
        if conn:
            conn.close()
        return None

def set_last_update_time():
    """设置最后一次成功更新的时间"""
    conn = get_db_connection()
    try:
        now_iso = datetime.now().isoformat()
        conn.execute("INSERT OR REPLACE INTO app_config (key, value) VALUES (?, ?)", ('last_successful_update', now_iso))
        conn.commit()
    except Exception as e:
        logger.error(f"设置最后更新时间失败: {e}")
    finally:
        if conn:
            conn.close()

# --- 用户认证相关函数 ---
def hash_password(password):
    """密码哈希函数"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed_password):
    """验证密码"""
    return hash_password(password) == hashed_password

def get_user_by_username(username):
    """根据用户名获取用户信息"""
    conn = get_db_connection()
    user_row = conn.execute(
        "SELECT * FROM users WHERE username = ? AND is_active = 1", (username,)
    ).fetchone()
    conn.close()
    return dict_from_row(user_row) if user_row else None

def get_user_membership_info(user_id):
    """获取用户会员信息"""
    conn = get_db_connection()
    user_info = conn.execute('''
        SELECT u.*, um.level, um.level_name, um.permissions
        FROM users u
        LEFT JOIN user_membership um ON u.id = um.user_id
        WHERE u.id = ?
    ''', (user_id,)).fetchone()
    conn.close()
    
    if user_info:
        user_dict = dict_from_row(user_info)
        if user_dict:  # 确保user_dict不是None
            # 如果没有会员信息，使用users表中的默认值
            if not user_dict.get('level'):
                user_dict['level'] = user_dict.get('membership_level', 1)
                level_names = {1: 'Basic', 2: 'VIP', 3: 'Premium'}
                permissions = {1: 'basic_predictions', 2: 'advanced_analysis', 3: 'full_access'}
                user_dict['level_name'] = level_names.get(user_dict['level'], 'Basic')
                user_dict['permissions'] = permissions.get(user_dict['level'], 'basic_predictions')
            # 添加兼容性字段
            user_dict['membership_level'] = user_dict.get('level', 1)
            user_dict['membership_name'] = user_dict.get('level_name', 'Basic')
            descriptions = {
                1: '只能查看老段推荐和一致看好',
                2: '可查看ML和DL预测，不含AI分析', 
                3: '可查看全部预测数据包括AI分析'
            }
            user_dict['membership_description'] = descriptions.get(user_dict.get('level', 1), '基础功能')
            return user_dict
    return None

# --- 模拟下注相关函数 ---
def get_user_wallet(user_id):
    """获取用户钱包信息"""
    conn = get_db_connection()
    wallet = conn.execute(
        "SELECT * FROM wallets WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return dict_from_row(wallet) if wallet else None

def create_bet(user_id, match_id, real_match_id, home_team, away_team, league, 
               match_date, match_time, bet_type, bet_option, bet_amount, odds):
    """创建下注记录"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 检查用户余额
    wallet = get_user_wallet(user_id)
    if not wallet or wallet['balance'] < bet_amount:
        conn.close()
        return False, "余额不足"
    
    # 计算潜在收益
    potential_win = bet_amount * odds
    
    try:
        # 扣除下注金额
        cursor.execute(
            "UPDATE wallets SET balance = balance - ?, total_bet = total_bet + ? WHERE user_id = ?",
            (bet_amount, bet_amount, user_id)
        )
        
        # 创建下注记录
        cursor.execute('''
            INSERT INTO betting_records (
                user_id, match_id, real_match_id, home_team, away_team, league,
                match_date, match_time, bet_type, bet_option, bet_amount, odds, potential_win
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, match_id, real_match_id, home_team, away_team, league,
              match_date, match_time, bet_type, bet_option, bet_amount, odds, potential_win))
        
        conn.commit()
        conn.close()
        return True, "下注成功"
    except Exception as e:
        conn.rollback()
        conn.close()
        return False, "下注失败: " + str(e)

def get_user_betting_records(user_id, limit=20):
    """获取用户投注记录"""
    conn = get_db_connection()
    records = conn.execute('''
        SELECT * FROM betting_records 
        WHERE user_id = ? 
        ORDER BY bet_time DESC 
        LIMIT ?
    ''', (user_id, limit)).fetchall()
    conn.close()
    return [dict_from_row(record) for record in records]

def settle_bets():
    """结算已完成的比赛投注"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 获取所有待结算的投注
    pending_bets = cursor.execute('''
        SELECT br.*, p.actual_result 
        FROM betting_records br
        LEFT JOIN predictions p ON br.match_id = p.match_id
        WHERE br.status = 'pending' AND p.actual_result IS NOT NULL AND p.actual_result != ''
    ''').fetchall()
    
    settled_count = 0
    for bet in pending_bets:
        bet_dict = dict_from_row(bet)
        if not bet_dict:
            continue
            
        actual_result = bet_dict.get('actual_result', '')
        bet_option = bet_dict.get('bet_option', '')
        bet_amount = bet_dict.get('bet_amount', 0)
        odds = bet_dict.get('odds', 1.0)
        user_id = bet_dict.get('user_id')
        bet_id = bet_dict.get('id')
        
        if not user_id or not bet_id:
            continue
            
        # 判断下注结果
        if bet_dict.get('bet_type') == 'HAD':  # 胜平负
            if ((bet_option == '胜' and actual_result == '胜') or
                (bet_option == '平' and actual_result == '平') or
                (bet_option == '负' and actual_result == '负')):
                bet_result = 'win'
                actual_win = bet_amount * odds
            else:
                bet_result = 'lose'
                actual_win = 0.0
        else:
            # 其他下注类型暂时默认为输
            bet_result = 'lose'
            actual_win = 0.0
        
        # 更新投注记录
        cursor.execute('''
            UPDATE betting_records 
            SET actual_result = ?, bet_result = ?, actual_win = ?, 
                settled_time = CURRENT_TIMESTAMP, status = 'settled'
            WHERE id = ?
        ''', (actual_result, bet_result, actual_win, bet_id))
        
        # 更新用户钱包
        if bet_result == 'win':
            cursor.execute(
                "UPDATE wallets SET balance = balance + ?, total_win = total_win + ? WHERE user_id = ?",
                (actual_win, actual_win, user_id)
            )
        else:
            cursor.execute(
                "UPDATE wallets SET total_loss = total_loss + ? WHERE user_id = ?",
                (bet_amount, user_id)
            )
        
        settled_count += 1
    
    conn.commit()
    conn.close()
    return settled_count

def get_betting_statistics(user_id):
    """获取用户投注统计"""
    conn = get_db_connection()
    
    # 基本统计
    stats = conn.execute('''
        SELECT 
            COUNT(*) as total_bets,
            COUNT(CASE WHEN status = 'settled' THEN 1 END) as settled_bets,
            COUNT(CASE WHEN bet_result = 'win' THEN 1 END) as win_bets,
            COUNT(CASE WHEN bet_result = 'lose' THEN 1 END) as lose_bets,
            SUM(bet_amount) as total_bet_amount,
            SUM(actual_win) as total_win_amount,
            AVG(odds) as avg_odds
        FROM betting_records 
        WHERE user_id = ?
    ''', (user_id,)).fetchone()
    
    # 钱包信息
    wallet = conn.execute(
        "SELECT * FROM wallets WHERE user_id = ?", (user_id,)
    ).fetchone()
    
    conn.close()
    
    stats_dict = dict_from_row(stats) if stats else {}
    wallet_dict = dict_from_row(wallet) if wallet else {}
    
    # 计算胜率
    settled_bets = stats_dict.get('settled_bets', 0) if stats_dict else 0
    win_bets = stats_dict.get('win_bets', 0) if stats_dict else 0
    win_rate = (win_bets / settled_bets * 100) if settled_bets > 0 else 0
    
    result = {'win_rate': round(win_rate, 2)}
    if stats_dict:
        result.update(stats_dict)
    if wallet_dict:
        result.update(wallet_dict)
    
    return result

def login_required(f):
    """登录验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def membership_required(min_level=1):
    """会员等级验证装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            
            user_info = get_user_membership_info(session['user_id'])
            if not user_info or user_info['membership_level'] < min_level:
                flash('您的会员等级不足，无法访问此功能', 'error')
                return redirect(url_for('index'))
            
            # 检查会员是否过期
            if user_info['expires_at']:
                try:
                    # 尝试解析带微秒的格式
                    expires_at = datetime.strptime(user_info['expires_at'], '%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    try:
                        # 尝试解析不带微秒的格式
                        expires_at = datetime.strptime(user_info['expires_at'], '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        # 如果都失败了，假设没有过期
                        expires_at = datetime.now() + timedelta(days=365)
                
                if datetime.now() > expires_at:
                    flash('您的会员已过期，请续费', 'error')
                    return redirect(url_for('index'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def safe_float_convert(value, default=0.0, round_goals=False):
    """安全转换浮点数，可选择是否对进球数取整"""
    try:
        if isinstance(value, (float, int)):
            return float(value)  # 确保返回 Python 原生 float 类型
        result = float(value)
        if round_goals:
            return result  # 不再对进球数取整，返回原始浮点数
        return result
    except (ValueError, TypeError):
        return default

def get_goals_range(predicted_goals):
    """将预测的进球数转换为区间
    
    Args:
        predicted_goals (float): 预测的进球数
        
    Returns:
        list: [最小进球数, 最大进球数]
    """
    if predicted_goals <= 0:
        return [0, 1]
    
    lower_bound = max(0, int(predicted_goals))  # 向下取整
    upper_bound = lower_bound + 1  # 向上取整加1
    
    return [lower_bound, upper_bound]

def make_predictions(match_data):
    """生成比赛预测并直接存入数据库，不再返回列表"""
    if not isinstance(match_data, list):
        logger.error("比赛数据格式错误: " + str(type(match_data)))
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 检查并添加新列（如果不存在）
    new_columns = [
        ("real_match_id", "TEXT"),
        ("ai_prediction", "TEXT"),
        ("ai_win_rate", "REAL"),
        ("ai_reason", "TEXT"),
        ("actual_score", "TEXT"),
        ("actual_total_goals", "INTEGER"),
        ("actual_home_goals", "INTEGER"),
        ("actual_away_goals", "INTEGER")
    ]
    
    for column_name, column_type in new_columns:
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN " + column_name + " " + column_type)
            conn.commit()
            logger.info("已添加" + column_name + "列到数据库")
        except sqlite3.OperationalError:
            # 列已存在，忽略错误
            pass
    
    saved_count = 0

    for match in match_data:
        try:
            # 检查必要字段
            required_fields = ['比赛编号', '比赛时间', '联赛名称', '主队', '客队', '胜赔率', '平赔率', '负赔率']
            if not all(field in match for field in required_fields):
                logger.warning("比赛数据缺少必要字段: " + str(match.get('比赛编号', 'Unknown')))
                continue

            match_id = match['比赛编号']  # 显示用的编号
            real_match_id = match.get('matchId', match.get('realMatchId', match_id))  # API用的真正ID
            match_datetime = datetime.strptime(match['比赛时间'], '%Y-%m-%d %H:%M')
            
            # ML 模型预测
            ml_pred_data = {'result': 'model_not_available', 'win_prob': 0, 'draw_prob': 0, 'lose_prob': 0, 'predicted_score': '', 'total_goals_range': '0-0'}
            if ml_predictor:
                try:
                    ml_pred = ml_predictor.predict_match(
                        home_team=match['主队'], away_team=match['客队'], league=match['联赛名称'],
                        win_odds=safe_float_convert(match['胜赔率']), draw_odds=safe_float_convert(match['平赔率']), lose_odds=safe_float_convert(match['负赔率'])
                    )
                    goals_range = get_goals_range(safe_float_convert(ml_pred.get('total_goals', 0)))
                    probabilities = ml_pred.get('probabilities', {})
                    ml_pred_data = {
                        'result': ml_pred.get('result', 'error'),
                        'win_prob': safe_float_convert(probabilities.get('win', 0)),
                        'draw_prob': safe_float_convert(probabilities.get('draw', 0)),
                        'lose_prob': safe_float_convert(probabilities.get('lose', 0)),
                        'predicted_score': ml_pred.get('predicted_score', ''),
                        'total_goals_range': str(goals_range[0]) + "-" + str(goals_range[1])
                    }
                except Exception as e:
                    logger.error("ML预测失败 - 比赛: " + str(match_id) + " - 错误: " + str(e))
                    ml_pred_data['result'] = 'error'

            # DL 模型预测
            dl_pred_data = {'result': 'model_not_available', 'win_prob': 0, 'draw_prob': 0, 'lose_prob': 0, 'predicted_score': '', 'total_goals_range': '0-0'}
            if dl_predictor:
                try:
                    dl_pred = dl_predictor.predict_match(
                        home_team=match['主队'], away_team=match['客队'], league=match['联赛名称'],
                        win_odds=safe_float_convert(match['胜赔率']), draw_odds=safe_float_convert(match['平赔率']), lose_odds=safe_float_convert(match['负赔率'])
                    )
                    goals_range = get_goals_range(safe_float_convert(dl_pred.get('total_goals', 0)))
                    probabilities = dl_pred.get('probabilities', {})
                    dl_pred_data = {
                        'result': dl_pred.get('result', 'error'),
                        'win_prob': safe_float_convert(probabilities.get('win', 0)),
                        'draw_prob': safe_float_convert(probabilities.get('draw', 0)),
                        'lose_prob': safe_float_convert(probabilities.get('lose', 0)),
                        'predicted_score': dl_pred.get('predicted_score', ''),
                        'total_goals_range': str(goals_range[0]) + "-" + str(goals_range[1])
                    }
                except Exception as e:
                    logger.error("DL预测失败 - 比赛: " + str(match_id) + " - 错误: " + str(e))
                    dl_pred_data['result'] = 'error'

            # 使用 INSERT OR REPLACE 插入或更新数据库记录
            sql = """
            INSERT OR REPLACE INTO predictions (
                match_id, real_match_id, match_date, match_time, league, home_team, away_team,
                ml_prediction, ml_win_prob, ml_draw_prob, ml_lose_prob, ml_predicted_score, ml_total_goals_range,
                dl_prediction, dl_win_prob, dl_draw_prob, dl_lose_prob, dl_predicted_score, dl_total_goals_range,
                ai_prediction, ai_win_rate, ai_reason,
                win_odds, draw_odds, lose_odds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            cursor.execute(sql, (
                match_id, real_match_id, match_datetime.strftime('%Y-%m-%d'), match_datetime.strftime('%H:%M'), match['联赛名称'], match['主队'], match['客队'],
                ml_pred_data['result'], ml_pred_data['win_prob'], ml_pred_data['draw_prob'], ml_pred_data['lose_prob'], ml_pred_data['predicted_score'], ml_pred_data['total_goals_range'],
                dl_pred_data['result'], dl_pred_data['win_prob'], dl_pred_data['draw_prob'], dl_pred_data['lose_prob'], dl_pred_data['predicted_score'], dl_pred_data['total_goals_range'],
                None, None, None,  # AI预测字段初始为空，将在AI分析时更新
                safe_float_convert(match['胜赔率']), safe_float_convert(match['平赔率']), safe_float_convert(match['负赔率'])  # 添加赔率数据
            ))
            saved_count += 1

        except Exception as e:
            logger.error("处理比赛数据时发生未知错误: " + str(match.get('比赛编号', 'Unknown')) + " - " + str(e))
            
    conn.commit()
    conn.close()
    logger.info("成功保存或更新 " + str(saved_count) + " 条预测记录到数据库。")

# --- 增强的准确率计算函数 ---
def calculate_enhanced_accuracy_stats(historical_matches):
    """计算增强的预测准确率统计，包括多维度验证"""
    if not historical_matches:
        return {
            'ml_accuracy': 0, 'dl_accuracy': 0, 'ai_accuracy': 0,
            'ml_score_accuracy': 0, 'dl_score_accuracy': 0,
            'ml_goals_accuracy': 0, 'dl_goals_accuracy': 0,
            'total_matches': 0
        }

    # 初始化计数器
    ml_correct = 0
    dl_correct = 0
    ai_correct = 0
    ml_score_correct = 0
    dl_score_correct = 0
    ml_goals_correct = 0
    dl_goals_correct = 0
    total_matches = len(historical_matches)

    for match in historical_matches:
        actual_result = match.get('actual_result')
        actual_score = match.get('actual_score')
        actual_total_goals = match.get('actual_total_goals')
        
        if not actual_result:
            continue

        # 胜平负预测准确率 - 直接比较英文预测和中文实际结果
        ml_pred = match.get('ml_prediction')
        dl_pred = match.get('dl_prediction')
        ai_pred = match.get('ai_prediction')
        
        # ML预测准确率
        if ((ml_pred == 'win' and actual_result == '胜') or
            (ml_pred == 'draw' and actual_result == '平') or
            (ml_pred == 'lose' and actual_result == '负')):
            ml_correct += 1
            
        # DL预测准确率
        if ((dl_pred == 'win' and actual_result == '胜') or
            (dl_pred == 'draw' and actual_result == '平') or
            (dl_pred == 'lose' and actual_result == '负')):
            dl_correct += 1
            
        # AI预测准确率 - AI预测也是中文格式
        if ai_pred == actual_result:
            ai_correct += 1
        
        # 比分预测准确率
        if actual_score:
            ml_predicted_score = match.get('ml_predicted_score')
            dl_predicted_score = match.get('dl_predicted_score')
            
            if ml_predicted_score and ml_predicted_score == actual_score:
                ml_score_correct += 1
            if dl_predicted_score and dl_predicted_score == actual_score:
                dl_score_correct += 1
        
        # 总进球数预测准确率
        if actual_total_goals is not None:
            ml_goals_range = match.get('ml_total_goals_range')
            dl_goals_range = match.get('dl_total_goals_range')
            
            if ml_goals_range and _is_goals_in_range(actual_total_goals, ml_goals_range):
                ml_goals_correct += 1
            if dl_goals_range and _is_goals_in_range(actual_total_goals, dl_goals_range):
                dl_goals_correct += 1
            
    # 计算准确率百分比
    ml_accuracy = (ml_correct / total_matches) * 100 if total_matches > 0 else 0
    dl_accuracy = (dl_correct / total_matches) * 100 if total_matches > 0 else 0
    ai_accuracy = (ai_correct / total_matches) * 100 if total_matches > 0 else 0
    ml_score_accuracy = (ml_score_correct / total_matches) * 100 if total_matches > 0 else 0
    dl_score_accuracy = (dl_score_correct / total_matches) * 100 if total_matches > 0 else 0
    ml_goals_accuracy = (ml_goals_correct / total_matches) * 100 if total_matches > 0 else 0
    dl_goals_accuracy = (dl_goals_correct / total_matches) * 100 if total_matches > 0 else 0

    return {
        'ml_accuracy': round(ml_accuracy, 2),
        'dl_accuracy': round(dl_accuracy, 2),
        'ai_accuracy': round(ai_accuracy, 2),
        'ml_score_accuracy': round(ml_score_accuracy, 2),
        'dl_score_accuracy': round(dl_score_accuracy, 2),
        'ml_goals_accuracy': round(ml_goals_accuracy, 2),
        'dl_goals_accuracy': round(dl_goals_accuracy, 2),
        'total_matches': total_matches,
        'detailed_stats': {
            'ml_correct': ml_correct,
            'dl_correct': dl_correct,
            'ai_correct': ai_correct,
            'ml_score_correct': ml_score_correct,
            'dl_score_correct': dl_score_correct,
            'ml_goals_correct': ml_goals_correct,
            'dl_goals_correct': dl_goals_correct
        }
    }

def _is_goals_in_range(actual_goals, goals_range):
    """检查实际进球数是否在预测范围内"""
    try:
        if '-' in goals_range:
            min_goals, max_goals = map(int, goals_range.split('-'))
            return min_goals <= actual_goals <= max_goals
        elif goals_range.endswith('+'):
            min_goals = int(goals_range[:-1])
            return actual_goals >= min_goals
        else:
            return actual_goals == int(goals_range)
    except (ValueError, TypeError):
        return False

# --- 保留原有的简单准确率计算函数以保持兼容性 ---
def calculate_accuracy_stats(historical_matches):
    """从传入的历史比赛数据中计算并返回 ML 和 DL 模型的预测准确率"""
    enhanced_stats = calculate_enhanced_accuracy_stats(historical_matches)
    return {
        'ml_accuracy': enhanced_stats['ml_accuracy'],
        'dl_accuracy': enhanced_stats['dl_accuracy'],
        'total_matches': enhanced_stats['total_matches']
    }

def calculate_yesterday_accuracy_stats():
    """计算前一天在售比赛的准确率"""
    try:
        conn = get_db_connection()
        
        # 获取所有有实际结果的比赛，按日期倒序排列
        all_finished_matches = conn.execute("""
            SELECT * FROM predictions 
            WHERE actual_result IS NOT NULL AND actual_result != ''
            ORDER BY match_date DESC, match_time DESC
        """).fetchall()
        
        if not all_finished_matches:
            conn.close()
            return {
                'ml_accuracy': 0, 'dl_accuracy': 0, 'ai_accuracy': 0,
                'ml_score_accuracy': 0, 'dl_score_accuracy': 0,
                'ml_goals_accuracy': 0, 'dl_goals_accuracy': 0,
                'total_matches': 0
            }
        
        # 按比赛编号前缀分组（如周天、周六等）
        match_groups = {}
        for match in all_finished_matches:
            match_dict = dict_from_row(match)
            match_id = match_dict.get('match_id', '')
            
            # 提取比赛编号前缀（如周天、周六、周日等）
            prefix = ''
            for i in range(len(match_id)):
                if match_id[i].isdigit():
                    prefix = match_id[:i]
                    break
            
            if prefix and prefix not in match_groups:
                match_groups[prefix] = []
            if prefix:
                match_groups[prefix].append(match_dict)
        
        # 找出最新的一组比赛（前一天在售的比赛）
        latest_group = []
        latest_date = None
        
        for prefix, matches in match_groups.items():
            if matches:
                group_date = matches[0].get('match_date')
                if group_date and (latest_date is None or group_date > latest_date):
                    latest_date = group_date
                    latest_group = matches
        
        conn.close()
        
        # 计算准确率
        if latest_group:
            return calculate_enhanced_accuracy_stats(latest_group)
        else:
            return {
                'ml_accuracy': 0, 'dl_accuracy': 0, 'ai_accuracy': 0,
                'ml_score_accuracy': 0, 'dl_score_accuracy': 0,
                'ml_goals_accuracy': 0, 'dl_goals_accuracy': 0,
                'total_matches': 0
            }
    except Exception as e:
        logger.error(f"计算前一天准确率失败: {e}")
        return {
            'ml_accuracy': 0, 'dl_accuracy': 0, 'ai_accuracy': 0,
            'ml_score_accuracy': 0, 'dl_score_accuracy': 0,
            'ml_goals_accuracy': 0, 'dl_goals_accuracy': 0,
            'total_matches': 0
        }

def is_match_finished(match_date, match_time):
    """检查比赛是否已经结束"""
    try:
        # 组合比赛日期和时间
        match_datetime_str = match_date + ' ' + match_time
        match_datetime = datetime.strptime(match_datetime_str, '%Y-%m-%d %H:%M')
        
        # 比赛开始后2小时认为比赛结束
        match_end_time = match_datetime + timedelta(hours=2)
        current_time = datetime.now()
        
        return current_time > match_end_time
    except (ValueError, TypeError):
        # 如果日期时间解析失败，默认认为比赛未结束
        return False

def filter_active_matches(matches):
    """过滤掉已结束的比赛"""
    active_matches = []
    for match in matches:
        if not is_match_finished(match.get('match_date', ''), match.get('match_time', '')):
            active_matches.append(match)
    return active_matches

# 全局变量控制自动结算线程
auto_settlement_thread = None
auto_settlement_running = False

def auto_settle_background():
    """后台自动结算线程 - 增强版"""
    global auto_settlement_running
    auto_settlement_running = True
    
    logger.info("增强版自动结算后台线程已启动")
    
    while auto_settlement_running:
        try:
            # 1. 首先检查是否有需要更新结果的比赛
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 查找已结束但未更新结果的比赛
            finished_matches = cursor.execute('''
                SELECT DISTINCT match_date 
                FROM predictions 
                WHERE actual_result IS NULL OR actual_result = ''
                AND datetime(match_date || ' ' || match_time) < datetime('now', '-2 hours')
                ORDER BY match_date DESC
                LIMIT 3
            ''').fetchall()
            
            conn.close()
            
            # 2. 如果有需要更新的比赛，尝试更新结果
            if finished_matches:
                logger.info("发现 " + str(len(finished_matches)) + " 个日期需要更新比赛结果")
                for match_row in finished_matches:
                    date_str = match_row[0]
                    try:
                        results = fetch_historical_match_results(date_str)
                        if results and not isinstance(results, str):
                            conn = get_db_connection()
                            cursor = conn.cursor()
                            updated_count = 0
                            
                            for result in results:
                                match_id = result.get('比赛编号')
                                final_result = result.get('actual_result')
                                actual_score = result.get('全场比分', '')
                                
                                # 解析比分详情
                                actual_home_goals = None
                                actual_away_goals = None
                                actual_total_goals = None
                                
                                if actual_score and ':' in actual_score:
                                    try:
                                        home_goals, away_goals = map(int, actual_score.split(':'))
                                        actual_home_goals = home_goals
                                        actual_away_goals = away_goals
                                        actual_total_goals = home_goals + away_goals
                                    except (ValueError, TypeError):
                                        pass
                                
                                if match_id and final_result:
                                    cursor.execute("""
                                        UPDATE predictions 
                                        SET actual_result = ?, actual_score = ?, 
                                            actual_home_goals = ?, actual_away_goals = ?, actual_total_goals = ?
                                        WHERE match_id = ?
                                    """, (final_result, actual_score, actual_home_goals, actual_away_goals, actual_total_goals, match_id))
                                    if cursor.rowcount > 0:
                                        updated_count += 1
                            
                            conn.commit()
                            conn.close()
                            
                            if updated_count > 0:
                                logger.info("自动更新了 " + str(date_str) + " 的 " + str(updated_count) + " 条比赛结果")
                    except Exception as e:
                        logger.error("更新 " + str(date_str) + " 比赛结果失败: " + str(e))
            
            # 3. 执行投注结算
            settled_count = settle_bets()
            if settled_count > 0:
                logger.info("自动结算完成，处理了 " + str(settled_count) + " 条投注记录")
            
            # 等待1小时后再次检查（3600秒）
            time.sleep(3600)
            
        except Exception as e:
            logger.error("自动结算过程中发生错误: " + str(e))
            # 发生错误后等待5分钟再重试
            time.sleep(300)

def start_auto_settlement():
    """启动自动结算线程"""
    global auto_settlement_thread
    
    if auto_settlement_thread is None or not auto_settlement_thread.is_alive():
        auto_settlement_thread = threading.Thread(target=auto_settle_background, daemon=True)
        auto_settlement_thread.start()
        logger.info("自动结算线程已启动")
    else:
        logger.info("自动结算线程已在运行中")

def stop_auto_settlement():
    """停止自动结算线程"""
    global auto_settlement_running
    auto_settlement_running = False
    logger.info("已请求停止自动结算线程")

@app.route('/login', methods=['GET', 'POST'])
def login():
    """用户登录"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('请输入用户名和密码', 'error')
            return render_template('login.html')
        
        user = get_user_by_username(username)
        if user and verify_password(password, user['password_hash']):
            # 检查会员是否过期
            if user['expires_at']:
                try:
                    # 尝试解析带微秒的格式
                    expires_at = datetime.strptime(user['expires_at'], '%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    try:
                        # 尝试解析不带微秒的格式
                        expires_at = datetime.strptime(user['expires_at'], '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        # 如果都失败了，假设没有过期
                        expires_at = datetime.now() + timedelta(days=365)
                
                if datetime.now() > expires_at:
                    flash('您的会员已过期，请联系管理员续费', 'error')
                    return render_template('login.html')
            
            # 登录成功
            session.permanent = True  # 设置为永久session
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['membership_level'] = user['membership_level']
            
            # 更新最后登录时间
            conn = get_db_connection()
            conn.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", 
                (user['id'],)
            )
            conn.commit()
            conn.close()
            
            flash('登录成功！', 'success')
            return redirect(url_for('index'))
        else:
            flash('用户名或密码错误', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """用户登出"""
    session.clear()
    flash('您已成功登出', 'info')
    return redirect(url_for('login'))

@app.route('/profile')
@login_required
def profile():
    """用户个人资料页面"""
    user_info = get_user_membership_info(session['user_id'])
    if not user_info:
        flash('用户信息获取失败', 'error')
        return redirect(url_for('login'))
    
    return render_template('profile.html', user_info=user_info)

@app.route('/betting')
@login_required
def betting_center():
    """投注中心页面"""
    user_id = session['user_id']
    
    # 获取用户钱包信息
    wallet = get_user_wallet(user_id)
    
    # 获取投注统计
    betting_stats = get_betting_statistics(user_id)
    
    # 获取最近投注记录
    recent_bets = get_user_betting_records(user_id, 10)
    
    # 获取可投注的比赛
    conn = get_db_connection()
    today_str = datetime.now().strftime('%Y-%m-%d')
    available_matches = conn.execute(
        "SELECT * FROM predictions WHERE match_date >= ? ORDER BY match_date, match_time", (today_str,)
    ).fetchall()
    conn.close()
    
    matches = [dict_from_row(match) for match in available_matches]
    matches = filter_active_matches(matches) # 过滤已结束的比赛
    
    return render_template('betting.html', 
                         wallet=wallet, 
                         betting_stats=betting_stats,
                         recent_bets=recent_bets,
                         matches=matches)

@app.route('/place_bets', methods=['POST'])
@login_required
def place_bets():
    """批量下注接口 (处理多个投注)"""
    bets = request.get_json().get('bets')
    if not bets:
        return jsonify({'status': 'error', 'message': '没有投注项'}), 400

    user_id = session['user_id']
    conn = get_db_connection()
    cursor = conn.cursor()

    # 检查总金额和余额
    total_bet_amount = sum(float(bet.get('amount', 0)) for bet in bets)
    wallet = get_user_wallet(user_id)
    if not wallet or wallet['balance'] < total_bet_amount:
        conn.close()
        return jsonify({'status': 'error', 'message': '余额不足'}), 400

    try:
        # 启动一个事务
        cursor.execute('BEGIN TRANSACTION')

        # 一次性扣除总金额
        cursor.execute(
            "UPDATE wallets SET balance = balance - ?, total_bet = total_bet + ? WHERE user_id = ?",
            (total_bet_amount, total_bet_amount, user_id)
        )

        # 循环插入每一条投注记录
        for bet in bets:
            potential_win = float(bet.get('amount', 0)) * float(bet.get('odds', 1.0))
            cursor.execute('''
                INSERT INTO betting_records (
                    user_id, match_id, real_match_id, home_team, away_team, league,
                    match_date, match_time, bet_type, bet_option, bet_amount, odds, potential_win
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, bet.get('match_id'), bet.get('real_match_id', bet.get('match_id')),
                bet.get('home_team'), bet.get('away_team'), bet.get('league'),
                bet.get('match_date'), bet.get('match_time'), bet.get('bet_type', 'HAD'),
                bet.get('bet_option'), float(bet.get('amount', 0)), float(bet.get('odds', 1.0)),
                potential_win
            ))

        # 提交事务
        conn.commit()
        return jsonify({'status': 'success', 'message': f'成功提交 {len(bets)} 笔投注！'})

    except Exception as e:
        # 如果有任何错误，回滚所有操作
        conn.rollback()
        logger.error(f"批量下注失败: {e}")
        return jsonify({'status': 'error', 'message': '投注处理失败，所有投注已取消'}), 500
    finally:
        conn.close()

@app.route('/place_bet', methods=['POST'])
@login_required
def place_bet():
    """下注接口 (单笔，兼容旧逻辑或未来使用)"""
    try:
        data = request.get_json()
        
        user_id = session['user_id']
        match_id = data.get('match_id')
        real_match_id = data.get('real_match_id', match_id)
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        league = data.get('league')
        match_date = data.get('match_date')
        match_time = data.get('match_time')
        bet_type = data.get('bet_type', 'HAD')
        bet_option = data.get('bet_option')
        bet_amount = float(data.get('bet_amount', 0))
        odds = float(data.get('odds', 1.0))
        
        # 验证下注金额
        if bet_amount < 10:
            return jsonify({'status': 'error', 'message': '最小下注金额为10元'})
        
        if bet_amount > 50000:
            return jsonify({'status': 'error', 'message': '最大下注金额为50,000元'})
        
        # 创建下注
        success, message = create_bet(
            user_id, match_id, real_match_id, home_team, away_team, league,
            match_date, match_time, bet_type, bet_option, bet_amount, odds
        )
        
        if success:
            return jsonify({'status': 'success', 'message': message})
        else:
            return jsonify({'status': 'error', 'message': message})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': '下注失败: ' + str(e)})

@app.route('/betting_records')
@login_required
def betting_records():
    """投注记录页面"""
    user_id = session['user_id']
    
    # 获取所有投注记录
    records = get_user_betting_records(user_id, 100)
    
    # 获取统计信息
    betting_stats = get_betting_statistics(user_id)
    
    return render_template('betting_records.html', 
                         records=records, 
                         betting_stats=betting_stats)

@app.route('/settle_bets', methods=['POST'])
@login_required
@membership_required(min_level=3)  # 只有高级会员可以手动结算
def manual_settle_bets():
    """手动结算投注（仅供测试）"""
    try:
        settled_count = settle_bets()
        return jsonify({
            'status': 'success', 
            'message': '成功结算 ' + str(settled_count) + ' 条投注记录'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': '结算失败: ' + str(e)})

@app.route('/auto_settlement_status')
@login_required
@membership_required(min_level=3)
def auto_settlement_status():
    """获取自动结算状态"""
    global auto_settlement_thread, auto_settlement_running
    
    status = {
        'running': auto_settlement_running,
        'thread_alive': auto_settlement_thread is not None and auto_settlement_thread.is_alive()
    }
    
    return jsonify({'status': 'success', 'auto_settlement': status})

@app.route('/start_auto_settlement', methods=['POST'])
@login_required
@membership_required(min_level=3)
def start_auto_settlement_route():
    """启动自动结算"""
    try:
        start_auto_settlement()
        return jsonify({'status': 'success', 'message': '自动结算已启动'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': '启动失败: ' + str(e)})

@app.route('/stop_auto_settlement', methods=['POST'])
@login_required
@membership_required(min_level=3)
def stop_auto_settlement_route():
    """停止自动结算"""
    try:
        stop_auto_settlement()
        return jsonify({'status': 'success', 'message': '自动结算已停止'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': '停止失败: ' + str(e)})

@app.route('/wallet')
@login_required
def wallet():
    """钱包页面"""
    user_id = session['user_id']
    wallet_info = get_user_wallet(user_id)
    betting_stats = get_betting_statistics(user_id)
    
    return render_template('wallet.html', 
                         wallet=wallet_info, 
                         stats=betting_stats)

@app.route('/update_predictions', methods=['POST'])
@login_required
@membership_required(min_level=2)
def update_predictions():
    """手动更新预测数据的路由"""
    try:
        # 首先获取最新的比赛数据
        match_data = fetch_match_data()
        
        # 检查是否获取成功
        if isinstance(match_data, str):
            # 如果是字符串，说明是错误信息
            flash(f'更新失败: {match_data}', 'error')
            return redirect(url_for('index'))
        
        # 生成预测并存入数据库
        make_predictions(match_data)
        
        # 更新成功时间
        set_last_update_time()
        
        flash('比赛预测已更新', 'success')
    except Exception as e:
        logger.error(f"更新预测时发生错误: {e}")
        flash('更新过程中发生未知错误', 'error')
        
    return redirect(url_for('index'))

def _update_predictions_logic():
    """后台自动更新预测的内部逻辑"""
    try:
        logger.info("开始后台自动更新预测...")
        match_data = fetch_match_data()
        if isinstance(match_data, str):
            logger.warning(f"后台更新失败: {match_data}")
            return
        make_predictions(match_data)
        set_last_update_time()
        logger.info("后台自动更新预测完成。")
    except Exception as e:
        logger.error(f"后台更新预测时发生错误: {e}")

@app.route('/')
@login_required
def index():
    """首页，展示今日预测"""
    
    # 自动更新逻辑
    last_update = get_last_update_time()
    # 如果超过30分钟未更新，则在后台线程中更新
    if not last_update or (datetime.now() - last_update) > timedelta(minutes=30):
        logger.info("数据陈旧，将启动后台线程更新预测...")
        update_thread = threading.Thread(target=_update_predictions_logic)
        update_thread.start()

    conn = get_db_connection()
    try:
        # 查询1: 获取今天及未来的比赛 (用于显示)
        today_str = datetime.now().strftime('%Y-%m-%d')
        available_matches_rows = conn.execute(
            "SELECT * FROM predictions WHERE match_date >= ? ORDER BY match_date, match_time", (today_str,)
        ).fetchall()
        
        # 查询2: 获取最近5条历史赛果 (用于显示)
        historical_preds_rows_for_display = conn.execute(
            "SELECT * FROM predictions WHERE actual_result IS NOT NULL AND actual_result != '' ORDER BY match_date DESC, match_time DESC LIMIT 5"
        ).fetchall()

        # 查询3: 获取全部历史赛果 (用于统计准确率)
        all_historical_preds_rows = conn.execute(
            "SELECT * FROM predictions WHERE actual_result IS NOT NULL AND actual_result != ''"
        ).fetchall()
    finally:
        conn.close()

    # --- 数据处理 ---
    # 今日比赛
    matches = [dict_from_row(match) for match in available_matches_rows]
    matches = filter_active_matches(matches) # 过滤已结束的比赛
    
    # 获取当前用户信息
    user_info = get_user_membership_info(session['user_id'])
    if not user_info:
        flash('用户信息获取失败', 'error')
        return redirect(url_for('login'))
    
    membership_level = user_info['membership_level']
    
    # 用于显示的5条历史记录
    historical_predictions_for_display = [dict_from_row(row) for row in historical_preds_rows_for_display]
    
    # --- 统计计算 ---
    # 计算推荐 (基于今日比赛)
    win_loss_recommendations = [
        p for p in matches if p and
        p.get('ml_prediction') and p.get('dl_prediction') and
        isinstance(p.get('ml_prediction'), str) and isinstance(p.get('dl_prediction'), str) and
        p.get('ml_prediction').strip().lower() == p.get('dl_prediction').strip().lower() and
        p.get('ml_prediction').strip().lower() in ['win', 'draw', 'lose']
    ]
    goals_recommendations = [
        p for p in matches if p and
        p.get('ml_total_goals_range') and p.get('dl_total_goals_range') and
        isinstance(p.get('ml_total_goals_range'), str) and isinstance(p.get('dl_total_goals_range'), str) and
        p.get('ml_total_goals_range').strip() == p.get('dl_total_goals_range').strip()
    ]
    
    # 使用全部历史数据计算准确率
    all_historical_predictions = [dict_from_row(row) for row in all_historical_preds_rows]
    accuracy_stats = calculate_enhanced_accuracy_stats(all_historical_predictions)
    
    # 计算前一天在售比赛的准确率
    yesterday_accuracy_stats = calculate_yesterday_accuracy_stats()

    # 为模板创建胜平负推荐的比赛ID集合，用于去重
    win_loss_match_ids = {p['match_id'] for p in win_loss_recommendations}

    return render_template(
        'index.html',
                           today_predictions=matches,
                           historical_predictions=historical_predictions_for_display, # 传给模板的是用于显示的5条
                           win_loss_recommendations=win_loss_recommendations,
                           goals_recommendations=goals_recommendations,
                           win_loss_match_ids=win_loss_match_ids, # 传递ID集合
        accuracy_stats=accuracy_stats,
        yesterday_accuracy_stats=yesterday_accuracy_stats,  # 添加新的统计数据
        user_info=user_info,
        membership_level=membership_level
    )

@app.route('/update_results', methods=['POST'])
def update_results():
    """从API获取历史赛果并更新到数据库，支持指定日期或自动检测"""
    try:
        # 获取请求参数
        data = request.get_json() if request.is_json else {}
        target_date = data.get('date') if data else None
        
        # 如果没有指定日期，检查数据库中需要更新的比赛
        if not target_date:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 查找已结束但未更新结果的比赛
            finished_matches = cursor.execute('''
                SELECT DISTINCT match_date 
                FROM predictions 
                WHERE actual_result IS NULL OR actual_result = ''
                AND datetime(match_date || ' ' || match_time) < datetime('now', '-2 hours')
                ORDER BY match_date DESC
                LIMIT 7
            ''').fetchall()
            
            conn.close()
            
            if not finished_matches:
                logger.info("没有需要更新的比赛")
                return jsonify({'status': 'no_updates_needed', 'message': '没有需要更新的比赛'})
            
            # 更新最近7天的比赛结果
            dates_to_update = [match[0] for match in finished_matches]
        else:
            # 使用指定日期
            dates_to_update = [target_date]

        total_updated = 0
        updated_dates = []
        
        for date_str in dates_to_update:
            logger.info("正在更新 " + str(date_str) + " 的比赛结果...")
            results = fetch_historical_match_results(date_str)
            
            if not results or isinstance(results, str):
                logger.warning("未获取到 " + str(date_str) + " 的比赛结果: " + str(results))
                continue

            conn = get_db_connection()
            cursor = conn.cursor()
            updated_count = 0

            for result in results:
                match_id = result.get('比赛编号')
                final_result = result.get('actual_result')  # 已经是 '胜', '平', '负'
                actual_score = result.get('全场比分', '')
                
                # 解析比分详情
                actual_home_goals = None
                actual_away_goals = None
                actual_total_goals = None
                
                if actual_score and ':' in actual_score:
                    try:
                        home_goals, away_goals = map(int, actual_score.split(':'))
                        actual_home_goals = home_goals
                        actual_away_goals = away_goals
                        actual_total_goals = home_goals + away_goals
                    except (ValueError, TypeError):
                        pass

                if match_id and final_result:
                    cursor.execute("""
                        UPDATE predictions 
                        SET actual_result = ?, actual_score = ?, 
                            actual_home_goals = ?, actual_away_goals = ?, actual_total_goals = ?
                        WHERE match_id = ?
                    """, (final_result, actual_score, actual_home_goals, actual_away_goals, actual_total_goals, match_id))
                    if cursor.rowcount > 0:
                        updated_count += 1
                        logger.info("更新比赛结果: " + str(match_id) + " -> " + str(final_result))
            
            conn.commit()
            conn.close()
            
            if updated_count > 0:
                total_updated += updated_count
                updated_dates.append(date_str)
                logger.info("成功更新 " + str(date_str) + " 的 " + str(updated_count) + " 条赛果")
        
        # 如果有更新，触发一次结算
        if total_updated > 0:
            try:
                settled_count = settle_bets()
                logger.info("自动结算完成，处理了 " + str(settled_count) + " 条投注记录")
                return jsonify({
                    'status': 'success', 
                    'updated_count': total_updated,
                    'updated_dates': updated_dates,
                    'settled_count': settled_count,
                    'message': '成功更新 ' + str(total_updated) + ' 条赛果，结算 ' + str(settled_count) + ' 条投注'
                })
            except Exception as e:
                logger.error("自动结算失败: " + str(e))
                return jsonify({
                    'status': 'partial_success', 
                    'updated_count': total_updated,
                    'updated_dates': updated_dates,
                    'message': '成功更新 ' + str(total_updated) + ' 条赛果，但结算失败: ' + str(e)
                })
        else:
            return jsonify({
                'status': 'no_results_found',
                'message': '未找到需要更新的比赛结果'
            })

    except Exception as e:
        logger.error("更新赛果失败: " + str(e), exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/comprehensive_analysis/<match_id>')
def get_comprehensive_analysis(match_id):
    """获取指定比赛的综合分析"""
    try:
        # 从数据库获取比赛基础信息
        conn = get_db_connection()
        match_row = conn.execute(
            "SELECT * FROM predictions WHERE match_id = ?", (match_id,)
        ).fetchone()
        conn.close()
        
        if not match_row:
            return jsonify({'status': 'error', 'message': '比赛不存在'}), 404
        
        match_info = dict_from_row(match_row)
        
        # 构建match_info格式以兼容comprehensive_match_analysis
        analysis_match_info = {
            '比赛编号': match_info['match_id'],
            '联赛名称': match_info['league'],
            '主队': match_info['home_team'],
            '客队': match_info['away_team'],
            '胜赔率': 2.0,  # 默认值，实际应该从数据库获取
            '平赔率': 3.0,
            '负赔率': 2.5,
            'matchId': match_info['match_id']
        }
        
        # 获取详细分析数据
        analysis_data = get_comprehensive_match_analysis(analysis_match_info)
        
        return jsonify({
            'status': 'success',
            'match_info': match_info,
            'analysis_data': analysis_data
        })
        
    except Exception as e:
        logger.error("获取综合分析失败: " + str(e), exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/ai_analysis/<match_id>')
@login_required
@membership_required(min_level=3)
def get_ai_analysis(match_id):
    """获取指定比赛的AI综合分析"""
    try:
        # 从数据库获取比赛基础信息
        conn = get_db_connection()
        
        # 首先尝试通过match_id查找
        match_row = conn.execute(
            "SELECT * FROM predictions WHERE match_id = ?", (match_id,)
        ).fetchone()
        
        # 如果没找到，尝试通过real_match_id查找
        if not match_row:
            match_row = conn.execute(
                "SELECT * FROM predictions WHERE real_match_id = ?", (match_id,)
            ).fetchone()
        
        conn.close()
        
        if not match_row:
            logger.warning("比赛不存在: " + str(match_id))
            return jsonify({'status': 'error', 'message': '比赛不存在'}), 404
        
        match_info = dict_from_row(match_row)
        
        # 获取真正的matchId用于API调用
        real_match_id = match_info.get('real_match_id') or match_info.get('match_id')
        
        if not real_match_id:
            logger.error("无法获取真实matchId: " + str(match_info))
            return jsonify({'status': 'error', 'message': '缺少比赛ID信息'}), 400
        
        # 构建match_info格式
        analysis_match_info = {
            '比赛编号': match_info['match_id'],
            '联赛名称': match_info['league'],
            '主队': match_info['home_team'],
            '客队': match_info['away_team'],
            '胜赔率': match_info.get('win_odds', 2.0),
            '平赔率': match_info.get('draw_odds', 3.0),
            '负赔率': match_info.get('lose_odds', 2.5),
            'matchId': real_match_id  # 使用真正的matchId进行API调用
        }
        
        logger.info("开始AI分析: 比赛编号=" + str(match_info['match_id']) + ", 真实ID=" + str(real_match_id) + ", 对阵=" + str(match_info['home_team']) + " vs " + str(match_info['away_team']))
        
        # 进行完整的综合分析
        result = comprehensive_match_analysis(analysis_match_info)
        
        # 保存AI预测结果到数据库
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE predictions 
                SET ai_prediction = ?, ai_win_rate = ?, ai_reason = ? 
                WHERE match_id = ? OR real_match_id = ?
            """, (
                result.get('prediction'),
                safe_float_convert(result.get('win_rate', '50')),
                result.get('reason'),
                match_info['match_id'],
                real_match_id
            ))
            conn.commit()
            conn.close()
            logger.info("AI预测结果已保存到数据库: " + str(match_info['match_id']))
        except Exception as e:
            logger.error("保存AI预测结果失败: " + str(e))
        
        return jsonify({
            'status': 'success',
            'ai_analysis': {
                'prediction': result.get('prediction'),
                'prediction_raw': result.get('prediction_raw', 'unknown'), # 传递 raw 字段
                'win_rate': result.get('win_rate'),
                'reason': result.get('reason')
            },
            'analysis_data': result.get('analysis_data', {}),
            'debug_info': {
                'match_id': match_info['match_id'],
                'real_match_id': real_match_id,
                'teams': str(match_info['home_team']) + " vs " + str(match_info['away_team'])
            }
        })
        
    except Exception as e:
        logger.error("AI分析失败: " + str(e), exc_info=True)
        return jsonify({
            'status': 'error', 
            'message': '分析过程出错: ' + str(e)[:100],
            'debug_info': {
                'match_id': match_id,
                'error_type': type(e).__name__
            }
        }), 500

@app.route('/export_predictions')
def export_predictions():
    """导出当天预测结果为CSV"""
    try:
        conn = get_db_connection()
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        # 获取今日所有预测
        predictions_rows = conn.execute(
            "SELECT * FROM predictions WHERE match_date >= ? ORDER BY match_date, match_time", (today_str,)
        ).fetchall()
        
        conn.close()
        
        if not predictions_rows:
            return jsonify({'status': 'error', 'message': '今日无预测数据可导出'}), 404
        
        # 转换为DataFrame
        predictions = [dict_from_row(row) for row in predictions_rows]
        df = pd.DataFrame(predictions)
        
        # 重新排列列顺序，使其更易读
        column_order = [
            'match_id', 'match_date', 'match_time', 'league', 'home_team', 'away_team',
            'ml_prediction', 'ml_win_prob', 'ml_draw_prob', 'ml_lose_prob', 
            'ml_predicted_score', 'ml_total_goals_range',
            'dl_prediction', 'dl_win_prob', 'dl_draw_prob', 'dl_lose_prob',
            'dl_predicted_score', 'dl_total_goals_range',
            'actual_result'
        ]
        
        # 确保所有列都存在
        for col in column_order:
            if col not in df.columns:
                df[col] = ''
        
        df = df[column_order]
        
        # 生成CSV内容
        import io
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8')
        csv_content = output.getvalue()
        output.close()
        
        # 创建响应
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = 'attachment; filename=predictions_' + today_str + '.csv'
        
        logger.info("成功导出 " + str(len(predictions)) + " 条预测记录")
        return response
        
    except Exception as e:
        logger.error("导出预测失败: " + str(e), exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/settlement_status')
@login_required
@membership_required(min_level=3)
def get_settlement_status():
    """获取详细的结算状态"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 待结算投注统计
        pending_stats = cursor.execute('''
            SELECT 
                COUNT(*) as total_pending,
                SUM(bet_amount) as total_amount,
                SUM(potential_win) as total_potential_win
            FROM betting_records 
            WHERE status = 'pending'
        ''').fetchone()
        
        # 可结算投注统计
        ready_stats = cursor.execute('''
            SELECT COUNT(*) as ready_count, SUM(br.bet_amount) as ready_amount
            FROM betting_records br
            LEFT JOIN predictions p ON br.match_id = p.match_id
            WHERE br.status = 'pending' 
            AND p.actual_result IS NOT NULL 
            AND p.actual_result != ''
        ''').fetchone()
        
        # 问题投注（已结束比赛但无结果）
        problem_bets = cursor.execute('''
            SELECT br.match_id, br.home_team, br.away_team, br.bet_option, 
                   br.bet_amount, p.match_date, p.match_time
            FROM betting_records br
            LEFT JOIN predictions p ON br.match_id = p.match_id
            WHERE br.status = 'pending' 
            AND (p.actual_result IS NULL OR p.actual_result = '')
            AND datetime(p.match_date || ' ' || p.match_time) < datetime('now', '-2 hours')
            ORDER BY p.match_date DESC, p.match_time DESC
            LIMIT 10
        ''').fetchall()
        
        conn.close()
        
        return jsonify({
            'status': 'success',
            'pending': dict_from_row(pending_stats) if pending_stats else {},
            'ready': dict_from_row(ready_stats) if ready_stats else {},
            'problem_bets': [dict_from_row(bet) for bet in problem_bets]
        })
        
    except Exception as e:
        logger.error("获取结算状态失败: " + str(e))
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/force_update_results', methods=['POST'])
@login_required
@membership_required(min_level=3)
def force_update_results():
    """强制更新比赛结果"""
    try:
        # 检查需要更新的比赛
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 查找已结束但未更新结果的比赛
        finished_matches = cursor.execute('''
            SELECT DISTINCT match_date 
            FROM predictions 
            WHERE actual_result IS NULL OR actual_result = ''
            AND datetime(match_date || ' ' || match_time) < datetime('now', '-2 hours')
            ORDER BY match_date DESC
            LIMIT 7
        ''').fetchall()
        
        conn.close()
        
        if not finished_matches:
            return jsonify({'status': 'no_updates_needed', 'message': '没有需要更新的比赛'})
        
        total_updated = 0
        updated_dates = []
        
        for match_row in finished_matches:
            date_str = match_row[0]
            logger.info("强制更新 " + str(date_str) + " 的比赛结果...")
            
            results = fetch_historical_match_results(date_str)
            if not results or isinstance(results, str):
                continue
            
            conn = get_db_connection()
            cursor = conn.cursor()
            updated_count = 0
            
            for result in results:
                match_id = result.get('比赛编号')
                final_result = result.get('actual_result')
                actual_score = result.get('全场比分', '')
                
                # 解析比分详情
                actual_home_goals = None
                actual_away_goals = None
                actual_total_goals = None
                
                if actual_score and ':' in actual_score:
                    try:
                        home_goals, away_goals = map(int, actual_score.split(':'))
                        actual_home_goals = home_goals
                        actual_away_goals = away_goals
                        actual_total_goals = home_goals + away_goals
                    except (ValueError, TypeError):
                        pass
                
                if match_id and final_result:
                    cursor.execute("""
                        UPDATE predictions 
                        SET actual_result = ?, actual_score = ?, 
                            actual_home_goals = ?, actual_away_goals = ?, actual_total_goals = ?
                        WHERE match_id = ?
                    """, (final_result, actual_score, actual_home_goals, actual_away_goals, actual_total_goals, match_id))
                    if cursor.rowcount > 0:
                        updated_count += 1
            
            conn.commit()
            conn.close()
            
            if updated_count > 0:
                total_updated += updated_count
                updated_dates.append(date_str)
                logger.info("强制更新了 " + str(date_str) + " 的 " + str(updated_count) + " 条赛果")
        
        # 自动触发结算
        settled_count = 0
        if total_updated > 0:
            settled_count = settle_bets()
        
        return jsonify({
            'status': 'success',
            'updated_count': total_updated,
            'updated_dates': updated_dates,
            'settled_count': settled_count,
            'message': '强制更新完成，共更新 ' + str(total_updated) + ' 条赛果，结算 ' + str(settled_count) + ' 条投注'
        })
        
    except Exception as e:
        logger.error("强制更新失败: " + str(e))
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 管理员会员管理功能
@app.route('/admin/users')
@login_required
def admin_users():
    """管理员用户管理页面"""
    if session.get('username') != 'admin':
        flash('权限不足，只有管理员可以访问此页面', 'error')
        return redirect(url_for('index'))
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 获取所有用户信息，确保数据完整性
        users = cursor.execute('''
            SELECT u.id, u.username, u.email, u.created_at,
                   COALESCE(m.level, u.membership_level, 1) as level,
                   COALESCE(m.level_name, 
                     CASE COALESCE(m.level, u.membership_level, 1)
                       WHEN 1 THEN 'Basic'
                       WHEN 2 THEN 'VIP' 
                       WHEN 3 THEN 'Premium'
                       ELSE 'Basic'
                     END) as level_name,
                   COALESCE(m.permissions,
                     CASE COALESCE(m.level, u.membership_level, 1)
                       WHEN 1 THEN 'basic_predictions'
                       WHEN 2 THEN 'advanced_analysis'
                       WHEN 3 THEN 'full_access'
                       ELSE 'basic_predictions'
                     END) as permissions,
                   COALESCE(w.balance, 0.0) as balance
            FROM users u
            LEFT JOIN user_membership m ON u.id = m.user_id
            LEFT JOIN wallets w ON u.id = w.user_id
            WHERE u.is_active = 1
            ORDER BY u.id ASC
        ''').fetchall()
        
        conn.close()
        
        # 转换为字典格式
        users_list = []
        for user in users:
            if user:  # 确保user不为None
                users_list.append({
                    'id': user[0],
                    'username': user[1] or '',
                    'email': user[2] or '未设置',
                    'created_at': user[3] or '',
                    'level': user[4] or 1,
                    'level_name': user[5] or 'Basic',
                    'permissions': user[6] or 'basic_predictions',
                    'balance': float(user[7] or 0.0)
                })
        
        return render_template('admin_users.html', users=users_list)
        
    except Exception as e:
        logger.error("获取用户列表失败: " + str(e))
        flash('获取用户列表失败', 'error')
        return redirect(url_for('index'))

@app.route('/admin/users/create', methods=['GET', 'POST'])
@login_required
def admin_create_user():
    """创建新用户"""
    if session.get('username') != 'admin':
        return jsonify({'status': 'error', 'message': '权限不足'}), 403
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            username = data.get('username', '').strip()
            password = data.get('password', '').strip()
            email = data.get('email', '').strip()
            level = int(data.get('level', 1))
            balance = float(data.get('balance', 1000.0))
            
            if not username or not password:
                return jsonify({'status': 'error', 'message': '用户名和密码不能为空'})
            
            if len(username) < 3:
                return jsonify({'status': 'error', 'message': '用户名至少需要3个字符'})
            
            if len(password) < 6:
                return jsonify({'status': 'error', 'message': '密码至少需要6个字符'})
            
            # 检查用户名是否已存在
            existing_user = get_user_by_username(username)
            if existing_user:
                return jsonify({'status': 'error', 'message': '用户名已存在'})
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 创建用户
            hashed_password = hash_password(password)
            expires_at = datetime.now() + timedelta(days=365)  # 一年有效期
            cursor.execute(
                "INSERT INTO users (username, password, password_hash, email, membership_level, expires_at, is_active) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (username, hashed_password, hashed_password, email or None, level, expires_at, 1)
            )
            user_id = cursor.lastrowid
            
            # 设置会员等级
            level_names = {1: 'Basic', 2: 'VIP', 3: 'Premium'}
            permissions = {1: 'basic_predictions', 2: 'advanced_analysis', 3: 'full_access'}
            
            cursor.execute(
                "INSERT INTO user_membership (user_id, level, level_name, permissions) VALUES (?, ?, ?, ?)",
                (user_id, level, level_names.get(level, 'Basic'), permissions.get(level, 'basic_predictions'))
            )
            
            # 创建钱包
            cursor.execute(
                "INSERT INTO wallets (user_id, balance) VALUES (?, ?)",
                (user_id, balance)
            )
            
            conn.commit()
            conn.close()
            
            logger.info("管理员创建新用户: " + str(username) + ", 等级: " + str(level) + ", 余额: " + str(balance))
            return jsonify({'status': 'success', 'message': '用户创建成功'})
            
        except Exception as e:
            logger.error("创建用户失败: " + str(e))
            return jsonify({'status': 'error', 'message': '创建用户失败: ' + str(e)})
    
    return jsonify({'status': 'error', 'message': '仅支持POST请求'})

@app.route('/admin/users/<int:user_id>/edit', methods=['POST'])
@login_required
def admin_edit_user(user_id):
    """编辑用户信息"""
    if session.get('username') != 'admin':
        return jsonify({'status': 'error', 'message': '权限不足'}), 403
    
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        level = int(data.get('level', 1))
        balance = float(data.get('balance', 0))
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 检查用户是否存在
        user = cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return jsonify({'status': 'error', 'message': '用户不存在'})
        
        # 更新用户邮箱和会员等级
        cursor.execute(
            "UPDATE users SET email = ?, membership_level = ? WHERE id = ?",
            (email or None, level, user_id)
        )
        
        # 更新会员等级
        level_names = {1: 'Basic', 2: 'VIP', 3: 'Premium'}
        permissions = {1: 'basic_predictions', 2: 'advanced_analysis', 3: 'full_access'}
        
        # 使用INSERT OR REPLACE确保记录存在
        cursor.execute(
            "INSERT OR REPLACE INTO user_membership (user_id, level, level_name, permissions) VALUES (?, ?, ?, ?)",
            (user_id, level, level_names.get(level, 'Basic'), permissions.get(level, 'basic_predictions'))
        )
        
        # 更新钱包余额，使用INSERT OR REPLACE确保记录存在
        cursor.execute(
            "INSERT OR REPLACE INTO wallets (user_id, balance, total_bet, total_win, total_loss) VALUES (?, ?, COALESCE((SELECT total_bet FROM wallets WHERE user_id = ?), 0), COALESCE((SELECT total_win FROM wallets WHERE user_id = ?), 0), COALESCE((SELECT total_loss FROM wallets WHERE user_id = ?), 0))",
            (user_id, balance, user_id, user_id, user_id)
        )
        
        conn.commit()
        conn.close()
        
        logger.info("管理员更新用户 " + str(user[0]) + ": 等级=" + str(level) + ", 余额=" + str(balance) + ", 邮箱=" + str(email))
        return jsonify({'status': 'success', 'message': '用户信息更新成功'})
        
    except Exception as e:
        logger.error("更新用户失败: " + str(e))
        return jsonify({'status': 'error', 'message': '更新用户失败: ' + str(e)})

@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    """删除用户"""
    if session.get('username') != 'admin':
        return jsonify({'status': 'error', 'message': '权限不足'}), 403
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 检查用户是否存在
        user = cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return jsonify({'status': 'error', 'message': '用户不存在'})
        
        username = user[0]
        
        # 不允许删除admin用户
        if username == 'admin':
            return jsonify({'status': 'error', 'message': '不能删除管理员账户'})
        
        # 删除相关数据（级联删除）
        cursor.execute("DELETE FROM betting_records WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM wallets WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM user_membership WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        
        conn.commit()
        conn.close()
        
        logger.info("管理员删除用户: " + str(username))
        return jsonify({'status': 'success', 'message': '用户 ' + str(username) + ' 删除成功'})
        
    except Exception as e:
        logger.error("删除用户失败: " + str(e))
        return jsonify({'status': 'error', 'message': '删除用户失败: ' + str(e)})

@app.route('/admin/users/<int:user_id>/reset_password', methods=['POST'])
@login_required
def admin_reset_password(user_id):
    """重置用户密码"""
    if session.get('username') != 'admin':
        return jsonify({'status': 'error', 'message': '权限不足'}), 403
    
    try:
        data = request.get_json()
        new_password = data.get('password', '').strip()
        
        if not new_password or len(new_password) < 6:
            return jsonify({'status': 'error', 'message': '密码至少需要6个字符'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 检查用户是否存在
        user = cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return jsonify({'status': 'error', 'message': '用户不存在'})
        
        # 更新密码，确保两个字段一致
        hashed_password = hash_password(new_password)
        cursor.execute(
            "UPDATE users SET password = ?, password_hash = ? WHERE id = ?",
            (hashed_password, hashed_password, user_id)
        )
        
        conn.commit()
        conn.close()
        
        logger.info("管理员重置用户 " + str(user[0]) + " 的密码")
        return jsonify({'status': 'success', 'message': '密码重置成功'})
        
    except Exception as e:
        logger.error("重置密码失败: " + str(e))
        return jsonify({'status': 'error', 'message': '重置密码失败: ' + str(e)})

@app.route('/test_bootstrap')
def test_bootstrap_page():
    """渲染一个裸露的Bootstrap测试页面，以诊断核心功能。"""
    return render_template('test_bootstrap.html')

# 初始化
setup_logging()
init_predictors()

if __name__ == '__main__':
    # 设置日志
    setup_logging()
    # 初始化预测器
    init_predictors()
    # 启动自动结算线程
    start_auto_settlement()
    logger.info("Flask应用启动，增强版自动结算功能已开启")
    app.run(debug=True, host='127.0.0.1',  port=80 )
