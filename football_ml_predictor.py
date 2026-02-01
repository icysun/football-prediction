# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import xgboost as xgb
import optuna
import joblib
from datetime import datetime, timedelta
import warnings
import os
from base_predictor import BasePredictor

warnings.filterwarnings('ignore')

class FootballPredictor(BasePredictor):
    def __init__(self):
        super().__init__()
        self.models = {}
        self.feature_columns = []
        self.studies = {}  # 存储每个模型的study对象
        self.stats_data = {}  # 添加统计信息
        # 移除这两行，因为父类已经初始化了
        # self.result_mapping = {}  # 添加结果映射
        # self.result_mapping_reverse = {}  # 添加反向结果映射
        self.unknown_categories = {
            '主队': set(),
            '客队': set(),
            '联赛名称': set()
        }
    
    def preprocess_data(self):
        """数据预处理"""
        super().preprocess_data() # 调用父类的基本预处理
        
        # 安全地解析主客队进球数
        home_goals_list = []
        away_goals_list = []
        total_goals_list = []
        
        for score in self.df['全场比分']:
            home, away = self._get_home_away(score)
            home_goals_list.append(home)
            away_goals_list.append(away)
            total_goals_list.append(self._get_total_goals(score))
        
        self.df['home_goals'] = home_goals_list
        self.df['away_goals'] = away_goals_list
        self.df['total_goals'] = total_goals_list

        # 确保进球数为整数类型（避免后续浮点数问题）
        self.df['home_goals'] = pd.to_numeric(self.df['home_goals'], errors='coerce').astype('Int64')
        self.df['away_goals'] = pd.to_numeric(self.df['away_goals'], errors='coerce').astype('Int64')
        self.df['total_goals'] = pd.to_numeric(self.df['total_goals'], errors='coerce').astype('Int64')

        # 处理比分数据
        def get_result(score):
            try:
                if pd.isna(score) or score == '':
                    return 'draw'  # 默认返回平局而不是none
                home, away = map(int, str(score).replace("'", '').split(':'))
                if home > away:
                    return 'win'
                elif home == away:
                    return 'draw'
                else:
                    return 'lose'
            except:
                return 'draw'  # 异常情况下默认返回平局

        # 生成目标变量
        self.df['result'] = self.df['全场比分'].apply(get_result)
        
        # 将结果标签转换为数字
        result_mapping = {'draw': 0, 'lose': 1, 'win': 2}
        self.df['result_encoded'] = self.df['result'].map(result_mapping)
        self.result_mapping = result_mapping
        self.result_mapping_reverse = {v: k for k, v in result_mapping.items()}
        
        # 处理赔率数据
        for col in ['胜赔率', '平赔率', '负赔率']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 处理让球数据
        def extract_handicap(handicap_str):
            try:
                if pd.isna(handicap_str) or handicap_str == '':
                    return 0
                # 提取让球数，如"（-1）"提取-1
                import re
                match = re.search(r'（([+-]?\d+)）', str(handicap_str))
                if match:
                    return int(match.group(1))
                return 0
            except:
                return 0
        
        self.df['handicap'] = self.df['主队（让球）vs客队'].apply(extract_handicap)
        
        # 清理无效数据 - 移除result_encoded为NaN的行
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['result_encoded', 'home_goals', 'away_goals', 'total_goals'])
        removed_count = initial_count - len(self.df)
        
        if removed_count > 0:
            print(f"移除了 {removed_count} 条比分格式无效的记录")
        
        # 处理比分数据
        def get_result(score):
            try:
                if pd.isna(score) or score == '':
                    return 'draw'  # 默认返回平局而不是none
                home, away = map(int, str(score).replace("'", '').split(':'))
                if home > away:
                    return 'win'
                elif home == away:
                    return 'draw'
                else:
                    return 'lose'
            except:
                return 'draw'  # 异常情况下默认返回平局

        # 生成目标变量
        self.df['result'] = self.df['全场比分'].apply(get_result)
        
        # 将结果标签转换为数字
        result_mapping = {'draw': 0, 'lose': 1, 'win': 2}
        self.df['result_encoded'] = self.df['result'].map(result_mapping)
        self.result_mapping = result_mapping
        self.result_mapping_reverse = {v: k for k, v in result_mapping.items()}
        
        # 处理赔率数据
        for col in ['胜赔率', '平赔率', '负赔率']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 处理让球数据
        def extract_handicap(handicap_str):
            try:
                if pd.isna(handicap_str) or handicap_str == '':
                    return 0
                # 提取让球数，如"（-1）"提取-1
                import re
                match = re.search(r'（([+-]?\d+)）', str(handicap_str))
                if match:
                    return int(match.group(1))
                return 0
            except:
                return 0
        
        self.df['handicap'] = self.df['主队（让球）vs客队'].apply(extract_handicap)
        
    def _get_home_away(self, score):
        """从比分字符串中提取主客队进球数"""
        try:
            if pd.isna(score) or score == '' or score == 'nan':
                return np.nan, np.nan
            
            # 清理比分字符串
            score_str = str(score).strip().replace("'", "").replace('"', '')
            
            # 检查是否包含冒号
            if ':' not in score_str:
                return np.nan, np.nan
            
            # 分割比分
            parts = score_str.split(':')
            if len(parts) != 2:
                return np.nan, np.nan
            
            # 尝试转换为整数
            try:
                home = int(parts[0].strip())
                away = int(parts[1].strip())
            except ValueError:
                return np.nan, np.nan
            
            # 检查比分是否为非负数
            if home < 0 or away < 0:
                return np.nan, np.nan
            
            return home, away
                
        except Exception as e:
            print(f"解析比分时出错: {score}, 错误: {e}")
            return np.nan, np.nan

    def _get_total_goals(self, score):
        try:
            if pd.isna(score) or score == '':
                return np.nan
            home, away = map(int, str(score).replace("'", '').split(':'))
            return home + away
        except:
            return np.nan
        
    def feature_engineering(self):
        """特征工程"""
        print("正在进行特征工程...")
        
        # 添加数据框空值检查
        if self.df is None or len(self.df) == 0:
            print("错误：数据框为空，无法进行特征工程！")
            print("可能的原因：")
            print("1. 数据文件不存在或为空")
            print("2. 所有数据的比分格式都无效")
            print("3. 数据预处理过程中过滤掉了所有数据")
            return
        
        # 添加调试信息
        original_size = len(self.df)
        print(f"特征工程前数据大小: {original_size}")
        
        # 编码类别特征
        for col in ['主队', '客队', '联赛名称']:
            le = LabelEncoder()
            self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        
        # 计算近期状态特征
        '''
        n=5 表示最近5场比赛
        '''
        def calc_recent_stats(df, team_col, score_col, n=5, home_or_away=None):
            stats = []
            for idx, row in df.iterrows():
                team = row[team_col]
                date = row['比赛时间']
                
                # 筛选该队的历史比赛
                if home_or_away == 'home':
                    prev_games = df[(df[team_col] == team) & (df['比赛时间'] < date) & 
                                  (df['主队'] == team)].sort_values('比赛时间', ascending=False).head(n)
                elif home_or_away == 'away':
                    prev_games = df[(df[team_col] == team) & (df['比赛时间'] < date) & 
                                  (df['客队'] == team)].sort_values('比赛时间', ascending=False).head(n)
                else:
                    prev_games = df[(df[team_col] == team) & (df['比赛时间'] < date)].sort_values('比赛时间', ascending=False).head(n)
                
                if prev_games.empty:
                    stats.append((np.nan, np.nan, np.nan, np.nan))  # 使用NaN而不是0，便于后续清理
                    continue
                
                wins = 0
                draws = 0
                goals = 0
                goals_against = 0
                
                for _, g in prev_games.iterrows():
                    try:
                        home, away = map(int, str(g[score_col]).replace("'", '').split(':'))
                        if team_col == '主队':
                            if home_or_away == 'home' or home_or_away is None:
                                goals += home
                                goals_against += away
                                if home > away:
                                    wins += 1
                                elif home == away:
                                    draws += 1
                        else:
                            if home_or_away == 'away' or home_or_away is None:
                                goals += away
                                goals_against += home
                                if away > home:
                                    wins += 1
                                elif away == home:
                                    draws += 1
                    except:
                        continue
                
                total_games = len(prev_games)
                if total_games > 0:
                    stats.append((wins/total_games, draws/total_games, goals/total_games, goals_against/total_games))
                else:
                    stats.append((np.nan, np.nan, np.nan, np.nan))
            
            return pd.DataFrame(stats, columns=[
                f'{team_col}_{home_or_away if home_or_away else "all"}_win_rate',
                f'{team_col}_{home_or_away if home_or_away else "all"}_draw_rate',
                f'{team_col}_{home_or_away if home_or_away else "all"}_avg_goals',
                f'{team_col}_{home_or_away if home_or_away else "all"}_avg_goals_against'
            ])
        
        # 计算各种近期状态
        recent_home_home = calc_recent_stats(self.df, '主队', '全场比分', n=5, home_or_away='home')
        recent_away_away = calc_recent_stats(self.df, '客队', '全场比分', n=5, home_or_away='away')
        recent_home = calc_recent_stats(self.df, '主队', '全场比分', n=5)
        recent_away = calc_recent_stats(self.df, '客队', '全场比分', n=5)
        
        self.df = pd.concat([self.df, recent_home_home, recent_away_away, recent_home, recent_away], axis=1)
        
        # 特征交互 - 增强检查
        if '主队_home_win_rate' in self.df.columns and '客队_away_win_rate' in self.df.columns:
            self.df['home_away_win_rate_diff'] = self.df['主队_home_win_rate'] - self.df['客队_away_win_rate']
        
        if '主队_all_avg_goals' in self.df.columns and '客队_all_avg_goals_against' in self.df.columns:
            self.df['home_goals_vs_away_lost'] = self.df['主队_all_avg_goals'] - self.df['客队_all_avg_goals_against']
        
        if '主队_all_win_rate' in self.df.columns and '客队_all_win_rate' in self.df.columns:
            self.df['win_rate_diff'] = self.df['主队_all_win_rate'] - self.df['客队_all_win_rate']
        
        if '主队_all_avg_goals' in self.df.columns and '客队_all_avg_goals' in self.df.columns:
            self.df['goals_diff'] = self.df['主队_all_avg_goals'] - self.df['客队_all_avg_goals']
        
        # 赔率特征 - 修复除零错误
        # 避免除零错误，将0赔率替换为一个很小的正数
        self.df['胜赔率'] = self.df['胜赔率'].replace(0, 0.001)
        self.df['平赔率'] = self.df['平赔率'].replace(0, 0.001)
        self.df['负赔率'] = self.df['负赔率'].replace(0, 0.001)
        
        self.df['win_prob'] = 1 / self.df['胜赔率']
        self.df['draw_prob'] = 1 / self.df['平赔率']
        self.df['lose_prob'] = 1 / self.df['负赔率']
        self.df['odds_diff'] = self.df['胜赔率'] - self.df['负赔率']
        
        # 定义特征列
        self.feature_columns = [
            '主队_encoded', '客队_encoded', '联赛名称_encoded', 
            '胜赔率', '平赔率', '负赔率', 'handicap',
            '主队_home_win_rate', '客队_away_win_rate',
            '主队_home_avg_goals', '客队_away_avg_goals',
            '主队_home_avg_goals_against', '客队_away_avg_goals_against',
            '主队_all_win_rate', '主队_all_avg_goals', '主队_all_avg_goals_against',
            '客队_all_win_rate', '客队_all_avg_goals', '客队_all_avg_goals_against',
            'home_away_win_rate_diff', 'home_goals_vs_away_lost',
            'win_rate_diff', 'goals_diff',
            'win_prob', 'draw_prob', 'lose_prob', 'odds_diff'
        ]
        
        # 清理数据 - 更宽松的清理策略
        # 首先确保所有特征列都存在
        existing_features = [col for col in self.feature_columns if col in self.df.columns]
        self.feature_columns = existing_features
        
        # 检查各列的NaN值数量
        nan_counts = self.df[self.feature_columns + ['result_encoded', 'home_goals', 'away_goals', 'total_goals']].isna().sum()
        print("各列NaN值数量:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"  {col}: {count}")
        
        # 分步清理数据，避免一次性移除所有数据
        # 1. 首先清理目标变量（这些必须有值）
        self.df = self.df.dropna(subset=['result_encoded', 'home_goals', 'away_goals', 'total_goals'])
        
        # 2. 然后清理特征列，但允许部分特征缺失
        if len(self.feature_columns) > 0:
            # 计算每行的NaN值数量
            nan_per_row = self.df[self.feature_columns].isna().sum(axis=1)
            # 只移除NaN值过多的行（比如超过50%的特征缺失）
            max_nan_features = len(self.feature_columns) // 2
            self.df = self.df[nan_per_row <= max_nan_features]
            
            # 对于剩余的NaN值，用列的中位数填充数值特征，用众数填充分类特征
            for col in self.feature_columns:
                if self.df[col].isna().any():
                    if self.df[col].dtype in ['int64', 'float64']:
                        self.df[col] = self.df[col].fillna(self.df[col].median())
                    else:
                        self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 0)
        
        # 添加调试信息
        cleaned_size = len(self.df)
        print(f"特征工程后数据大小: {cleaned_size}")
        
        # 修复除零错误
        if original_size > 0:
            print(f"保留数据比例: {cleaned_size/original_size*100:.1f}%")
        else:
            print("原始数据为空，无法计算保留比例")
        
        if cleaned_size == 0:
            print("警告：特征工程后没有剩余数据！请检查数据质量和特征工程逻辑。")
            # 保留一些数据用于调试
            if original_size > 0:
                self.df = self.df.head(min(100, original_size))
                print(f"为调试目的保留 {len(self.df)} 行数据")
            else:
                print("无法保留数据，因为原始数据为空")
                return
        
        print(f"特征工程完成，最终特征数: {len(self.feature_columns)}")
        
    def train_models(self, n_trials=50):
        """训练模型"""
        print("开始训练模型...")
        
        # 添加数据框空值检查
        if self.df is None or len(self.df) == 0:
            print("错误：数据框为空，无法训练模型！")
            print("可能的原因：")
            print("1. 数据文件不存在或为空")
            print("2. 所有数据的比分格式都无效")
            print("3. 数据预处理过程中过滤掉了所有数据")
            print("4. 特征工程过程中过滤掉了所有数据")
            return
        
        # 准备数据
        X = self.df[self.feature_columns].copy()
        y_cls = self.df['result_encoded'].copy()
        y_home = self.df['home_goals'].copy()
        y_away = self.df['away_goals'].copy()
        y_total = self.df['total_goals'].copy()
        
        # 准备比分标签
        def create_score_label(home, away):
            # 转换为整数，避免浮点数（如 4.0）导致标签不一致
            home = int(min(6, home))  # 将6及以上的进球数归为6+
            away = int(min(6, away))  # 将6及以上的进球数归为6+
            return f"{home if home < 6 else '6+'}:{away if away < 6 else '6+'}"
        
        y_score = self.df.apply(lambda x: create_score_label(x['home_goals'], x['away_goals']), axis=1)
        
        # 确保没有缺失值
        mask = ~(X.isna().any(axis=1) | y_cls.isna() | y_home.isna() | y_away.isna() | y_total.isna())
        X = X[mask]
        y_cls = y_cls[mask]
        y_home = y_home[mask]
        y_away = y_away[mask]
        y_total = y_total[mask]
        y_score = y_score[mask]
        
        # 额外检查：确保所有目标变量都没有nan值
        assert not y_cls.isna().any(), "y_cls contains nan values"
        assert not y_home.isna().any(), "y_home contains nan values"
        assert not y_away.isna().any(), "y_away contains nan values"
        assert not y_total.isna().any(), "y_total contains nan values"
        
        # 同时过滤self.df以保持时间信息
        filtered_df = self.df[mask].copy()
        
        print(f"训练数据大小: {len(X)}")
        
        # 添加保护性检查
        if len(X) == 0:
            print("错误：没有可用的训练数据！")
            print("可能的原因：")
            print("1. 原始数据中缺少必要的列")
            print("2. 特征工程过程中产生了过多的NaN值")
            print("3. 数据清理过于严格")
            return
        
        # 按时间排序进行数据切分（用过去预测未来）
        print("使用按时间排序的数据切分方式（前80%训练，后20%测试）...")
        filtered_df = filtered_df.sort_values('比赛时间').reset_index(drop=True)
        X = X.reindex(filtered_df.index).reset_index(drop=True)
        y_cls = y_cls.reindex(filtered_df.index).reset_index(drop=True)
        y_home = y_home.reindex(filtered_df.index).reset_index(drop=True)
        y_away = y_away.reindex(filtered_df.index).reset_index(drop=True)
        y_total = y_total.reindex(filtered_df.index).reset_index(drop=True)
        y_score = y_score.reindex(filtered_df.index).reset_index(drop=True)
        
        # 计算切分点（前80%用于训练，后20%用于测试）
        split_idx = int(len(X) * 0.8)
        
        # 添加保护性检查
        if split_idx == 0:
            print("警告：训练集为空，使用所有数据进行训练")
            split_idx = len(X)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_cls_train = y_cls.iloc[:split_idx]
        y_cls_test = y_cls.iloc[split_idx:]
        y_home_train = y_home.iloc[:split_idx]
        y_home_test = y_home.iloc[split_idx:]
        y_away_train = y_away.iloc[:split_idx]
        y_away_test = y_away.iloc[split_idx:]
        y_total_train = y_total.iloc[:split_idx]
        y_total_test = y_total.iloc[split_idx:]
        y_score_train = y_score.iloc[:split_idx]
        y_score_test = y_score.iloc[split_idx:]
        
        if len(X) > 0:
            print(f"训练集大小: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
            print(f"测试集大小: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        else:
            print(f"训练集大小: {len(X_train)} (0%)")
            print(f"测试集大小: {len(X_test)} (0%)")
        print(f"训练集时间范围: {filtered_df.iloc[0]['比赛时间']} 到 {filtered_df.iloc[split_idx-1]['比赛时间']}")
        print(f"测试集时间范围: {filtered_df.iloc[split_idx]['比赛时间']} 到 {filtered_df.iloc[-1]['比赛时间']}")
        
        # 训练胜平负分类模型
        print("\n训练胜平负分类模型...")
        self.models['classification'] = self._train_classification_model(
            X_train, y_cls_train, X_test, y_cls_test, n_trials
        )
        
        # 训练主队进球数预测模型
        print("\n训练主队进球数预测模型...")
        self.models['home_goals'] = self._train_regression_model(
            X_train, y_home_train, X_test, y_home_test, n_trials, 'home_goals'
        )
        
        # 训练客队进球数预测模型
        print("\n训练客队进球数预测模型...")
        self.models['away_goals'] = self._train_regression_model(
            X_train, y_away_train, X_test, y_away_test, n_trials, 'away_goals'
        )
        
        # 训练总进球数预测模型
        print("\n训练总进球数预测模型...")
        self.models['total_goals'] = self._train_regression_model(
            X_train, y_total_train, X_test, y_total_test, n_trials, 'total_goals'
        )
        
        # 训练比分预测模型
        print("\n训练比分预测模型...")
        self.models['score'] = self._train_score_model(
            X_train, y_score_train, X_test, y_score_test, n_trials
        )
        
        print("\n所有模型训练完成")
        
    def _train_classification_model(self, X_train, y_train, X_test, y_test, n_trials):
        """训练分类模型"""
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'objective': 'multi:softmax',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'use_label_encoder': False,
                'random_state': 42,
                'verbosity': 0
            }
            
            try:
                model = xgb.XGBClassifier(**params)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = []
                
                for train_idx, valid_idx in cv.split(X_train, y_train):
                    X_fold_train = X_train.iloc[train_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    X_fold_valid = X_train.iloc[valid_idx]
                    y_fold_valid = y_train.iloc[valid_idx]
                    
                    # 确保数据对齐
                    X_fold_train = X_fold_train.reset_index(drop=True)
                    y_fold_train = y_fold_train.reset_index(drop=True)
                    X_fold_valid = X_fold_valid.reset_index(drop=True)
                    y_fold_valid = y_fold_valid.reset_index(drop=True)
                    
                    # 检查并移除NaN值
                    valid_mask = ~(y_fold_valid.isna() | X_fold_valid.isna().any(axis=1))
                    if not valid_mask.all():
                        print(f"警告：发现NaN值，移除 {sum(~valid_mask)} 个样本")
                        X_fold_valid = X_fold_valid[valid_mask]
                        y_fold_valid = y_fold_valid[valid_mask]
                    
                    model.fit(X_fold_train, y_fold_train)
                    preds = model.predict(X_fold_valid)
                    scores.append(accuracy_score(y_fold_valid, preds))
                
                return np.mean(scores)
            except Exception as e:
                print(f"训练过程中出现错误: {str(e)}")
                return 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        self.studies['classification'] = study  # 保存study对象
        
        # 使用最佳参数训练最终模型
        best_params = study.best_params
        best_params.update({
            'objective': 'multi:softmax',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'random_state': 42,
            'verbosity': 0
        })
        
        # 确保数据对齐
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        # 在训练前检查并移除NaN值
        train_mask = ~(y_train.isna() | X_train.isna().any(axis=1))
        test_mask = ~(y_test.isna() | X_test.isna().any(axis=1))
        
        if not train_mask.all():
            print(f"警告：训练集发现NaN值，移除 {sum(~train_mask)} 个样本")
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
        
        if not test_mask.all():
            print(f"警告：测试集发现NaN值，移除 {sum(~test_mask)} 个样本")
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]
        
        # 最终检查
        assert not y_train.isna().any(), f"训练集y仍包含NaN值"
        assert not y_test.isna().any(), f"测试集y仍包含NaN值"
        assert not X_train.isna().any().any(), f"训练集X仍包含NaN值"
        assert not X_test.isna().any().any(), f"测试集X仍包含NaN值"
        
        # 检查数据是否为空
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"错误：{model_name} 训练数据为空")
            return None
        
        best_model = xgb.XGBClassifier(**best_params)
        best_model.fit(X_train, y_train)
        
        # 评估
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"胜平负分类准确率: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        return best_model
    
    def _train_regression_model(self, X_train, y_train, X_test, y_test, n_trials, model_name):
        """训练回归模型"""
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'objective': 'reg:squarederror',
                'random_state': 42,
                'verbosity': 0
            }
            
            try:
                model = xgb.XGBRegressor(**params)
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                scores = []
                
                for train_idx, valid_idx in cv.split(X_train):
                    X_fold_train = X_train.iloc[train_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    X_fold_valid = X_train.iloc[valid_idx]
                    y_fold_valid = y_train.iloc[valid_idx]
                    
                    # 确保数据对齐
                    X_fold_train = X_fold_train.reset_index(drop=True)
                    y_fold_train = y_fold_train.reset_index(drop=True)
                    X_fold_valid = X_fold_valid.reset_index(drop=True)
                    y_fold_valid = y_fold_valid.reset_index(drop=True)
                    
                    # 在交叉验证中也检查NaN值
                    fold_train_mask = ~(y_fold_train.isna() | X_fold_train.isna().any(axis=1))
                    fold_valid_mask = ~(y_fold_valid.isna() | X_fold_valid.isna().any(axis=1))
                    
                    if not fold_train_mask.all():
                        X_fold_train = X_fold_train[fold_train_mask]
                        y_fold_train = y_fold_train[fold_train_mask]
                    
                    if not fold_valid_mask.all():
                        X_fold_valid = X_fold_valid[fold_valid_mask]
                        y_fold_valid = y_fold_valid[fold_valid_mask]
                    
                    # 检查折叠数据是否为空
                    if len(X_fold_train) == 0 or len(X_fold_valid) == 0:
                        print(f"警告：交叉验证折叠数据为空，跳过此折叠")
                        continue
                    
                    model.fit(X_fold_train, y_fold_train)
                    preds = model.predict(X_fold_valid)
                    scores.append(-mean_squared_error(y_fold_valid, preds))
                
                return np.mean(scores) if scores else float('-inf')
            except Exception as e:
                print(f"训练过程中出现错误: {str(e)}")
                return float('-inf')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        self.studies[model_name] = study  # 保存study对象
        
        # 使用最佳参数训练最终模型
        best_params = study.best_params
        best_params.update({
            'objective': 'reg:squarederror',
            'random_state': 42,
            'verbosity': 0
        })
        
        # 确保数据对齐
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        # 在训练前检查并移除NaN值
        train_mask = ~(y_train.isna() | X_train.isna().any(axis=1))
        test_mask = ~(y_test.isna() | X_test.isna().any(axis=1))
        
        if not train_mask.all():
            print(f"警告：训练集发现NaN值，移除 {sum(~train_mask)} 个样本")
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
        
        if not test_mask.all():
            print(f"警告：测试集发现NaN值，移除 {sum(~test_mask)} 个样本")
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]
        
        # 最终检查
        assert not y_train.isna().any(), f"训练集y仍包含NaN值"
        assert not y_test.isna().any(), f"测试集y仍包含NaN值"
        assert not X_train.isna().any().any(), f"训练集X仍包含NaN值"
        assert not X_test.isna().any().any(), f"测试集X仍包含NaN值"
        
        # 检查数据是否为空
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"错误：{model_name} 训练数据为空")
            return None
        
        best_model = xgb.XGBRegressor(**best_params)
        best_model.fit(X_train, y_train)
        
        # 评估
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{model_name} MSE: {mse:.4f}, R2: {r2:.4f}")
        
        return best_model
    
    def _train_score_model(self, X_train, y_train, X_test, y_test, n_trials):
        """训练比分预测模型"""
        # 创建完整的7x7比分标签集（49种可能）
        def create_all_possible_scores():
            """生成所有可能的比分标签"""
            all_scores = []
            for home in range(7):  # 0,1,2,3,4,5,6+
                home_label = home if home < 6 else '6+'
                for away in range(7):  # 0,1,2,3,4,5,6+
                    away_label = away if away < 6 else '6+'
                    all_scores.append(f"{home_label}:{away_label}")
            return all_scores

        # 创建全局标签编码器（包含所有49种可能的比分）
        all_possible_labels = create_all_possible_scores()
        le = LabelEncoder()
        le.fit(all_possible_labels)

        NUM_CLASSES = 49  # 7x7 比分矩阵

        # 过滤掉不在标签编码器中的样本
        valid_train_mask = y_train.isin(le.classes_)
        valid_test_mask = y_test.isin(le.classes_)

        if not valid_train_mask.all():
            print(f"警告：过滤掉 { (~valid_train_mask).sum() } 个训练集样本（标签无效）")
        if not valid_test_mask.all():
            print(f"警告：过滤掉 { (~valid_test_mask).sum() } 个测试集样本（标签无效）")

        X_train_filtered = X_train[valid_train_mask]
        y_train_filtered = y_train[valid_train_mask]
        X_test_filtered = X_test[valid_test_mask]
        y_test_filtered = y_test[valid_test_mask]

        # 编码标签
        y_train_encoded = le.transform(y_train_filtered)
        y_test_encoded = le.transform(y_test_filtered)

        # 关键修复：确保训练集包含所有49个类别
        # 如果训练集缺少某些类别，创建虚拟样本添加到训练集
        train_classes = set(y_train_encoded)
        all_classes = set(range(NUM_CLASSES))
        missing_classes = all_classes - train_classes

        if missing_classes:
            print(f"警告：训练集缺少 {len(missing_classes)} 个类别: {sorted(missing_classes)}")

            # 策略：为缺失的类别创建虚拟样本，保持数据类型一致
            is_dataframe = isinstance(X_train_filtered, pd.DataFrame)

            if is_dataframe:
                # DataFrame: 创建一个带列名的虚拟样本行
                mean_row = pd.DataFrame([X_train_filtered.mean().values],
                                       columns=X_train_filtered.columns,
                                       index=[0])
                for missing_class in sorted(missing_classes):
                    X_train_filtered = pd.concat([X_train_filtered, mean_row],
                                                 ignore_index=True)
                    y_train_encoded = np.concatenate([y_train_encoded, [missing_class]])
                    print(f"  已为类别 {missing_class} 添加虚拟样本到训练集")
            else:
                # numpy array: 创建虚拟样本行
                mean_features = X_train_filtered.mean(axis=0).reshape(1, -1)
                for missing_class in sorted(missing_classes):
                    X_train_filtered = np.vstack([X_train_filtered, mean_features])
                    y_train_encoded = np.concatenate([y_train_encoded, [missing_class]])
                    print(f"  已为类别 {missing_class} 添加虚拟样本到训练集")

            print(f"训练集现在包含 {len(set(y_train_encoded))} 个类别")

        def objective(trial):
            params = {
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': NUM_CLASSES,
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }

            # 训练模型（确保使用包含所有类别的训练集）
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train_filtered, y_train_encoded,
                eval_set=[(X_test_filtered, y_test_encoded)],
                early_stopping_rounds=20,
                verbose=False
            )

            # 预测并计算准确率
            y_pred = model.predict(X_test_filtered)
            accuracy = accuracy_score(y_test_encoded, y_pred)

            return accuracy

        # 创建study对象
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # 使用最佳参数训练最终模型
        best_params = study.best_params
        best_params.update({
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'num_class': NUM_CLASSES,
            'random_state': 42
        })

        # 保存全局标签编码器
        self.label_encoders['score'] = le

        # 训练最终模型（使用过滤后的数据）
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(
            X_train_filtered, y_train_encoded,
            eval_set=[(X_test_filtered, y_test_encoded)],
            early_stopping_rounds=20,
            verbose=False
        )

        # 保存study对象
        self.studies['score'] = study

        return final_model
    
    def save_models(self, filepath='football_models.pkl'):
        """保存模型"""
        # 计算并保存统计信息
        stats_data = {
            'home_stats': {},
            'away_stats': {},
            'global_stats': {}
        }
        
        # 检查数据框是否为空
        if self.df is None or len(self.df) == 0:
            print("警告：数据框为空，无法计算统计信息")
            # 保存模型但不包含统计信息
            model_data = {
                'models': self.models,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                'stats_data': stats_data,  # 空的统计信息
                'result_mapping': self.result_mapping,
                'result_mapping_reverse': self.result_mapping_reverse
            }
            joblib.dump(model_data, filepath)
            print(f"模型已保存到 {filepath}（无统计信息）")
            return
        
        # 定义需要统计的列
        global_stat_columns = [
            '主队_home_win_rate', '主队_home_avg_goals', '主队_home_avg_goals_against',
            '主队_all_win_rate', '主队_all_avg_goals', '主队_all_avg_goals_against',
            '客队_away_win_rate', '客队_away_avg_goals', '客队_away_avg_goals_against',
            '客队_all_win_rate', '客队_all_avg_goals', '客队_all_avg_goals_against'
        ]
        
        # 计算全局统计信息 - 添加列存在性检查
        for col in global_stat_columns:
            if col in self.df.columns:
                try:
                    stats_data['global_stats'][col] = self.df[col].mean()
                except Exception as e:
                    print(f"警告：无法计算列 {col} 的统计信息: {e}")
                    stats_data['global_stats'][col] = 0.0  # 使用默认值
            else:
                print(f"警告：列 {col} 不存在，跳过统计计算")
                stats_data['global_stats'][col] = 0.0  # 使用默认值
        
        # 定义主队统计列
        home_stat_columns = [
            '主队_home_win_rate', '主队_home_avg_goals', '主队_home_avg_goals_against',
            '主队_all_win_rate', '主队_all_avg_goals', '主队_all_avg_goals_against'
        ]
        
        # 计算每个主队的统计信息 - 添加列存在性检查
        for team in self.df['主队'].unique():
            team_stats = self.df[self.df['主队'] == team]
            stats_data['home_stats'][team] = {}
            
            for col in home_stat_columns:
                if col in team_stats.columns:
                    try:
                        stats_data['home_stats'][team][col] = team_stats[col].mean()
                    except Exception as e:
                        print(f"警告：无法计算主队 {team} 的列 {col} 的统计信息: {e}")
                        stats_data['home_stats'][team][col] = 0.0  # 使用默认值
                else:
                    print(f"警告：主队 {team} 的列 {col} 不存在，跳过统计计算")
                    stats_data['home_stats'][team][col] = 0.0  # 使用默认值
        
        # 定义客队统计列
        away_stat_columns = [
            '客队_away_win_rate', '客队_away_avg_goals', '客队_away_avg_goals_against',
            '客队_all_win_rate', '客队_all_avg_goals', '客队_all_avg_goals_against'
        ]
        
        # 计算每个客队的统计信息 - 添加列存在性检查
        for team in self.df['客队'].unique():
            team_stats = self.df[self.df['客队'] == team]
            stats_data['away_stats'][team] = {}
            
            for col in away_stat_columns:
                if col in team_stats.columns:
                    try:
                        stats_data['away_stats'][team][col] = team_stats[col].mean()
                    except Exception as e:
                        print(f"警告：无法计算客队 {team} 的列 {col} 的统计信息: {e}")
                        stats_data['away_stats'][team][col] = 0.0  # 使用默认值
                else:
                    print(f"警告：客队 {team} 的列 {col} 不存在，跳过统计计算")
                    stats_data['away_stats'][team][col] = 0.0  # 使用默认值
        
        # 添加调试信息
        print(f"统计信息计算完成：")
        print(f"  - 全局统计项: {len(stats_data['global_stats'])}")
        print(f"  - 主队统计: {len(stats_data['home_stats'])} 队")
        print(f"  - 客队统计: {len(stats_data['away_stats'])} 队")
        
        model_data = {
            'models': self.models,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'stats_data': stats_data,  # 添加统计信息
            'result_mapping': self.result_mapping,  # 添加结果映射
            'result_mapping_reverse': self.result_mapping_reverse  # 添加反向结果映射
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到 {filepath}")

    def load_models(self, filepath='football_models.pkl'):
        """加载模型"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.stats_data = model_data['stats_data']  # 加载统计信息
        self.result_mapping = model_data['result_mapping']  # 加载结果映射
        self.result_mapping_reverse = model_data['result_mapping_reverse']  # 加载反向结果映射
        print(f"模型已从 {filepath} 加载")

    def _handle_unknown_category(self, value, category):
        """处理未知类别"""
        if value not in self.label_encoders[category].classes_:
            self.unknown_categories[category].add(value)
            # 使用最相似的已知类别
            known_categories = self.label_encoders[category].classes_
            # 如果没有已知类别，返回第一个类别
            if len(known_categories) == 0:
                return known_categories[0]
            # 使用最简单的字符串相似度
            similarities = [(c, self._string_similarity(value, c)) for c in known_categories]
            most_similar = max(similarities, key=lambda x: x[1])[0]
            return most_similar
        return value

    def _string_similarity(self, s1, s2):
        """计算两个字符串的相似度"""
        # 使用最长公共子序列长度作为相似度度量
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if s1[i] == s2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[m][n] / max(m, n)  # 归一化相似度

    def predict_match(self, home_team, away_team, league, win_odds, draw_odds, lose_odds, handicap=0):
        """预测比赛结果"""
        try:
            # 处理未知类别
            home_team = self._handle_unknown_category(home_team, '主队')
            away_team = self._handle_unknown_category(away_team, '客队')
            league = self._handle_unknown_category(league, '联赛名称')

            # 编码类别特征
            home_team_encoded = self.label_encoders['主队'].transform([home_team])[0]
            away_team_encoded = self.label_encoders['客队'].transform([away_team])[0]
            league_encoded = self.label_encoders['联赛名称'].transform([league])[0]

            # 创建特征向量
            features = {
                '主队_encoded': home_team_encoded,
                '客队_encoded': away_team_encoded,
                '联赛名称_encoded': league_encoded,
                '胜赔率': win_odds,
                '平赔率': draw_odds,
                '负赔率': lose_odds,
                'handicap': handicap
            }

            # 添加其他特征（如果有）
            for col in self.feature_columns:
                if col not in features:
                    features[col] = 0  # 对于缺失的特征，使用0填充

            # 转换为特征向量
            X = pd.DataFrame([features])
            X = X[self.feature_columns]

            # 进行预测
            result_probs = self.models['classification'].predict_proba(X)[0]
            home_goals = self.models['home_goals'].predict(X)[0]
            away_goals = self.models['away_goals'].predict(X)[0]
            total_goals = self.models['total_goals'].predict(X)[0]
            score_probs = self.models['score'].predict_proba(X)[0]

            # 获取结果
            result_idx = result_probs.argmax()
            result = self.result_mapping_reverse[result_idx]

            # 将预测值映射到合理区间
            def map_total_goals(goals):
                if goals < 1.5:
                    return "0-1"
                elif goals < 2.5:
                    return "2"
                elif goals < 3.5:
                    return "3"
                elif goals < 4.5:
                    return "4"
                elif goals < 5.5:
                    return "5"
                else:
                    return "6+"
            
            total_goals_str = map_total_goals(total_goals)

            # 强制将回归预测的进球数四舍五入为整数，并生成比分字符串
            home_goals_int = int(np.round(home_goals))
            away_goals_int = int(np.round(away_goals))
            predicted_score = f"{home_goals_int}:{away_goals_int}"

            # 获取前三最可能的比分
            top_scores = []
            top_k = min(3, len(score_probs))
            top_indices = np.argsort(score_probs)[-top_k:][::-1]
            for idx in top_indices:
                score = self.label_encoders['score'].inverse_transform([idx])[0]
                prob = score_probs[idx]
                top_scores.append((score, prob))

            return {
                'result': result,
                'probabilities': {
                    'win': result_probs[self.result_mapping['win']],
                    'draw': result_probs[self.result_mapping['draw']],
                    'lose': result_probs[self.result_mapping['lose']]
                },
                'total_goals': total_goals_str,
                'total_goals_raw': total_goals,
                'predicted_score': predicted_score,
                'top_scores': top_scores,
                'goals_prediction': {
                    'home': round(home_goals),
                    'away': round(away_goals),
                    'total': round(total_goals)
                },
                'unknown_categories': {
                    'home_team': home_team in self.unknown_categories['主队'],
                    'away_team': away_team in self.unknown_categories['客队'],
                    'league': league in self.unknown_categories['联赛名称']
                }
            }
        except Exception as e:
            raise Exception(f"预测失败: {str(e)}")