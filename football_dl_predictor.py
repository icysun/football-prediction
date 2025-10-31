# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import os
import joblib
from base_predictor import BasePredictor

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class FootballDataset(Dataset):
    """足球数据集"""
    def __init__(self, features, result_labels, goals_labels, score_labels):
        self.features = torch.FloatTensor(features)
        self.result_labels = torch.LongTensor(result_labels)
        self.goals_labels = torch.FloatTensor(goals_labels)
        self.score_labels = torch.LongTensor(score_labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.result_labels[idx], self.goals_labels[idx], self.score_labels[idx]

class ResidualBlock(nn.Module):
    def __init__(self, in_features, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.relu = nn.ReLU()

class FootballNet(nn.Module):
    """多任务学习网络 - 优化版本"""
    def __init__(self, input_size, dropout_rate=0.3):
        super(FootballNet, self).__init__()
        
        # 简化网络结构，使用更小的层
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2)
        )
        
        # 胜平负分类任务 - 简化结构
        self.result_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate/3),
            nn.Linear(32, 3)  # 3分类：胜、平、负
        )
        
        # 进球数预测任务 - 简化结构
        self.goals_regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate/3),
            nn.Linear(32, 3)  # 3个输出：主队进球、客队进球、总进球
        )

        # 添加比分预测任务
        self.score_classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/3),
            nn.Linear(64, 49)  # 7x7的比分矩阵（0-6+）
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        result_out = self.result_classifier(shared_features)
        goals_out = self.goals_regressor(shared_features)
        score_out = self.score_classifier(shared_features)
        return result_out, goals_out, score_out

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def augment_data(X, y_cls, y_goals, noise_level=0.05):
    """数据增强：添加高斯噪声"""
    X_aug = X.copy()
    # 只对数值特征添加噪声
    noise = np.random.normal(0, noise_level, X_aug.shape)
    X_aug += noise
    return X_aug, y_cls.copy(), y_goals.copy()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class FootballDLPredictor(BasePredictor):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
        # 移除这两行，让基类的result_mapping和result_mapping_reverse保持有效
        # self.result_mapping = None
        # self.result_mapping_reverse = None
        self.unknown_categories = {
            '主队': set(),
            '客队': set(),
            '联赛名称': set()
        }
    
    def preprocess_data(self):
        """数据预处理"""
        super().preprocess_data()
        
        # 针对DL模型的特定预处理
        self.df['home_goals'], self.df['away_goals'] = zip(*self.df['全场比分'].apply(self._get_home_away))
        self.df['total_goals'] = self.df['全场比分'].apply(self._get_total_goals)
    
    def _get_home_away(self, score):
            try:
                if pd.isna(score) or score == '':
                    return np.nan, np.nan
                home, away = map(int, str(score).replace("'", '').split(':'))
                return home, away
            except:
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
        
        # 编码类别特征
        for col in ['主队', '客队', '联赛名称']:
            le = LabelEncoder()
            self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        
        # 计算近期状态特征
        def calc_recent_stats(df, team_col, score_col, n=5, home_or_away=None):
            stats = []
            for idx, row in df.iterrows():
                team = row[team_col]
                date = row['比赛时间']
                
                if home_or_away == 'home':
                    prev_games = df[(df[team_col] == team) & (df['比赛时间'] < date) & 
                                  (df['主队'] == team)].sort_values('比赛时间', ascending=False).head(n)
                elif home_or_away == 'away':
                    prev_games = df[(df[team_col] == team) & (df['比赛时间'] < date) & 
                                  (df['客队'] == team)].sort_values('比赛时间', ascending=False).head(n)
                else:
                    prev_games = df[(df[team_col] == team) & (df['比赛时间'] < date)].sort_values('比赛时间', ascending=False).head(n)
                
                if prev_games.empty:
                    stats.append((0, 0, 0, 0))
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
                stats.append((wins/total_games, draws/total_games, goals/total_games, goals_against/total_games))
            
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
        
        # 特征交互
        if '主队_home_win_rate' in self.df.columns and '客队_away_win_rate' in self.df.columns:
            self.df['home_away_win_rate_diff'] = self.df['主队_home_win_rate'] - self.df['客队_away_win_rate']
        
        if '主队_all_avg_goals' in self.df.columns and '客队_all_avg_goals_against' in self.df.columns:
            self.df['home_goals_vs_away_lost'] = self.df['主队_all_avg_goals'] - self.df['客队_all_avg_goals_against']
        
        if '主队_all_win_rate' in self.df.columns and '客队_all_win_rate' in self.df.columns:
            self.df['win_rate_diff'] = self.df['主队_all_win_rate'] - self.df['客队_all_win_rate']
        
        if '主队_all_avg_goals' in self.df.columns and '客队_all_avg_goals' in self.df.columns:
            self.df['goals_diff'] = self.df['主队_all_avg_goals'] - self.df['客队_all_avg_goals']
        
        # 赔率特征
        self.df['win_prob'] = 1 / self.df['胜赔率']
        self.df['draw_prob'] = 1 / self.df['平赔率']
        self.df['lose_prob'] = 1 / self.df['负赔率']
        self.df['odds_diff'] = self.df['胜赔率'] - self.df['负赔率']
        
        # 定义特征列
        self.feature_columns = [
            '主队_encoded', '客队_encoded', '联赛名称_encoded', 
            '胜赔率', '平赔率', '负赔率', 'handicap',
            '主队_home_win_rate', '主队_home_avg_goals', '主队_home_avg_goals_against',
            '主队_all_win_rate', '主队_all_avg_goals', '主队_all_avg_goals_against',
            '客队_away_win_rate', '客队_away_avg_goals', '客队_away_avg_goals_against',
            '客队_all_win_rate', '客队_all_avg_goals', '客队_all_avg_goals_against',
            'home_away_win_rate_diff', 'home_goals_vs_away_lost',
            'win_rate_diff', 'goals_diff',
            'win_prob', 'draw_prob', 'lose_prob', 'odds_diff'
        ]
        
        # 清理数据
        self.df = self.df.dropna(subset=self.feature_columns + ['result_encoded', 'home_goals', 'away_goals', 'total_goals'])
        
        print(f"特征工程完成，最终特征数: {len(self.feature_columns)}")
    
    def train_model(self, batch_size=64, epochs=100, learning_rate=0.001, weight_decay=0.01, 
                   patience=7, augment=True):
        """训练模型"""
        print("开始训练模型...")
        
        # 准备数据
        X = self.df[self.feature_columns].values
        y_cls = self.df['result_encoded'].values
        y_goals = np.column_stack((
            self.df['home_goals'].values,
            self.df['away_goals'].values,
            self.df['total_goals'].values
        ))
        
        # 准备比分标签
        def create_score_label(home, away):
            home = min(6, home)  # 将6及以上的进球数归为6+
            away = min(6, away)  # 将6及以上的进球数归为6+
            return home * 7 + away  # 7x7的比分矩阵
        
        y_score = np.array([create_score_label(h, a) for h, a in zip(self.df['home_goals'], self.df['away_goals'])])
        
        # 按时间排序进行数据切分（用过去预测未来）
        print("使用按时间排序的数据切分方式（前80%训练，后20%测试）...")
        sorted_df = self.df.sort_values('比赛时间').reset_index(drop=True)
        
        # 重新排序所有数据
        X = sorted_df[self.feature_columns].values
        y_cls = sorted_df['result_encoded'].values
        y_goals = np.column_stack((
            sorted_df['home_goals'].values,
            sorted_df['away_goals'].values,
            sorted_df['total_goals'].values
        ))
        y_score = np.array([create_score_label(h, a) for h, a in zip(sorted_df['home_goals'], sorted_df['away_goals'])])
        
        # 计算切分点（前80%用于训练，后20%用于测试）
        split_idx = int(len(X) * 0.8)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_cls_train = y_cls[:split_idx]
        y_cls_test = y_cls[split_idx:]
        y_goals_train = y_goals[:split_idx]
        y_goals_test = y_goals[split_idx:]
        y_score_train = y_score[:split_idx]
        y_score_test = y_score[split_idx:]
        
        print(f"训练集大小: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"测试集大小: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        print(f"训练集时间范围: {sorted_df.iloc[0]['比赛时间']} 到 {sorted_df.iloc[split_idx-1]['比赛时间']}")
        print(f"测试集时间范围: {sorted_df.iloc[split_idx]['比赛时间']} 到 {sorted_df.iloc[-1]['比赛时间']}")
        
        # 数据增强（仅对训练集）
        if augment:
            X_aug, y_cls_aug, y_goals_aug = augment_data(X_train, y_cls_train, y_goals_train)
            y_score_aug = np.array([create_score_label(h, a) for h, a in zip(y_goals_aug[:, 0], y_goals_aug[:, 1])])
            X_train = np.vstack([X_train, X_aug])
            y_cls_train = np.hstack([y_cls_train, y_cls_aug])
            y_goals_train = np.vstack([y_goals_train, y_goals_aug])
            y_score_train = np.hstack([y_score_train, y_score_aug])
            print(f"数据增强后训练集大小: {len(X_train)}")
        
        # 创建数据加载器
        train_dataset = FootballDataset(X_train, y_cls_train, y_goals_train, y_score_train)
        test_dataset = FootballDataset(X_test, y_cls_test, y_goals_test, y_score_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 创建模型
        self.model = FootballNet(len(self.feature_columns)).to(self.device)
        
        # 定义损失函数和优化器
        cls_criterion = nn.CrossEntropyLoss()
        reg_criterion = nn.MSELoss()
        score_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 早停
        early_stopping = EarlyStopping(patience=patience)
        
        # 训练循环
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y_cls, batch_y_goals, batch_y_score in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y_cls = batch_y_cls.to(self.device)
                batch_y_goals = batch_y_goals.to(self.device)
                batch_y_score = batch_y_score.to(self.device)
                
                optimizer.zero_grad()
                result_out, goals_out, score_out = self.model(batch_X)
                
                # 计算损失
                cls_loss = cls_criterion(result_out, batch_y_cls)
                reg_loss = reg_criterion(goals_out, batch_y_goals)
                score_loss = score_criterion(score_out, batch_y_score)
                loss = cls_loss + reg_loss + score_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = result_out.max(1)
                train_total += batch_y_cls.size(0)
                train_correct += predicted.eq(batch_y_cls).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_accuracy = 100. * train_correct / train_total
            
            # 测试阶段
            self.model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y_cls, batch_y_goals, batch_y_score in test_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y_cls = batch_y_cls.to(self.device)
                    batch_y_goals = batch_y_goals.to(self.device)
                    batch_y_score = batch_y_score.to(self.device)
                    
                    result_out, goals_out, score_out = self.model(batch_X)
                    
                    # 计算损失
                    cls_loss = cls_criterion(result_out, batch_y_cls)
                    reg_loss = reg_criterion(goals_out, batch_y_goals)
                    score_loss = score_criterion(score_out, batch_y_score)
                    loss = cls_loss + reg_loss + score_loss
                    
                    test_loss += loss.item()
                    _, predicted = result_out.max(1)
                    test_total += batch_y_cls.size(0)
                    test_correct += predicted.eq(batch_y_cls).sum().item()
            
            test_loss = test_loss / len(test_loader)
            test_accuracy = 100. * test_correct / test_total
            
            # 记录损失和准确率
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}]')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
                print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
            
            # 早停检查
            early_stopping(test_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
        # 绘制训练过程
        self._plot_training_process(train_losses, test_losses, train_accuracies, test_accuracies)
        
        print("模型训练完成")
    
    def _plot_training_process(self, train_losses, test_losses, train_accuracies, test_accuracies):
        """记录训练过程（已移除绘图功能）"""
        # 只保留基本的训练过程记录
        print(f"训练完成 - 最终训练损失: {train_losses[-1]:.4f}, 最终测试损失: {test_losses[-1]:.4f}")
        print(f"最终训练准确率: {train_accuracies[-1]:.2f}%, 最终测试准确率: {test_accuracies[-1]:.2f}%")
    
    def save_model(self, filepath='football_dl_model.pth'):
        """保存模型"""
        # 创建保存目录
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # 保存模型状态
        model_data = {
            'model_state': self.model.state_dict(),
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'result_mapping': self.result_mapping,
            'result_mapping_reverse': self.result_mapping_reverse
        }
        torch.save(model_data, filepath)
        print(f"模型已保存到 {filepath}")
    
    def load_model(self, filepath='football_dl_model.pth'):
        """加载模型"""
        model_data = torch.load(filepath)
        
        # 恢复模型状态
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.scaler = model_data['scaler']
        self.result_mapping = model_data['result_mapping']
        self.result_mapping_reverse = model_data['result_mapping_reverse']
        
        # 创建模型并加载权重
        self.model = FootballNet(len(self.feature_columns)).to(self.device)
        self.model.load_state_dict(model_data['model_state'])
        self.model.eval()
        
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

            # 转换为张量
            X_tensor = torch.FloatTensor([list(features.values())]).to(self.device)

            # 进行预测
            self.model.eval()
            with torch.no_grad():
                result_logits, goals_pred, score_pred = self.model(X_tensor)
                result_probs = torch.softmax(result_logits, dim=1)[0]
                score_probs = torch.softmax(score_pred, dim=1)[0]
                
            # 获取结果
            result_idx = result_probs.argmax().item()
            result = self.result_mapping_reverse[result_idx]

            # 获取胜平负概率
            win_prob = result_probs[self.result_mapping['win']].item()
            draw_prob = result_probs[self.result_mapping['draw']].item()
            lose_prob = result_probs[self.result_mapping['lose']].item()

            # 获取总进球预测并进行区间映射
            total_goals_raw = goals_pred[0][2].item()
            
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
            
            total_goals = map_total_goals(total_goals_raw)

            # 获取最可能的比分
            score_idx = score_probs.argmax().item()
            home_score = score_idx // 7
            away_score = score_idx % 7
            if home_score == 6:
                predicted_score = "6+:"
            else:
                predicted_score = f"{home_score}:"
            if away_score == 6:
                predicted_score += "6+"
            else:
                predicted_score += str(away_score)

            # 获取前三最可能的比分
            top_scores = []
            top_k = min(3, len(score_probs))
            top_indices = torch.topk(score_probs, top_k).indices
            for idx in top_indices:
                idx = idx.item()
                h = idx // 7
                a = idx % 7
                prob = score_probs[idx].item()
                score_str = f"{h if h < 6 else '6+'}:{a if a < 6 else '6+'}"
                top_scores.append((score_str, prob))

            return {
                'result': result,
                'probabilities': {
                    'win': win_prob,
                    'draw': draw_prob,
                    'lose': lose_prob
                },
                'total_goals': total_goals,
                'total_goals_raw': total_goals_raw,
                'predicted_score': predicted_score,
                'top_scores': top_scores,
                'unknown_categories': {
                    'home_team': home_team in self.unknown_categories['主队'],
                    'away_team': away_team in self.unknown_categories['客队'],
                    'league': league in self.unknown_categories['联赛名称']
                }
            }
        except Exception as e:
            raise Exception(f"预测失败: {str(e)}")