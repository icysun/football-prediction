# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re

class BasePredictor:
    """
    预测器基类，包含数据加载和预处理的通用逻辑。
    """
    def __init__(self):
        self.df = None
        self.label_encoders = {}
        self.result_mapping = {'draw': 0, 'lose': 1, 'win': 2}
        self.result_mapping_reverse = {v: k for k, v in self.result_mapping.items()}

    def load_data(self, csv_file):
        """通用数据加载方法"""
        print(f"正在从 {csv_file} 加载数据...")
        self.df = pd.read_csv(csv_file)
        print(f"数据加载完成，共{len(self.df)}条记录")
        return self.df

    def preprocess_data(self):
        """通用数据预处理方法"""
        print("正在进行数据预处理...")
        
        # 添加调试信息 - 查看前几条比分数据
        print("前5条比分数据示例:")
        for i in range(min(5, len(self.df))):
            score = self.df.iloc[i]['全场比分']
            print(f"  原始比分: {score}, 类型: {type(score)}")
        
        # 处理比分数据以生成目标变量
        self.df['result'] = self.df['全场比分'].apply(self._get_result_from_score)
        
        # 添加调试信息 - 查看result_mapping
        print(f"result_mapping内容: {self.result_mapping}")
        print(f"result列的唯一值: {self.df['result'].unique()}")
        
        # 检查result列中的每个值是否在result_mapping中
        for i in range(min(5, len(self.df))):
            result_val = self.df.iloc[i]['result']
            if result_val in self.result_mapping:
                print(f"'{result_val}' -> {self.result_mapping[result_val]}")
            else:
                print(f"'{result_val}' 不在result_mapping中")
        
        self.df['result_encoded'] = self.df['result'].map(self.result_mapping)
        
        # 添加调试信息 - 查看处理结果
        print(f"处理后的result列前5个值: {self.df['result'].head().tolist()}")
        print(f"处理后的result_encoded列前5个值: {self.df['result_encoded'].head().tolist()}")
        
        # 处理赔率和让球数据
        for col in ['胜赔率', '平赔率', '负赔率']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df['handicap'] = self.df['主队（让球）vs客队'].apply(self._extract_handicap)
        
        # 清理无效数据 - 移除result_encoded为NaN的行
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['result_encoded'])
        removed_count = initial_count - len(self.df)
        
        if removed_count > 0:
            print(f"移除了 {removed_count} 条比分格式无效的记录")
        
        print("数据预处理完成")

    def _get_result_from_score(self, score):
        """从'xx:xx'格式的比分中提取比赛结果"""
        try:
            if pd.isna(score) or score == '' or score == 'nan':
                return None
            
            # 清理比分字符串
            score_str = str(score).strip().replace("'", "").replace('"', '')
            
            # 检查是否包含冒号
            if ':' not in score_str:
                return None
            
            # 分割比分
            parts = score_str.split(':')
            if len(parts) != 2:
                return None
            
            # 尝试转换为整数
            try:
                home = int(parts[0].strip())
                away = int(parts[1].strip())
            except ValueError:
                return None
            
            # 检查比分是否为非负数
            if home < 0 or away < 0:
                return None
            
            # 确定比赛结果
            if home > away:
                return 'win'
            elif home == away:
                return 'draw'
            else:
                return 'lose'
                
        except Exception as e:
            print(f"解析比分时出错: {score}, 错误: {e}")
            return None

    def _extract_handicap(self, handicap_str):
        """从让球字符串中提取让球数"""
        try:
            if pd.isna(handicap_str) or handicap_str == '':
                return 0
            match = re.search(r'（([+-]?\d+)）', str(handicap_str))
            if match:
                return int(match.group(1))
            return 0
        except:
            return 0