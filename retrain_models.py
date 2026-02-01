#!/usr/bin/env python3
"""
模型重训练脚本
用于定期使用最新的历史数据重新训练ML和DL模型
采用按时间切分的方式，用过去预测未来
"""

import os
import sys
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('retrain_models.log', encoding='utf-8')
    ]
)

def retrain_ml_model():
    """重新训练机器学习模型"""
    try:
        logging.info("开始重新训练机器学习模型...")
        
        from football_ml_predictor import FootballPredictor
        
        # 创建预测器
        predictor = FootballPredictor()
        
        # 加载数据
        logging.info("加载历史数据...")
        predictor.load_data('jc_history_api.csv')
        
        # 数据预处理
        logging.info("进行数据预处理...")
        predictor.preprocess_data()
        
        # 特征工程
        logging.info("进行特征工程...")
        predictor.feature_engineering()
        
        # 训练模型（使用按时间切分）
        logging.info("开始训练模型（按时间切分）...")
        predictor.train_models(n_trials=30)  # 减少试验次数以节省时间
        
        # 保存模型
        logging.info("保存模型...")
        predictor.save_models('football_models.pkl')
        
        logging.info("机器学习模型重训练完成！")
        return True
        
    except Exception as e:
        logging.error(f"机器学习模型重训练失败: {e}", exc_info=True)
        return False

def retrain_dl_model():
    """重新训练深度学习模型"""
    try:
        logging.info("开始重新训练深度学习模型...")
        
        from football_dl_predictor import FootballDLPredictor
        
        # 创建预测器
        predictor = FootballDLPredictor()
        
        # 加载数据
        logging.info("加载历史数据...")
        predictor.load_data('jc_history_api.csv')
        
        # 数据预处理
        logging.info("进行数据预处理...")
        predictor.preprocess_data()
        
        # 特征工程
        logging.info("进行特征工程...")
        predictor.feature_engineering()
        
        # 训练模型（使用按时间切分）
        logging.info("开始训练模型（按时间切分）...")
        predictor.train_model(
            batch_size=64, 
            epochs=50,  # 减少epochs以节省时间
            learning_rate=0.001,
            patience=5
        )
        
        # 保存模型
        logging.info("保存模型...")
        predictor.save_model('football_dl_model.pth')
        
        logging.info("深度学习模型重训练完成！")
        return True
        
    except Exception as e:
        logging.error(f"深度学习模型重训练失败: {e}", exc_info=True)
        return False

def main():
    """主函数"""
    print("="*60)
    print("足球预测模型重训练脚本")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 检查数据文件是否存在
    if not os.path.exists('jc_history_api.csv'):
        logging.error("历史数据文件 jc_history_api.csv 不存在！")
        print("请先运行 update_historical_data.py 获取历史数据")
        return
    
    success_count = 0
    
    # 重训练机器学习模型
    print("\n1. 重新训练机器学习模型...")
    if retrain_ml_model():
        success_count += 1
        print("✅ 机器学习模型重训练成功")
    else:
        print("❌ 机器学习模型重训练失败")
    
    # 重训练深度学习模型
    # print("\n2. 重新训练深度学习模型...")
    # if retrain_dl_model():
    #     success_count += 1
    #     print("✅ 深度学习模型重训练成功")
    # else:
    #     print("❌ 深度学习模型重训练失败")
    
    print("\n" + "="*60)
    print(f"重训练完成: {success_count}/2 个模型成功")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    if success_count == 2:
        print("\n🎉 所有模型重训练成功！现在可以重启Web服务以使用新模型。")
    elif success_count == 1:
        print("\n⚠️  部分模型重训练成功，请检查日志文件了解详情。")
    else:
        print("\n💥 所有模型重训练失败，请检查日志文件了解详情。")

if __name__ == "__main__":
    main() 