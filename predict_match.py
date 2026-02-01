#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
足球比赛预测命令行工具

使用示例:
python predict_match.py --home_team "多特蒙德" --away_team "蒙特雷" --league "世俱杯" --win_odds 1.57 --draw_odds 3.65 --lose_odds 4.60

python predict_match.py --home_team "皇家马德里" --away_team "巴塞罗那" --league "西甲" --win_odds 2.10 --draw_odds 3.40 --lose_odds 3.60 --handicap -0.5
"""

import argparse
import json
import sys
import os
from datetime import datetime
import traceback

# 导入预测模型
try:
    from football_ml_predictor import FootballPredictor
    from football_dl_predictor import FootballDLPredictor
except ImportError as e:
    print(f"❌ 导入模型失败: {e}")
    print("请确保 football_ml_predictor.py 和 football_dl_predictor.py 文件存在")
    sys.exit(1)

def load_models():
    """加载训练好的模型"""
    ml_predictor = None
    dl_predictor = None
    
    # 加载机器学习模型
    try:
        if os.path.exists('football_models.pkl'):
            ml_predictor = FootballPredictor()
            ml_predictor.load_models('football_models.pkl')
            print("✅ 机器学习模型加载成功")
        else:
            print("⚠️  未找到机器学习模型文件 (football_models.pkl)")
    except Exception as e:
        print(f"❌ 机器学习模型加载失败: {e}")
    
    # 加载深度学习模型
    try:
        if os.path.exists('football_dl_model.pth'):
            dl_predictor = FootballDLPredictor()
            dl_predictor.load_model('football_dl_model.pth')
            print("✅ 深度学习模型加载成功")
        else:
            print("⚠️  未找到深度学习模型文件 (football_dl_model.pth)")
    except Exception as e:
        print(f"❌ 深度学习模型加载失败: {e}")
    
    return ml_predictor, dl_predictor

def format_prediction_result(prediction, model_name):
    """格式化预测结果"""
    if not prediction:
        return f"❌ {model_name}模型预测失败"
    
    result_emoji = {
        'win': '🏆',
        'draw': '🤝', 
        'lose': '💔'
    }
    
    result_text = {
        'win': '主胜',
        'draw': '平局',
        'lose': '客胜'
    }
    
    output = []
    output.append(f"\n🤖 === {model_name}模型预测结果 ===")
    
    # 主要预测结果
    result = prediction.get('result', 'unknown')
    emoji = result_emoji.get(result, '❓')
    text = result_text.get(result, '未知')
    output.append(f"{emoji} 预测结果: {text}")
    
    # 概率分布
    probs = prediction.get('probabilities', {})
    if probs:
        output.append("\n📊 胜平负概率:")
        output.append(f"   🏆 主胜: {probs.get('win', 0):.1%}")
        output.append(f"   🤝 平局: {probs.get('draw', 0):.1%}")
        output.append(f"   💔 客胜: {probs.get('lose', 0):.1%}")
    
    # 进球预测
    total_goals = prediction.get('total_goals', 'N/A')
    total_goals_raw = prediction.get('total_goals_raw', 0)
    output.append(f"\n⚽ 总进球数: {total_goals} (原始值: {total_goals_raw:.1f})")
    
    # 比分预测
    predicted_score = prediction.get('predicted_score', 'N/A')
    output.append(f"🎯 预测比分: {predicted_score}")
    
    # 详细进球数（ML模型特有）
    goals_pred = prediction.get('goals_prediction')
    if goals_pred:
        output.append(f"📈 详细进球: 主队{goals_pred['home']}球, 客队{goals_pred['away']}球")
    
    # 前三可能比分
    top_scores = prediction.get('top_scores', [])
    if top_scores:
        output.append("\n🏅 前三可能比分:")
        for i, (score, prob) in enumerate(top_scores[:3], 1):
            output.append(f"   {i}. {score} ({prob:.1%})")
    
    # 未知类别警告
    unknown_cats = prediction.get('unknown_categories', {})
    if any(unknown_cats.values()):
        output.append("\n⚠️  未知类别警告:")
        if unknown_cats.get('home_team'):
            output.append("   - 主队在训练数据中未见过")
        if unknown_cats.get('away_team'):
            output.append("   - 客队在训练数据中未见过")
        if unknown_cats.get('league'):
            output.append("   - 联赛在训练数据中未见过")
        output.append("   注意: 预测结果可能不够准确")
    
    return '\n'.join(output)

def compare_predictions(ml_pred, dl_pred):
    """对比两个模型的预测结果"""
    if not ml_pred or not dl_pred:
        return ""
    
    output = []
    output.append("\n🔍 === 模型对比分析 ===")
    
    # 结果一致性
    ml_result = ml_pred.get('result')
    dl_result = dl_pred.get('result')
    
    if ml_result == dl_result:
        result_text = {'win': '主胜', 'draw': '平局', 'lose': '客胜'}.get(ml_result, '未知')
        output.append(f"✅ 预测一致: 两模型都预测 {result_text}")
    else:
        ml_text = {'win': '主胜', 'draw': '平局', 'lose': '客胜'}.get(ml_result, '未知')
        dl_text = {'win': '主胜', 'draw': '平局', 'lose': '客胜'}.get(dl_result, '未知')
        output.append(f"⚠️  预测不一致: ML模型预测{ml_text}, DL模型预测{dl_text}")
    
    # 概率对比
    ml_probs = ml_pred.get('probabilities', {})
    dl_probs = dl_pred.get('probabilities', {})
    
    if ml_probs and dl_probs:
        output.append("\n📈 概率对比:")
        for result_type in ['win', 'draw', 'lose']:
            ml_prob = ml_probs.get(result_type, 0)
            dl_prob = dl_probs.get(result_type, 0)
            diff = abs(ml_prob - dl_prob)
            result_name = {'win': '主胜', 'draw': '平局', 'lose': '客胜'}[result_type]
            output.append(f"   {result_name}: ML {ml_prob:.1%} vs DL {dl_prob:.1%} (差异{diff:.1%})")
    
    # 进球数对比
    ml_goals = ml_pred.get('total_goals_raw', 0)
    dl_goals = dl_pred.get('total_goals_raw', 0)
    goals_diff = abs(ml_goals - dl_goals)
    
    output.append(f"\n⚽ 进球数对比:")
    output.append(f"   ML模型: {ml_goals:.1f}球")
    output.append(f"   DL模型: {dl_goals:.1f}球")
    output.append(f"   差异: {goals_diff:.1f}球")
    
    if goals_diff < 0.3:
        output.append("   ✅ 进球数预测高度一致")
    elif goals_diff < 0.5:
        output.append("   ⚠️  进球数预测略有差异")
    else:
        output.append("   ❌ 进球数预测差异较大")
    
    # 投注建议
    output.append("\n💡 综合建议:")
    
    # 胜负判断
    if ml_result == dl_result:
        max_prob = max(ml_probs.get(ml_result, 0), dl_probs.get(dl_result, 0))
        if max_prob > 0.5:
            output.append("   🔥 强烈推荐: 两模型高置信度一致预测")
        elif max_prob > 0.4:
            output.append("   ⭐ 推荐: 两模型预测一致")
        else:
            output.append("   ⚠️  谨慎: 虽然预测一致但置信度不高")
    else:
        output.append("   ❓ 观望: 两模型预测不一致，建议谨慎投注")
    
    # 进球数建议
    if goals_diff < 0.3:
        avg_goals = (ml_goals + dl_goals) / 2
        if avg_goals < 2.25:
            output.append("   ⚽ 进球数: 建议考虑小球 (低于2.5球)")
        elif avg_goals > 2.75:
            output.append("   ⚽ 进球数: 建议考虑大球 (超过2.5球)")
        else:
            output.append("   ⚽ 进球数: 2.5球附近，不建议投注大小球")
    
    return '\n'.join(output)

def generate_json_output(ml_pred, dl_pred, match_info):
    """生成JSON格式的输出"""
    return {
        'match_info': match_info,
        'timestamp': datetime.now().isoformat(),
        'predictions': {
            'ml_model': ml_pred,
            'dl_model': dl_pred
        },
        'comparison': {
            'result_consistent': ml_pred.get('result') == dl_pred.get('result') if ml_pred and dl_pred else False,
            'goals_difference': abs(ml_pred.get('total_goals_raw', 0) - dl_pred.get('total_goals_raw', 0)) if ml_pred and dl_pred else None
        }
    }

def main():
    parser = argparse.ArgumentParser(
        description='足球比赛预测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python predict_match.py --home_team "多特蒙德" --away_team "蒙特雷" --league "世俱杯" --win_odds 1.57 --draw_odds 3.65 --lose_odds 4.60
  
  python predict_match.py --home_team "皇家马德里" --away_team "巴塞罗那" --league "西甲" --win_odds 2.10 --draw_odds 3.40 --lose_odds 3.60 --handicap -0.5
        """
    )
    
    # 必需参数
    parser.add_argument('--home_team', required=True, help='主队名称')
    parser.add_argument('--away_team', required=True, help='客队名称')
    parser.add_argument('--league', required=True, help='联赛名称')
    parser.add_argument('--win_odds', type=float, required=True, help='主胜赔率')
    parser.add_argument('--draw_odds', type=float, required=True, help='平局赔率')
    parser.add_argument('--lose_odds', type=float, required=True, help='客胜赔率')
    
    # 可选参数
    parser.add_argument('--handicap', type=float, default=0, help='让球数 (默认: 0)')
    parser.add_argument('--output_json', action='store_true', help='输出JSON格式结果')
    parser.add_argument('--quiet', action='store_true', help='静默模式，只输出结果')
    parser.add_argument('--ml_only', action='store_true', help='只使用机器学习模型')
    parser.add_argument('--dl_only', action='store_true', help='只使用深度学习模型')
    
    args = parser.parse_args()
    
    # 验证赔率
    if args.win_odds <= 1 or args.draw_odds <= 1 or args.lose_odds <= 1:
        print("❌ 错误: 赔率必须大于1")
        sys.exit(1)
    
    # 比赛信息
    match_info = {
        'home_team': args.home_team,
        'away_team': args.away_team,
        'league': args.league,
        'win_odds': args.win_odds,
        'draw_odds': args.draw_odds,
        'lose_odds': args.lose_odds,
        'handicap': args.handicap
    }
    
    if not args.quiet:
        print("⚽ === 足球比赛预测工具 ===")
        print(f"🏠 主队: {args.home_team}")
        print(f"✈️  客队: {args.away_team}")
        print(f"🏆 联赛: {args.league}")
        print(f"💰 赔率: 胜{args.win_odds} 平{args.draw_odds} 负{args.lose_odds}")
        if args.handicap != 0:
            print(f"⚖️  让球: {args.handicap:+.1f}")
        print()
    
    # 加载模型
    if not args.quiet:
        print("🔄 正在加载预测模型...")
    
    ml_predictor, dl_predictor = load_models()
    
    if not ml_predictor and not dl_predictor:
        print("❌ 没有可用的预测模型，请先训练模型")
        sys.exit(1)
    
    # 根据参数决定使用哪些模型
    if args.ml_only:
        dl_predictor = None
    elif args.dl_only:
        ml_predictor = None
    
    # 进行预测
    ml_prediction = None
    dl_prediction = None
    
    try:
        if ml_predictor:
            if not args.quiet:
                print("🤖 正在使用机器学习模型预测...")
            ml_prediction = ml_predictor.predict_match(
                args.home_team, args.away_team, args.league,
                args.win_odds, args.draw_odds, args.lose_odds, args.handicap
            )
        
        if dl_predictor:
            if not args.quiet:
                print("🧠 正在使用深度学习模型预测...")
            dl_prediction = dl_predictor.predict_match(
                args.home_team, args.away_team, args.league,
                args.win_odds, args.draw_odds, args.lose_odds, args.handicap
            )
    
    except Exception as e:
        print(f"❌ 预测过程中发生错误: {e}")
        if not args.quiet:
            print("\n🔍 错误详情:")
            traceback.print_exc()
        sys.exit(1)
    
    # 输出结果
    if args.output_json:
        # JSON输出
        result = generate_json_output(ml_prediction, dl_prediction, match_info)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # 标准输出
        if ml_prediction:
            print(format_prediction_result(ml_prediction, "机器学习"))
        
        if dl_prediction:
            print(format_prediction_result(dl_prediction, "深度学习"))
        
        if ml_prediction and dl_prediction:
            print(compare_predictions(ml_prediction, dl_prediction))
    
    if not args.quiet:
        print(f"\n✅ 预测完成! 用时: {datetime.now().strftime('%H:%M:%S')}")
        print("\n📌 免责声明: 预测结果仅供参考，不构成投注建议。请理性对待体育博彩。")

if __name__ == '__main__':
    main() 