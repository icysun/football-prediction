import re
import time
import random
import requests
from datetime import datetime
from itertools import combinations
from requests.exceptions import RequestException
import json

'''
足球比赛数据采集和AI分析模块
整合多个数据源：盘口变化、历史交锋、小组排名、比赛近况、射手信息、伤员情况、支持率等
为AI提供全面的比赛分析信息
'''

# 腾讯混元配置
HUNYUAN_API_KEY = [
                "sk-ixxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
]
HUNYUAN_API_URL = "https://api.hunyuan.cloud.tencent.com/v1/chat/completions"

# 通用请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://www.sporttery.cn/"
}

# 优化后的AI提示词模板 - 整合所有分析因素
AI_COMPREHENSIVE_PROMPT = """你是一位专业的足球分析师，请基于以下信息，分析足球比赛并严格按JSON格式返回分析结果：

【比赛基本信息】
- 比赛编号: {match_id}
- 比赛时间: {match_time}
- 联赛: {league_name}
- 主队: {home_team}
- 客队: {away_team}
- 初始赔率: 胜 {win_odds}, 平 {draw_odds}, 负 {lose_odds}

【详细分析数据】
- 盘口分析: {odds_analysis}
- 历史交锋: {head_to_head}
- 特征分析: {feature_analysis}
- 联赛排名: {group_standings}
- 关键射手: {top_scorers}
- 伤停情况: {injury_suspension}
- 支持率: {support_rate}

【分析要求】
请综合以上所有信息，进行深度分析，并严格按照以下JSON格式返回结果（不要包含任何其他文字、注释或代码块标记）：
{{
    "match_id": "{match_id}",
    "match_info": "{home_team} vs {away_team}",
    "win_rate": <主队获胜的概率，整数>,
    "draw_rate": <平局的概率，整数>,
    "lose_rate": <客队获胜的概率，整数>,
    "total_goals_prediction": <预计总进球数，整数>,
    "score_prediction": "<最可能的比分，例如 '2-1'>",
    "confidence_level": "<高/中/低>",
    "analysis_reason": "<简要分析理由，说明你的预测依据>"
}}"""

MATCH_TIME_FORMAT = "%Y-%m-%d %H:%M"

def validate_api_keys(keys):
    """验证API密钥有效性"""
    valid_keys = []
    for key in keys:
        try:
            response = requests.get(
                "https://api.hunyuan.cloud.tencent.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=5
            )
            if response.status_code == 200:
                valid_keys.append(key)
        except:
            continue
    return valid_keys

def safe_request(url, params=None, headers=None, timeout=10):
    """安全的HTTP请求包装"""
    try:
        response = requests.get(url, params=params, headers=headers or HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"请求失败 {url}: {e}")
        return None

def fetch_match_data():
    """获取今日在售比赛数据"""
    url = "https://webapi.sporttery.cn/gateway/uniform/football/getMatchListV1.qry"
    params = {"clientCode": 3001}
    
    data = safe_request(url, params)
    if not data or not data.get('success'):
        return "获取比赛数据失败"

    matches = []
    for match_group in data.get('value', {}).get('matchInfoList', []):
        for match in match_group.get('subMatchList', []):
            if match.get('matchStatus') != 'Selling':
                continue
            
            had_odds = next((o for o in match.get('oddsList', []) if o.get('poolCode') == 'HAD'), None)
            if not had_odds or not all([had_odds.get('h'), had_odds.get('d'), had_odds.get('a')]):
                continue

            # 获取真正的数字ID
            real_match_id = match.get('matchId')
            match_num_str = match.get('matchNumStr')
            
            # 确保我们有有效的matchId
            if not real_match_id:
                print(f"警告: 比赛 {match_num_str} 缺少matchId，跳过")
                continue

            match_info = {
                '比赛编号': match_num_str,  # 显示用的编号（如"周日005"）
                '联赛名称': match['leagueAllName'],
                '比赛时间': f"{match['matchDate']} {match['matchTime']}",
                '主队': match['homeTeamAllName'],
                '客队': match['awayTeamAllName'],
                '胜赔率': had_odds['h'],
                '平赔率': had_odds['d'],
                '负赔率': had_odds['a'],
                'matchId': real_match_id,  # API调用用的真正ID
                'realMatchId': real_match_id  # 备用字段
            }
            matches.append(match_info)

    return matches if matches else "今日无在售比赛数据"

def fetch_odds_changes(match_id):
    """获取盘口变化信息"""
    url = "https://webapi.sporttery.cn/gateway/uniform/football/getFixedBonusV1.qry"
    params = {"clientCode": 3001, "matchId": match_id}
    
    data = safe_request(url, params)
    if not data or not data.get('success'):
        return "盘口数据获取失败"
    
    value = data.get('value', {})
    if not value:
        return "暂无盘口数据"
    
    analysis = []
    
    # 检查oddsHistory中的数据
    odds_history = value.get('oddsHistory', {})
    if odds_history:
        # 检查胜平负盘口数据
        had_list = odds_history.get('hadList', [])
        if had_list:
            latest_had = had_list[-1]  # 取最新的赔率
            analysis.append("胜平负盘口:")
            analysis.append(f"  胜: {latest_had.get('h', 'N/A')}")
            analysis.append(f"  平: {latest_had.get('d', 'N/A')}")
            analysis.append(f"  负: {latest_had.get('a', 'N/A')}")
            
            # 如果有历史变化，显示变化趋势
            if len(had_list) > 1:
                first_had = had_list[0]
                analysis.append(f"  变化: 胜{first_had.get('h', 'N/A')}→{latest_had.get('h', 'N/A')}")
        
        # 检查让球盘口
        hhad_list = odds_history.get('hhadList', [])
        if hhad_list:
            latest_hhad = hhad_list[-1]  # 取最新的赔率
            analysis.append("让球盘口:")
            analysis.append(f"  让球数: {latest_hhad.get('goalLine', 'N/A')}")
            analysis.append(f"  胜: {latest_hhad.get('h', 'N/A')}")
            analysis.append(f"  平: {latest_hhad.get('d', 'N/A')}")
            analysis.append(f"  负: {latest_hhad.get('a', 'N/A')}")
    
    return "\n".join(analysis) if analysis else "盘口信息暂不可用"

def fetch_head_to_head(match_id):
    """获取历史交锋记录"""
    url = "https://webapi.sporttery.cn/gateway/uniform/football/getResultHistoryV1.qry"
    params = {
        "sportteryMatchId": match_id,
        "termLimits": 10,
        "tournamentFlag": 0,
        "homeAwayFlag": 0
    }
    
    data = safe_request(url, params)
    if not data or not data.get('success'):
        return "历史交锋数据获取失败"
    
    value = data.get('value')
    if not value:
        return "历史交锋数据暂不可用"
    
    match_list = value.get('matchList', [])
    
    if not match_list:
        return "暂无历史交锋记录"
    
    analysis = []
    home_wins = draw_count = away_wins = 0
    
    for match in match_list[:10]:
        # 新的数据结构
        winning_team = match.get('winningTeam', '')
        score = match.get('fullCourtGoal', '')
        match_date = match.get('matchDate', '')
        home_team = match.get('homeTeamShortName', '')
        away_team = match.get('awayTeamShortName', '')
        
        # 转换胜负结果
        if winning_team == 'home':
            result = '胜'
            home_wins += 1
        elif winning_team == 'away':
            result = '负'
            away_wins += 1
        elif winning_team == 'draw':
            result = '平'
            draw_count += 1
        else:
            result = '未知'
        
        if len(analysis) < 5:  # 只显示最近5场
            analysis.append(f"{match_date} {home_team} {score} {away_team} ({result})")
    
    summary = f"近{len(match_list)}场交锋: 主队{home_wins}胜{draw_count}平{away_wins}负"
    return f"{summary}\n" + "\n".join(analysis)

def fetch_feature_analysis(match_id):
    """获取比赛特征分析"""
    url = "https://webapi.sporttery.cn/gateway/uniform/football/getMatchFeatureV1.qry"
    params = {"sportteryMatchId": match_id, "termLimits": 10}

    data = safe_request(url, params)
    if not data or not data.get('success'):
        return "特征分析数据获取失败"

    value = data.get('value')
    required_keys = ['homeTeamShortName', 'awayTeamShortName', 'last', 'sameHomeAway', 'eachHomeAway', 'eachSameHomeAway', 'goalAvg', 'lossGoalAvg']
    if not value or not all(key in value for key in required_keys):
        return "特征分析数据暂不可用"
        
    home_name = value.get('homeTeamShortName', '主队')
    away_name = value.get('awayTeamShortName', '客队')

    # 数据提取
    last = value['last']
    same_ha = value['sameHomeAway']
    recent = value['eachHomeAway']
    recent_same_ha = value['eachSameHomeAway']
    goal_avg = value['goalAvg']
    loss_avg = value['lossGoalAvg']

    analysis = []

    # 主队
    analysis.append(f" 近10场交锋   {home_name} (主)   {last.get('homeWinGoalMatchCnt', 0)}胜/{last.get('homeDrawMatchCnt', 0)}平/{last.get('homeLossGoalMatchCnt', 0)}负")
    analysis.append(f" 近10场同主客队交锋   {same_ha.get('homeWinGoalMatchCnt', 0)}胜/{same_ha.get('homeDrawMatchCnt', 0)}平/{same_ha.get('homeLossGoalMatchCnt', 0)}负")
    analysis.append(f" 近10场战况     {recent.get('homeWinGoalMatchCnt', 0)}胜/{recent.get('homeDrawMatchCnt', 0)}平/{recent.get('homeLossGoalMatchCnt', 0)}负")
    analysis.append(f" 同主客队战况  {recent_same_ha.get('homeWinGoalMatchCnt', 0)}胜/{recent_same_ha.get('homeDrawMatchCnt', 0)}平/{recent_same_ha.get('homeLossGoalMatchCnt', 0)}负")
    analysis.append(f" 主队场均进球   {goal_avg.get('homeGoalAvgCnt', '0.0')}个")
    analysis.append(f" 主队场均失球   {loss_avg.get('homeLossGoalAvgCnt', '0.0')}个")

    # 客队
    analysis.append(f" 近10场交锋  {away_name} (客)   {last.get('awayWinGoalMatchCnt', 0)}胜/{last.get('awayDrawMatchCnt', 0)}平/{last.get('awayLossGoalMatchCnt', 0)}负")
    analysis.append(f" 同主客交锋    {same_ha.get('awayWinGoalMatchCnt', 0)}胜/{same_ha.get('awayDrawMatchCnt', 0)}平/{same_ha.get('awayLossGoalMatchCnt', 0)}负")
    analysis.append(f" 近10场战况   {recent.get('awayWinGoalMatchCnt', 0)}胜/{recent.get('awayDrawMatchCnt', 0)}平/{recent.get('awayLossGoalMatchCnt', 0)}负")
    analysis.append(f" 同主客战况   {recent_same_ha.get('awayWinGoalMatchCnt', 0)}胜/{recent_same_ha.get('awayDrawMatchCnt', 0)}平/{recent_same_ha.get('awayLossGoalMatchCnt', 0)}负")
    analysis.append(f" 场均进球     {goal_avg.get('awayGoalAvgCnt', '0.0')}个")
    analysis.append(f" 场均失球     {loss_avg.get('awayLossGoalAvgCnt', '0.0')}个")

    return "\n".join(analysis)

def fetch_group_standings(match_id):
    """获取小组赛排名"""
    url = "https://webapi.sporttery.cn/gateway/uniform/football/getMatchTablesV2.qry"
    params = {"gmMatchId": match_id}
    
    data = safe_request(url, params)
    if not data or not data.get('success'):
        return "小组排名数据获取失败"
    
    value = data.get('value')
    if not value:
        return "小组排名数据暂不可用"
    
    analysis = []
    
    # 检查主队数据
    home_tables = value.get('homeTables', {})
    if home_tables:
        home_total = home_tables.get('total', {})
        if home_total:
            team_name = home_total.get('teamShortName', '主队')
            ranking = home_total.get('ranking', 'N/A')
            points = home_total.get('points', 'N/A')
            total_legs = home_total.get('totalLegCnt', 'N/A')
            win_goals = home_total.get('winGoalMatchCnt', 'N/A')
            draw_goals = home_total.get('drawMatchCnt', 'N/A')
            loss_goals = home_total.get('lossGoalMatchCnt', 'N/A')
            
            analysis.append(f"主队排名: {team_name}")
            analysis.append(f"  排名: 第{ranking}名")
            analysis.append(f"  积分: {points}分")
            analysis.append(f"  战绩: {total_legs}场 {win_goals}胜{draw_goals}平{loss_goals}负")
    
    # 检查客队数据
    away_tables = value.get('awayTables', {})
    if away_tables:
        away_total = away_tables.get('total', {})
        if away_total:
            team_name = away_total.get('teamShortName', '客队')
            ranking = away_total.get('ranking', 'N/A')
            points = away_total.get('points', 'N/A')
            total_legs = away_total.get('totalLegCnt', 'N/A')
            win_goals = away_total.get('winGoalMatchCnt', 'N/A')
            draw_goals = away_total.get('drawMatchCnt', 'N/A')
            loss_goals = away_total.get('lossGoalMatchCnt', 'N/A')
            
            analysis.append(f"客队排名: {team_name}")
            analysis.append(f"  排名: 第{ranking}名")
            analysis.append(f"  积分: {points}分")
            analysis.append(f"  战绩: {total_legs}场 {win_goals}胜{draw_goals}平{loss_goals}负")
    
    # 添加联赛信息
    league_name = value.get('leagueShortName', '')
    season_name = value.get('seasonName', '')
    if league_name and season_name:
        analysis.insert(0, f"联赛: {league_name} {season_name}")
    
    return "\n".join(analysis) if analysis else "小组排名信息不足"

def fetch_top_scorers(match_id):
    """获取射手信息"""
    url = "https://webapi.sporttery.cn/gateway/uniform/football/getMatchPlayerV1.qry"
    params = {"sportteryMatchId": match_id, "termLimits": 3}
    
    data = safe_request(url, params)
    if not data or not data.get('success'):
        return "射手信息获取失败"
    
    value = data.get('value')
    if not value:
        return "射手信息暂不可用"
    
    analysis = []
    
    # 主队射手
    home_data = value.get('home', {})
    home_players = home_data.get('playerList', []) if home_data else []
    if home_players:
        analysis.append("主队射手:")
        for player in home_players[:3]:
            name = player.get('personName', 'N/A')
            goals = player.get('goalCnt', 0)
            position = player.get('playerPositionDesc', '')
            goal_prob = player.get('goalProbability', '')
            analysis.append(f"  {name} ({position}) {goals}球 进球率:{goal_prob}")
    
    # 客队射手
    away_data = value.get('away', {})
    away_players = away_data.get('playerList', []) if away_data else []
    if away_players:
        analysis.append("客队射手:")
        for player in away_players[:3]:
            name = player.get('personName', 'N/A')
            goals = player.get('goalCnt', 0)
            position = player.get('playerPositionDesc', '')
            goal_prob = player.get('goalProbability', '')
            analysis.append(f"  {name} ({position}) {goals}球 进球率:{goal_prob}")
    
    return "\n".join(analysis) if analysis else "射手信息不足"

def fetch_injury_suspension(match_id):
    """获取伤员停赛信息"""
    url = "https://webapi.sporttery.cn/gateway/uniform/football/getInjurySuspensionV1.qry"
    params = {"sportteryMatchId": match_id}
    
    data = safe_request(url, params)
    if not data or not data.get('success'):
        return "伤员信息获取失败"
    
    value = data.get('value')
    if not value:
        return "伤员信息暂不可用"
    
    analysis = []
    
    # 主队伤员
    home_data = value.get('home', {})
    home_injuries = home_data.get('injuriesAndSuspensionsList', []) if home_data else []
    if home_injuries:
        analysis.append("主队伤停:")
        for injury in home_injuries[:5]:
            name = injury.get('personName', 'N/A')
            position = injury.get('playerPositionDesc', '')
            injury_flag = injury.get('injuryFlag', 0)
            suspension_flag = injury.get('suspensionFlag', 0)
            
            status = []
            if injury_flag:
                status.append("伤")
            if suspension_flag:
                status.append("停")
            
            status_str = "/".join(status) if status else "正常"
            analysis.append(f"  {name} ({position}) - {status_str}")
    
    # 客队伤员
    away_data = value.get('away', {})
    away_injuries = away_data.get('injuriesAndSuspensionsList', []) if away_data else []
    if away_injuries:
        analysis.append("客队伤停:")
        for injury in away_injuries[:5]:
            name = injury.get('personName', 'N/A')
            position = injury.get('playerPositionDesc', '')
            injury_flag = injury.get('injuryFlag', 0)
            suspension_flag = injury.get('suspensionFlag', 0)
            
            status = []
            if injury_flag:
                status.append("伤")
            if suspension_flag:
                status.append("停")
            
            status_str = "/".join(status) if status else "正常"
            analysis.append(f"  {name} ({position}) - {status_str}")
    
    if not analysis:
        return "暂无伤员停赛信息"
    
    return "\n".join(analysis)

def fetch_support_rate(match_ids):
    """获取支持率信息"""
    if isinstance(match_ids, str):
        match_ids = [match_ids]
    
    # 确保match_ids是字符串列表
    match_ids = [str(mid) for mid in match_ids if mid]
    
    if not match_ids:
        return "支持率信息不足"
    
    url = "https://webapi.sporttery.cn/gateway/jc/common/getSupportRateV1.qry"
    params = {
        "matchIds": ",".join(match_ids),
        "poolCode": "hhad,had",
        "sportType": 1
    }
    
    data = safe_request(url, params)
    if not data or not data.get('success'):
        return "支持率数据获取失败"
    
    value = data.get('value', {})
    analysis = []
    
    for match_id in match_ids:
        # API返回的key格式是 "_" + match_id
        match_key = f"_{match_id}"
        match_support = value.get(match_key, {})
        
        # 胜平负支持率
        had_support = match_support.get('HAD', {})
        if had_support:
            h_rate = had_support.get('hSupportRate', 'N/A')
            d_rate = had_support.get('dSupportRate', 'N/A')
            a_rate = had_support.get('aSupportRate', 'N/A')
            analysis.append(f"胜平负支持率: 胜{h_rate} 平{d_rate} 负{a_rate}")
        
        # 让球支持率
        hhad_support = match_support.get('HHAD', {})
        if hhad_support:
            h_rate = hhad_support.get('hSupportRate', 'N/A')
            d_rate = hhad_support.get('dSupportRate', 'N/A')
            a_rate = hhad_support.get('aSupportRate', 'N/A')
            analysis.append(f"让球支持率: 胜{h_rate} 平{d_rate} 负{a_rate}")
    
    return "\n".join(analysis) if analysis else "支持率信息不足"

def get_comprehensive_match_analysis(match_info):
    """获取比赛的全面分析信息"""
    match_id = match_info.get('matchId', match_info.get('比赛编号'))
    
    print(f"正在获取比赛 {match_id} 的详细信息...")
    
    # 并行获取各种信息（为了提高效率，可以考虑使用线程池）
    analysis_data = {
        'odds_analysis': fetch_odds_changes(match_id),
        'head_to_head': fetch_head_to_head(match_id),
        'feature_analysis': fetch_feature_analysis(match_id),
        'group_standings': fetch_group_standings(match_id),
        'top_scorers': fetch_top_scorers(match_id),
        'injury_suspension': fetch_injury_suspension(match_id),
        'support_rate': fetch_support_rate([match_id])
    }
    
    return analysis_data

def analyze_match_with_ai(match_info, analysis_data):
    """使用AI进行综合分析"""
    current_date = datetime.now().strftime('%Y年%m月%d日')
    
    # 检查数据可用性并提供fallback
    odds_analysis = analysis_data.get('odds_analysis', '盘口数据暂不可用')
    head_to_head = analysis_data.get('head_to_head', '历史交锋数据暂不可用')
    feature_analysis = analysis_data.get('feature_analysis', '特征分析数据暂不可用')
    group_standings = analysis_data.get('group_standings', '排名数据暂不可用')
    top_scorers = analysis_data.get('top_scorers', '射手信息暂不可用')
    injury_suspension = analysis_data.get('injury_suspension', '伤员信息暂不可用')
    support_rate = analysis_data.get('support_rate', '支持率数据暂不可用')
    
    # 构建完整的提示词
    prompt = AI_COMPREHENSIVE_PROMPT.format(
        match_id=match_info.get('比赛编号', 'N/A'),
        match_time=match_info.get('比赛时间', 'N/A'),
        league_name=match_info.get('联赛名称', '未知联赛'),
        home_team=match_info.get('主队', '未知主队'),
        away_team=match_info.get('客队', '未知客队'),
        win_odds=match_info.get('胜赔率', 'N/A'),
        draw_odds=match_info.get('平赔率', 'N/A'),
        lose_odds=match_info.get('负赔率', 'N/A'),
        odds_analysis=odds_analysis,
        head_to_head=head_to_head,
        feature_analysis=feature_analysis,
        group_standings=group_standings,
        top_scorers=top_scorers,
        injury_suspension=injury_suspension,
        support_rate=support_rate
    )

    # 调用AI API
    valid_keys = validate_api_keys(HUNYUAN_API_KEY)
    if not valid_keys:
        return {
            'prediction': '分析失败',
            'win_rate': '50',
            'reason': 'API密钥验证失败'
        }
    
    random_api_key = random.choice(valid_keys)
    payload = {
        "model": "hunyuan-t1-20250529",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": False
    }
    
    headers = {
        "Authorization": f"Bearer {random_api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"正在请求AI分析...")
        time.sleep(2)  # 控制请求频率
        
        response = requests.post(HUNYUAN_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        ai_response = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        print(f"AI原始回复: {ai_response}")
        
        # 解析AI回复
        try:
            # 清理响应文本，确保是有效的JSON
            content = ai_response.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            analysis_result = json.loads(content)
            
            # 验证返回的结果
            required_result_fields = ['match_id', 'match_info', 'win_rate', 'draw_rate', 
                                   'lose_rate', 'total_goals_prediction', 'score_prediction', 
                                   'confidence_level', 'analysis_reason']
            for field in required_result_fields:
                if field not in analysis_result:
                    return parse_ai_response_intelligently(ai_response, match_info)

            return analysis_result

        except json.JSONDecodeError:
            return parse_ai_response_intelligently(ai_response, match_info)
        
    except requests.exceptions.Timeout:
        return {
            'prediction': '分析超时',
            'win_rate': '50',
            'reason': 'AI分析请求超时，请稍后重试'
        }
    except requests.exceptions.RequestException as e:
        return {
            'prediction': '网络错误',
            'win_rate': '50',
            'reason': f'网络请求失败: {str(e)[:50]}'
        }
    except Exception as e:
        print(f"AI分析失败: {e}")
        return {
            'prediction': '分析异常',
            'win_rate': '50',
            'reason': f'系统异常: {str(e)[:50]}'
        }

def parse_ai_response_intelligently(ai_response, match_info):
    """智能解析非标准格式的AI回复"""
    try:
        # 提取JSON部分
        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                # 补充缺失字段
                data.setdefault('match_id', match_info.get('比赛编号'))
                data.setdefault('match_info', f"{match_info.get('主队')} vs {match_info.get('客队')}")
                data.setdefault('analysis_reason', 'AI回复格式不规范，部分解析')
                return data
            except json.JSONDecodeError:
                pass

        # 如果JSON解析失败，则进行文本规则提取
        win_rate = re.search(r'["\']win_rate["\']:\s*(\d+)', ai_response)
        draw_rate = re.search(r'["\']draw_rate["\']:\s*(\d+)', ai_response)
        lose_rate = re.search(r'["\']lose_rate["\']:\s*(\d+)', ai_response)
        score_prediction = re.search(r'["\']score_prediction["\']:\s*["\']([^"\']+)["\']', ai_response)

        return {
            'match_id': match_info.get('比赛编号'),
            'match_info': f"{match_info.get('主队')} vs {match_info.get('客队')}",
            'win_rate': int(win_rate.group(1)) if win_rate else 50,
            'draw_rate': int(draw_rate.group(1)) if draw_rate else 25,
            'lose_rate': int(lose_rate.group(1)) if lose_rate else 25,
            'score_prediction': score_prediction.group(1) if score_prediction else "1-1",
            'confidence_level': "低",
            'analysis_reason': "AI回复格式错误，智能解析"
        }
    except Exception:
        return {}

def comprehensive_match_analysis(match_info):
    """获取并整合所有维度的分析数据"""
    print(f"\n开始分析比赛: {match_info.get('主队')} vs {match_info.get('客队')}")
    
    # 获取详细分析数据
    analysis_data = get_comprehensive_match_analysis(match_info)
    
    # 使用AI进行综合分析
    ai_result = analyze_match_with_ai(match_info, analysis_data)
    
    # 从返回的概率中确定预测结果
    if 'prediction' not in ai_result and all(k in ai_result for k in ['win_rate', 'draw_rate', 'lose_rate']):
        try:
            rates = {
                '胜': int(ai_result['win_rate']),
                '平': int(ai_result['draw_rate']),
                '负': int(ai_result['lose_rate'])
            }
            prediction = max(rates, key=lambda k: rates[k])
            ai_result['prediction'] = prediction
        except (ValueError, TypeError):
            pass  # 保持 'prediction' 未定义

    # 增加 prediction_raw 字段
    if 'prediction' in ai_result:
        if ai_result['prediction'] == '胜':
            ai_result['prediction_raw'] = 'win'
        elif ai_result['prediction'] == '平':
            ai_result['prediction_raw'] = 'draw'
        elif ai_result['prediction'] == '负':
            ai_result['prediction_raw'] = 'lose'
        else:
            ai_result['prediction_raw'] = 'unknown'

    # 最终返回整合后的结果
    final_result = {
        'match_info': match_info,
        **ai_result,  # 合并AI分析结果
        'analysis_data': analysis_data
    }
    
    print(f"分析完成: {ai_result.get('prediction', '未知')} (胜率: {ai_result.get('win_rate', 'N/A')}%)")
    print(f"理由: {ai_result.get('analysis_reason', ai_result.get('reason', 'N/A'))}")
    
    return final_result

# 历史数据获取函数保持不变
def fetch_historical_match_results(target_date):
    """获取历史比赛结果"""
    base_url = "https://webapi.sporttery.cn/gateway/uniform/football/getUniformMatchResultV1.qry"
    
    all_matches_with_results = []
    page_no = 1
    
    while True:
        params = {
            "matchBeginDate": target_date,
            "matchEndDate": target_date,
            "pageSize": 100,
            "pageNo": page_no,
            "isFix": 0,
            "matchPage": 1,
            "pcOrWap": 1
        }

        data = safe_request(base_url, params)
        if not data or not data.get('success'):
            break
            
        matches_on_page = data.get('value', {}).get('matchResult', [])
        if not matches_on_page:
            break

        for match in matches_on_page:
            match_id = match.get('matchNumStr')
            full_score_raw = match.get('sectionsNo999')

            if not match_id:
                continue

            # 计算赛果
            actual_result = ''
            if full_score_raw and ':' in full_score_raw:
                try:
                    home_score, away_score = map(int, full_score_raw.split(':'))
                    actual_result = '胜' if home_score > away_score else ('负' if home_score < away_score else '平')
                except (ValueError, TypeError):
                    pass

            # 构建完整的比赛信息
            match_info = {
                '比赛时间': target_date,
                '比赛编号': match_id,
                '联赛名称': match.get('leagueName', ''),
                '主队': match.get('allHomeTeam', match.get('homeTeam', '')),
                '客队': match.get('allAwayTeam', match.get('awayTeam', '')),
                '主队（让球）vs客队': f"{match.get('allHomeTeam', match.get('homeTeam', ''))}（{match.get('goalLine', '0')}）vs{match.get('allAwayTeam', match.get('awayTeam', ''))}",
                '半场比分': match.get('sectionsNo1', ''),
                '全场比分': full_score_raw or '',
                '胜赔率': match.get('h', ''),
                '平赔率': match.get('d', ''),
                '负赔率': match.get('a', ''),
                'actual_result': actual_result
            }

            # 计算总进球数
            if full_score_raw and ':' in full_score_raw:
                try:
                    home_score, away_score = map(int, full_score_raw.split(':'))
                    match_info['总进球数'] = str(home_score + away_score)
                except (ValueError, TypeError):
                    match_info['总进球数'] = ''

            all_matches_with_results.append(match_info)
        
        if len(matches_on_page) < 100:
            break 
        
        page_no += 1
        time.sleep(1)

    return all_matches_with_results

def fetch_historical_match_details(target_date):
    """获取指定日期的完整历史比赛信息"""
    return fetch_historical_match_results(target_date)

# 兼容旧版本的函数
def analyze_match(home_team, away_team, win_odds, draw_odds, lose_odds, current_date, league_all_name):
    """简化版分析函数，保持向后兼容"""
    match_info = {
        '主队': home_team,
        '客队': away_team,
        '胜赔率': win_odds,
        '平赔率': draw_odds,
        '负赔率': lose_odds,
        '联赛名称': league_all_name,
        'matchId': '0000000'  # 默认ID
    }

    try:
        result = comprehensive_match_analysis(match_info)
        return float(result.get('win_rate', '50'))
    except:
        return 50.0
