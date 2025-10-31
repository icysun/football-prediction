import pandas as pd
import os
from datetime import datetime, timedelta, date
from football_data_crawler import fetch_historical_match_details
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

HISTORICAL_CSV = 'jc_history_api.csv'

# 原csv字段顺序
CSV_COLUMNS = [
    '比赛时间','比赛编号','联赛名称','主队','客队','主队（让球）vs客队','半场比分','全场比分','总进球数',
    '胜赔率','平赔率','负赔率'
]

def get_latest_match_date(csv_file):
    if not os.path.exists(csv_file):
        return None
    try:
        # 使用utf-8-sig编码处理BOM字符
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        print(f"CSV columns: {df.columns.tolist()}")  # 调试信息
        print(f"First few rows of 比赛时间: {df['比赛时间'].head(3).tolist()}")  # 调试信息
        
        if '比赛时间' not in df.columns or df.empty:
            return None
        
        # 直接尝试解析比赛时间列
        df['日期'] = pd.to_datetime(df['比赛时间'], errors='coerce')
        # 删除无法解析的日期行
        df = df.dropna(subset=['日期'])
        if df.empty:
            return None
        latest_date = df['日期'].max()
        return latest_date
    except Exception as e:
        logging.warning(f'无法解析比赛时间: {e}')
        return None

def main():
    logging.info('=== 历史数据自动更新脚本启动 ===')
    latest_date = get_latest_match_date(HISTORICAL_CSV)
    today = datetime.now().date()
    if latest_date is not None:
        logging.info(f'历史数据最新比赛时间: {latest_date}')
        if isinstance(latest_date, pd.Timestamp):
            next_date = latest_date + timedelta(days=1)
            start_date = next_date.date()
        elif isinstance(latest_date, datetime):
            next_date = latest_date + timedelta(days=1)
            start_date = next_date.date()
        elif isinstance(latest_date, date):
            start_date = latest_date + timedelta(days=1)
        else:
            raise TypeError(f'未知日期类型: {type(latest_date)}')
    else:
        logging.info('未检测到历史数据，将全量拉取')
        start_date = today - timedelta(days=365*5)  # 默认拉取近5年

    if start_date > today:
        logging.info(f'历史数据已是最新，无需更新。起始日期 {start_date.strftime("%Y-%m-%d")} 晚于今日 {today.strftime("%Y-%m-%d")}')
        logging.info('=== 历史数据自动更新完成 ===')
        return

    date_list = pd.date_range(start=start_date, end=today, freq='D')
    all_new_rows = []
    for d in date_list:
        date_str = d.strftime('%Y-%m-%d')
        matches = fetch_historical_match_details(date_str)
        if isinstance(matches, str):
            logging.warning(f'{date_str} fetch_historical_match_details 失败: {matches}')
            continue
        for match in matches:
            # 检查是否有比赛编号，如果没有则跳过
            if not match.get('比赛编号'):
                continue
            # 确保所有字段都存在，缺失的补空字符串
            row = {}
            for col in CSV_COLUMNS:
                row[col] = match.get(col, '')
            all_new_rows.append(row)
        if matches:
            logging.info(f'{date_str} 新增 {len(matches)} 条完整比赛数据')

    if not all_new_rows:
        logging.info('没有检测到新的比赛数据，无需更新。')
        return

    new_df = pd.DataFrame(all_new_rows)
    
    file_exists = os.path.exists(HISTORICAL_CSV)
    if file_exists:
        try:
            # 读取已有的比赛编号用于去重
            old_ids = pd.read_csv(HISTORICAL_CSV, usecols=['比赛编号'], encoding='utf-8-sig')['比赛编号'].astype(str).tolist()
            new_df = new_df[~new_df['比赛编号'].astype(str).isin(old_ids)]
        except (FileNotFoundError, ValueError, KeyError) as e:
            logging.warning(f"读取旧数据比赛编号失败，可能文件为空或格式不正确: {e}")
            # 文件存在但无法正确读取，当作新文件处理
            file_exists = False
    
    if new_df.empty:
        logging.info('所有新获取的数据已存在，无需更新。')
        return

    # 严格按原字段顺序输出
    for col in CSV_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = ''
    
    # 按照比赛时间排序新数据
    if '比赛时间' in new_df.columns:
        new_df['比赛时间_parsed'] = pd.to_datetime(new_df['比赛时间'], errors='coerce')
        new_df = new_df.sort_values('比赛时间_parsed').drop('比赛时间_parsed', axis=1)

    final_df_to_write = new_df[CSV_COLUMNS]
    
    final_df_to_write.to_csv(HISTORICAL_CSV, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
    
    logging.info(f'成功追加 {len(final_df_to_write)} 条新比赛数据。')
    logging.info('=== 历史数据自动更新完成 ===')

if __name__ == '__main__':
    main() 