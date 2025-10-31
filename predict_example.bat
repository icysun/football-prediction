@echo off
chcp 65001 >nul
echo ⚽ === 足球比赛预测工具 - 快速示例 ===
echo.

echo 选择要运行的预测示例:
echo.
echo 1. 世俱杯: 多特蒙德 vs 蒙特雷
echo 2. 西甲: 皇家马德里 vs 巴塞罗那 (带让球)
echo 3. 英超: 曼联 vs 切尔西
echo 4. 德甲: 拜仁慕尼黑 vs 多特蒙德 (仅ML模型)
echo 5. 意甲: 尤文图斯 vs AC米兰 (仅DL模型)
echo 6. 法甲: 巴黎圣日耳曼 vs 马赛 (JSON输出)
echo 7. 自定义输入
echo 8. 运行测试套件
echo 9. 显示帮助
echo 0. 退出
echo.

set /p choice="请输入选择 (0-9): "

if "%choice%"=="1" (
    echo.
    echo 🚀 运行示例 1: 世俱杯比赛
    python predict_match.py --home_team "多特蒙德" --away_team "蒙特雷" --league "世俱杯" --win_odds 1.57 --draw_odds 3.65 --lose_odds 4.60
    goto end
)

if "%choice%"=="2" (
    echo.
    echo 🚀 运行示例 2: 西甲比赛 (带让球)
    python predict_match.py --home_team "皇家马德里" --away_team "巴塞罗那" --league "西甲" --win_odds 2.10 --draw_odds 3.40 --lose_odds 3.60 --handicap -0.5
    goto end
)

if "%choice%"=="3" (
    echo.
    echo 🚀 运行示例 3: 英超比赛
    python predict_match.py --home_team "曼联" --away_team "切尔西" --league "英超" --win_odds 2.45 --draw_odds 3.20 --lose_odds 2.80
    goto end
)

if "%choice%"=="4" (
    echo.
    echo 🚀 运行示例 4: 德甲比赛 (仅ML模型)
    python predict_match.py --home_team "拜仁慕尼黑" --away_team "多特蒙德" --league "德甲" --win_odds 1.85 --draw_odds 3.80 --lose_odds 4.20 --ml_only
    goto end
)

if "%choice%"=="5" (
    echo.
    echo 🚀 运行示例 5: 意甲比赛 (仅DL模型)
    python predict_match.py --home_team "尤文图斯" --away_team "AC米兰" --league "意甲" --win_odds 2.25 --draw_odds 3.30 --lose_odds 3.10 --dl_only
    goto end
)

if "%choice%"=="6" (
    echo.
    echo 🚀 运行示例 6: 法甲比赛 (JSON输出)
    python predict_match.py --home_team "巴黎圣日耳曼" --away_team "马赛" --league "法甲" --win_odds 1.45 --draw_odds 4.20 --lose_odds 6.50 --output_json
    goto end
)

if "%choice%"=="7" (
    echo.
    echo 📝 自定义输入模式
    echo 请按提示输入比赛信息:
    echo.
    
    set /p home_team="主队名称: "
    set /p away_team="客队名称: "
    set /p league="联赛名称: "
    set /p win_odds="主胜赔率: "
    set /p draw_odds="平局赔率: "
    set /p lose_odds="客胜赔率: "
    set /p handicap="让球数 (可选,默认0): "
    
    if "%handicap%"=="" set handicap=0
    
    echo.
    echo 🚀 运行自定义预测...
    python predict_match.py --home_team "%home_team%" --away_team "%away_team%" --league "%league%" --win_odds %win_odds% --draw_odds %draw_odds% --lose_odds %lose_odds% --handicap %handicap%
    goto end
)

if "%choice%"=="8" (
    echo.
    echo 🧪 运行测试套件
    python test_predictions.py
    goto end
)

if "%choice%"=="9" (
    echo.
    echo 📖 显示帮助信息
    python predict_match.py --help
    goto end
)

if "%choice%"=="0" (
    echo.
    echo 👋 再见!
    goto end
)

echo.
echo ❌ 无效选择，请重新运行脚本
goto end

:end
echo.
echo 按任意键继续...
pause >nul 