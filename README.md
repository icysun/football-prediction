# ⚽ 老段足球预测平台 3.0

一个基于机器学习和深度学习的足球比赛预测系统，集成了实时数据爬取、AI预测分析、模拟投注等功能。

## 🌟 主要功能

### 1. 智能预测系统
- **双模型预测**：集成机器学习（ML）和深度学习（DL）模型，提供更准确的预测结果
- **实时数据更新**：自动爬取最新比赛数据和赔率信息
- **AI综合分析**：结合多维度数据提供专业的比赛分析和预测建议
- **历史准确率统计**：实时跟踪和展示模型预测准确率

### 2. 模拟投注系统
- **虚拟钱包**：每个用户初始获得10,000元虚拟资金
- **实时赔率**：显示每场比赛的真实赔率数据
- **批量投注**：支持多场比赛同时下注
- **自动结算**：比赛结束后自动结算投注结果

### 3. 用户管理系统
- **会员等级**：普通会员、高级会员、VIP会员三个等级
- **权限管理**：不同等级享有不同功能权限
- **个人中心**：查看个人信息、投注记录、收益统计

### 4. 数据分析功能
- **比赛预测导出**：支持CSV格式导出预测结果
- **投注统计分析**：胜率、盈亏、投注习惯等多维度统计
- **历史数据查询**：查看历史比赛结果和预测准确率

## 🛠️ 技术栈

- **后端框架**：Flask 2.3.3
- **数据库**：SQLite3
- **机器学习**：scikit-learn 1.3.0, XGBoost
- **深度学习**：PyTorch 2.0.1
- **数据处理**：pandas 2.0.3, numpy 1.24.3
- **前端技术**：Bootstrap 5, JavaScript, Chart.js
- **数据爬取**：requests, BeautifulSoup

## 📋 系统要求

- Python 3.8+
- Windows/Linux/macOS
- 至少4GB RAM（训练模型时建议8GB+）

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/icysun/football-prediction.git
cd football-prediction
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 初始化数据库
```bash
# 运行数据库迁移
python db_migration.py

# 初始化用户表
python init_users.py

# 初始化投注系统表
python init_betting.py
```

### 4. 准备模型文件
确保以下模型文件存在：
- `football_models.pkl` - 机器学习模型
- `football_dl_model.pth` - 深度学习模型
- `jc_history_api.csv` - 历史数据文件

如果没有模型文件，可以运行训练脚本：
```bash
python retrain_models.py
```

### 5. 启动应用
```bash
python app.py
```

访问 http://localhost:5000 即可使用系统

## 👤 默认账户

系统预设了以下测试账户：

| 用户名 | 密码 | 会员等级 |
|--------|------|----------|
| user1 | 123456 | 普通会员 |
| user2 | 123456 | 高级会员 |
| user3 | 123456 | VIP会员 |

## 📁 项目结构

```
football-prediction/
├── app.py                    # 主应用程序
├── requirements.txt          # 项目依赖
├── predictions.db           # SQLite数据库
├── README.md               # 项目说明文档
│
├── models/                 # 模型相关
│   ├── football_models.pkl      # ML模型文件
│   ├── football_dl_model.pth    # DL模型文件
│   ├── football_ml_predictor.py # ML预测器
│   ├── football_dl_predictor.py # DL预测器
│   └── base_predictor.py        # 预测器基类
│
├── data/                   # 数据相关
│   ├── jc_history_api.csv       # 历史数据
│   ├── football_data_crawler.py # 数据爬虫
│   └── update_historical_data.py # 历史数据更新
│
├── utils/                  # 工具脚本
│   ├── init_users.py           # 用户初始化
│   ├── init_betting.py         # 投注系统初始化
│   ├── db_migration.py         # 数据库迁移
│   ├── retrain_models.py       # 模型重训练
│   └── predict_match.py        # 命令行预测工具
│
├── templates/              # HTML模板
│   ├── index.html             # 首页
│   ├── login.html             # 登录页
│   ├── betting.html           # 投注中心
│   ├── betting_records.html   # 投注记录
│   ├── wallet.html            # 钱包页面
│   ├── profile.html           # 个人中心
│   ├── admin_users.html       # 用户管理
│   └── error.html             # 错误页面
│
└── static/                 # 静态资源
    ├── css/
    │   ├── bootstrap.min.css
    │   ├── bootstrap-icons.css
    │   └── mobile-responsive.css
    ├── js/
    │   └── bootstrap.bundle.min.js
    └── fonts/
        ├── bootstrap-icons.woff
        └── bootstrap-icons.woff2
```

## 🔧 配置说明

### 数据库配置
系统使用SQLite数据库，数据库文件为 `predictions.db`。主要数据表包括：

- `users` - 用户信息表
- `predictions` - 比赛预测表
- `wallets` - 用户钱包表
- `betting_records` - 投注记录表
- `bet_types` - 投注类型配置表

### 模型配置
- ML模型使用XGBoost算法，包含分类、回归等多个子模型
- DL模型使用PyTorch实现的神经网络
- 模型训练数据来源于历史比赛数据

### 自动任务
- **自动更新预测**：每30分钟检查并更新一次预测数据
- **自动结算投注**：每小时检查并结算已完成的比赛投注

## 📊 功能权限说明

| 功能 | 普通会员 | 高级会员 | VIP会员 |
|------|----------|----------|---------|
| 查看预测结果 | ✓ | ✓ | ✓ |
| 模拟投注 | ✓ | ✓ | ✓ |
| 查看投注记录 | ✓ | ✓ | ✓ |
| 更新预测数据 | ✗ | ✓ | ✓ |
| AI综合分析 | ✗ | ✗ | ✓ |
| 手动结算 | ✗ | ✗ | ✓ |
| 导出预测结果 | ✓ | ✓ | ✓ |
| 用户管理 | ✗ | ✗ | ✓ |

## 🔍 命令行工具

### 单场比赛预测
```bash
python predict_match.py --home_team "皇家马德里" --away_team "巴塞罗那" --league "西甲" --win_odds 2.10 --draw_odds 3.40 --lose_odds 3.60
```

### 批量预测示例
```bash
# Windows
predict_example.bat

# Linux/Mac
./predict_example.sh
```

详细使用说明请查看 [PREDICTION_USAGE.md](PREDICTION_USAGE.md)

## 📈 API接口

### 预测相关
- `GET /` - 首页，显示今日预测
- `POST /update_predictions` - 更新预测数据
- `GET /ai_analysis/<match_id>` - 获取AI分析（VIP）

### 投注相关
- `GET /betting` - 投注中心
- `POST /place_bet` - 单笔投注
- `POST /place_bets` - 批量投注
- `GET /betting_records` - 投注记录

### 用户相关
- `POST /login` - 用户登录
- `GET /logout` - 用户登出
- `GET /profile` - 个人中心
- `GET /wallet` - 钱包信息

### 管理相关
- `GET /admin/users` - 用户管理（VIP）
- `POST /settle_bets` - 手动结算（VIP）
- `GET /export_predictions` - 导出预测

## ⚠️ 注意事项

1. **数据更新**：首次运行需要更新历史数据，可能需要较长时间
2. **模型准确率**：预测仅供参考，不构成实际投注建议
3. **虚拟货币**：系统使用的是虚拟货币，仅用于模拟投注
4. **浏览器兼容**：建议使用Chrome、Firefox等现代浏览器

## 🐛 故障排除

### 常见问题

1. **模型文件缺失**
   - 错误：`FileNotFoundError: football_models.pkl not found`
   - 解决：运行 `python retrain_models.py` 重新训练模型

2. **数据库错误**
   - 错误：`sqlite3.OperationalError: no such table`
   - 解决：运行数据库初始化脚本

3. **依赖包问题**
   - 错误：`ModuleNotFoundError`
   - 解决：确保已安装所有依赖 `pip install -r requirements.txt`

## 📝 更新日志

### v3.0.0 (2025-07)
- 集成双模型预测系统（ML+DL）
- 新增模拟投注功能
- 优化移动端响应式设计
- 修复赔率显示问题
- 清理无用代码文件

### v2.0.0 (2025-07)
- 添加AI综合分析功能
- 实现自动结算系统
- 优化数据爬取逻辑

### v1.0.0 (2025-07)
- 初始版本发布
- 基础预测功能
- 用户管理系统

## 📄 许可证

本项目仅供学习和研究使用，不得用于商业用途。

## 🤝 贡献指南

欢迎提交Issue和Pull Request。在提交PR前，请确保：

1. 代码符合PEP8规范
2. 添加必要的注释和文档
3. 通过所有测试用例
4. 更新相关文档

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- Email: icysun@qq.com
- GitHub Issues: https://github.com/icysun/football-prediction/issues

---

**免责声明**：本系统仅供学习和娱乐使用，预测结果不构成任何投注建议。请理性对待预测结果，切勿沉迷赌博。 