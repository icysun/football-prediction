# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a football (soccer) match prediction platform that combines machine learning (ML) and deep learning (DL) models with real-time data crawling from Chinese sports lottery APIs. The system provides match predictions, virtual betting simulation, and AI-powered comprehensive analysis.

### Key Components

1. **Dual Model Prediction System**
   - ML Model (`football_ml_predictor.py`): XGBoost-based classification and regression models
   - DL Model (`football_dl_predictor.py`): PyTorch neural network with multi-task learning
   - Both models predict: win/draw/lose outcome, goal counts, and exact scores

2. **Data Crawling** (`football_data_crawler.py`)
   - Fetches live match data from `webapi.sporttery.cn`
   - Retrieves odds changes, head-to-head records, standings, injuries, etc.
   - AI analysis via Tencent Hunyuan API

3. **Flask Web Application** (`app.py`)
   - User authentication with membership tiers (Basic/VIP/Premium)
   - Virtual betting system with wallet management
   - Auto-settlement background thread
   - Prediction display and export

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python db_migration.py
python init_users.py
python init_betting.py
```

### Model Training
```bash
# Train/retrain ML model
python -c "from football_ml_predictor import FootballPredictor; p = FootballPredictor(); p.load_data('jc_history_api.csv'); p.preprocess_data(); p.feature_engineering(); p.train_models(n_trials=50); p.save_models('football_models.pkl')"

# Train/retrain DL model
python -c "from football_dl_predictor import FootballDLPredictor; p = FootballDLPredictor(); p.load_data('jc_history_api.csv'); p.preprocess_data(); p.feature_engineering(); p.train_model(batch_size=64, epochs=100); p.save_model('football_dl_model.pth')"

# Or use the retrain script
python retrain_models.py
```

### Running the Application
```bash
# Start Flask server
python app.py
# Runs on http://127.0.0.1:80

# Single match prediction (CLI)
python predict_match.py --home_team "皇家马德里" --away_team "巴塞罗那" --league "西甲" --win_odds 2.10 --draw_odds 3.40 --lose_odds 3.60
```

### Database Operations
```bash
# SQLite database location
predictions.db  # Main database file

# Key tables: users, predictions, wallets, betting_records, user_membership, app_config
```

## Architecture Notes

### Prediction Flow
1. `fetch_match_data()` → Fetches today's matches from API
2. `make_predictions()` → Runs ML/DL models on match data
3. Results stored in `predictions` table
4. AI analysis available on-demand via `/ai_analysis/<match_id>` (VIP only)

### Model Features
Both predictors use these engineered features:
- Team encodings (LabelEncoder for home/away teams and leagues)
- Recent form stats (win rate, avg goals, last 5 matches)
- Odds-derived features (implied probabilities, odds differences)
- Handicap values extracted from match titles

### Data Splitting Strategy
Models use **time-based splitting** (not random): first 80% of historical data for training, last 20% for testing. This simulates real-world "past predicting future" scenarios.

### Membership System
- Level 1 (Basic): View ML/DL predictions only
- Level 2 (VIP): Full predictions + can trigger manual updates
- Level 3 (Premium): Everything + AI comprehensive analysis

### Background Tasks
- Auto-update predictions: Every 30 minutes (threaded)
- Auto-settlement: Every hour (checks finished matches, updates results, settles bets)

## File Locations

- Main app: `chengxu-final1/app.py` or `chengxu-wu/app.py`
- ML predictor: `football_ml_predictor.py`
- DL predictor: `football_dl_predictor.py`
- Base predictor: `base_predictor.py`
- Data crawler: `football_data_crawler.py`
- Historical data: `jc_history_api.csv`
- Model files: `football_models.pkl`, `football_dl_model.pth`

## Important Constraints

1. **Chinese API**: Data source uses Chinese field names (`比赛编号`, `主队`, `客队`, etc.)
2. **Result Mapping**: ML/DL models output 'win'/'draw'/'lose', database stores '胜'/'平'/'负'
3. **Model State**: Models must be loaded before app starts (`init_predictors()`)
4. **Database**: SQLite with `row_factory` for dict-like access
5. **Unknown Teams**: Predictors handle unseen teams via string similarity matching

## API Endpoints Reference

- `GET /` - Main prediction page
- `POST /update_predictions` - Manual prediction update (VIP+)
- `GET /ai_analysis/<match_id>` - AI comprehensive analysis (Premium)
- `POST /place_bets` - Batch betting
- `GET /betting_records` - User betting history
- `POST /settle_bets` - Manual settlement (Premium)
- `GET /export_predictions` - CSV export of predictions
