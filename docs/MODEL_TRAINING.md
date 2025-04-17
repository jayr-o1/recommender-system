# Model Training with Recent Changes

This document explains how to train the career recommendation model with recent changes, including new user preferences and feedback.

## Overview

The career recommender system now includes a functionality that allows you to train the existing model with recent user data. This enables the model to adapt to new user preferences and feedback without requiring a full retraining, which can be computationally expensive.

## Training Methods

There are two main ways to train the model:

1. **Interactive Training** - Using the menu option in the career recommender application
2. **Command-Line Training** - Using the `retrain.py` script

## Interactive Training (Through the Application)

The simplest way to train the model is through the career recommender application interface:

1. Run the career recommender: `python run_recommender.py`
2. Enter your user ID (an admin account is recommended)
3. From the main menu, select option `4. Train model with recent changes (admin)`
4. Follow the prompts to configure the training:
    - Whether to use only user preferences or include feedback data
    - How many days of recent data to consider
    - Minimum number of data points required for training

The application will display training progress and evaluation metrics when complete.

## Command-Line Training

For more control or for scheduled training, you can use the `retrain.py` script:

```bash
# Basic usage (uses defaults)
python retrain.py

# Incremental training with recent changes (default)
python retrain.py --days 30 --min-count 5

# Only use user preferences (ignore feedback)
python retrain.py --prefs-only

# Force training even with insufficient data
python retrain.py --force

# Full retraining (uses all historical data)
python retrain.py --full

# Quiet mode (less output)
python retrain.py --quiet
```

### Command-Line Arguments

-   `--full`: Perform a full retraining rather than just updating with recent changes
-   `--days N`: Only consider data from the last N days (default: 30)
-   `--min-count N`: Minimum number of user data points required (default: 5)
-   `--prefs-only`: Only use user preferences for training (ignore feedback)
-   `--force`: Force retraining even if thresholds are not met
-   `--quiet`: Suppress detailed output

## Scheduled Training

To keep your model up-to-date automatically, you can schedule the training script to run periodically:

### Windows (Task Scheduler)

```
schtasks /create /sc WEEKLY /d MON /tn "TrainCareerRecommenderModel" /tr "python C:\path\to\src\recommender\retrain.py" /st 03:00
```

### Linux/Mac (Cron)

Add to crontab (runs every Monday at 3 AM):

```
0 3 * * 1 python /path/to/src/recommender/retrain.py
```

## Considerations

-   **Data Sufficiency**: For effective training, ensure you have enough user data (preferences and feedback).
-   **Backup**: The system automatically backs up the previous model before applying changes.
-   **Evaluation**: After training, review the model's performance metrics to ensure it has improved.
-   **Frequency**: Regular training with new data helps keep recommendations relevant, but training too frequently with minimal new data can be inefficient.

## Troubleshooting

-   If training fails, check logs for specific error messages.
-   Use `--force` to override minimum data requirements if needed for testing.
-   If performance degrades, you can restore a previous model version from the `models/history/` directory.
-   For persistent issues, consider running a full retraining with `--full` option.
