# Career Recommendation System: Consolidated Workflow

This module provides a consolidated workflow for generating data, training models, and getting career recommendations based on skills and proficiency levels.

## Features

-   **JSON-based data storage**: All data is stored in JSON format for more structured and flexible access
-   **Consolidated data**: All data sources are consolidated into a single unified structure for faster processing
-   **Skill-proficiency support**: Input skills with proficiency levels (0-100%) for more accurate recommendations
-   **Interactive tool**: User-friendly command-line interface for entering skills and getting recommendations
-   **Comprehensive recommendations**: Get field, specialization, top matches, skills to develop, and proficiency improvement suggestions

## Quick Start

### Interactive Tool

For a user-friendly experience, use the interactive tool:

```bash
python get_career_recommendation.py
```

This tool will walk you through entering your skills and proficiency levels, then provide career recommendations.

Options:

-   `--save`: Save your skills to a file for future reference
-   `--file PATH`: Load skills from a file instead of interactive prompt

### Command-line Usage

Use the consolidated workflow script directly for more options:

```bash
python consolidated_workflow.py --recommend --skills "Python 80, Java 60, SQL 75"
```

### Skill-Proficiency Format

You can specify skills with proficiency levels in two formats:

1. **Comma-separated**: `"Python 80, Java 60, SQL 75"`
2. **Multi-line** (in a file):
    ```
    Python 80
    Java 60
    SQL 75
    ```

Each skill is followed by a space and a proficiency level (0-100%).

## Complete Workflow

The consolidated workflow can handle the entire process of generating data, training models, and providing recommendations:

```bash
# Run the complete workflow
python consolidated_workflow.py --run-all

# Generate synthetic data
python consolidated_workflow.py --generate-data

# Consolidate all data sources
python consolidated_workflow.py --consolidate

# Train the model
python consolidated_workflow.py --train-model

# Get recommendations
python consolidated_workflow.py --recommend --skills "Python 80, Machine Learning 75"
```

## Advanced Options

-   `--employee-count`: Number of employee records to generate (default: 1000)
-   `--career-path-count`: Number of career path records to generate (default: 800)
-   `--seed`: Random seed for reproducibility (default: 42)
-   `--replace`: Replace existing data files (default: true)
-   `--quiet`: Suppress detailed output
-   `--skills-file`: File containing skill-proficiency pairs (one pair per line)

## File Locations

-   Generated data: `data/`
-   Consolidated data: `data/consolidated_data.json`
-   Trained model: `models/career_path_recommendation_model.pkl`
-   User skills (if saved): `user_data/`

## Example Usage

Get recommendations for data science skills:

```bash
python consolidated_workflow.py --recommend --skills "Python 85, SQL 90, Machine Learning 75, Statistics 70, Data Analysis 80, TensorFlow 65"
```

Use a skills file:

```bash
python consolidated_workflow.py --recommend --skills-file data/example_skill_proficiency.txt
```

Interactive mode with skill saving:

```bash
python get_career_recommendation.py --save
```

## Recommendations Output

The system provides:

1. **Recommended field and specialization** based on your skills
2. **Top career path matches** with match scores
3. **Skills to develop** for each career path
4. **Proficiency improvement suggestions** for skills you already have
