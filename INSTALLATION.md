# Career Recommender System Installation Guide

This document provides instructions for setting up and using the Career Recommender System after moving it from its original codebase.

## Prerequisites

Ensure you have the following installed:

-   Python 3.8 or higher
-   pip (Python package manager)

## Installation Steps

1. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

2. **Verify Data Files**

    Ensure that the `data` directory contains all required files:

    - `specialization_skills.json`
    - `skill_weights.json`
    - `skill_weights_metadata.json`
    - Other synthetic data files for training

3. **Test the Installation**

    Run the test script to verify everything is working correctly:

    ```bash
    python get_career_recommendation.py
    ```

## Usage Examples

### Basic Career Recommendation

```python
from recommender import recommend_career_path

# Example skills
skills = "Python, Java, Machine Learning, Data Science, SQL"

# Get recommendations
recommendations = recommend_career_path(skills)
print(recommendations)
```

### Advanced Career Recommendation with Weighted Skills

```python
from weighted_recommender import WeightedSkillRecommender

# Initialize the recommender
recommender = WeightedSkillRecommender()

# User skills with proficiency levels (0-100)
user_skills = {
    "Python": 90,
    "Machine Learning": 75,
    "Data Analysis": 85,
    "SQL": 80,
    "Java": 60
}

# Get recommendations
results = recommender.recommend(user_skills, top_n=5)
print(results)
```

## System Architecture

The Career Recommender System uses a weighted skill matching approach that accounts for:

-   Skill importance weighting
-   User proficiency levels
-   Partial skill matching using similarity metrics

Key components:

-   `recommender.py` - Core recommendation engine
-   `weighted_recommender.py` - Enhanced recommender with proficiency support
-   `utils/model_trainer.py` - Model training utilities
-   `utils/data_manager.py` - Data management functions
-   `data/` - Configuration and training data

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are installed correctly
2. Verify that all data files are present in the `data` directory
3. Check for any path references that might still point to the old location

For more information, see the main README.md file.
