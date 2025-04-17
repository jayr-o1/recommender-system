# Career Path Recommender System

A standalone machine learning-based system for recommending career paths based on a user's skills and experience.

## Overview

This system helps users identify suitable career paths based on their skill sets. It uses a weighted matching algorithm that considers:

-   Skill importance within each specialization
-   User proficiency levels for each skill
-   Partial skill matching using similarity metrics
-   Field and specialization recommendations with confidence scores

## Features

-   Predicts the most suitable career field and specialization based on user skills
-   Provides confidence scores for each recommendation
-   Identifies missing skills for recommended specializations
-   Offers multiple specialization recommendations with confidence scores
-   Specialization-specific skill recommendations tailored to each role
-   Accounts for skill proficiency levels in recommendations

## Installation

See [INSTALLATION.md](INSTALLATION.md) for setup instructions.

## Quick Start

```python
from recommender import recommend_career_path

# Example skills
skills = "Python, Java, Machine Learning, Data Science, SQL"

# Get recommendations
recommendations = recommend_career_path(skills)

print(recommendations)
```

## Advanced Usage with Weighted Recommendations

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

## Directory Structure

```
career-recommender/
├── data/                      # Data files
│   ├── specialization_skills.json      # Specialization-specific skills
│   ├── skill_weights.json              # Skill importance weights
│   └── skill_weights_metadata.json     # Metadata for skill weights
│
├── models/                    # Machine learning models
│   └── trained_models.pkl        # Trained models (if applicable)
│
├── utils/                     # Utility modules
│   ├── model_trainer.py         # Model training utilities
│   ├── data_manager.py          # Data management functions
│   └── feedback_handler.py      # Feedback processing
│
├── README.md                  # This file
├── INSTALLATION.md            # Installation guide
├── recommender.py             # Core recommendation engine
├── weighted_recommender.py    # Enhanced recommender with proficiency support
├── test_recommender.py        # Test script for verification
└── requirements.txt           # Required Python packages
```

## Testing

Run the test script to verify that everything is working correctly:

```bash
python test_recommender.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

-   This system uses synthetic training data for machine learning models
-   Recommender functionality is based on skill weighting and proficiency analysis
