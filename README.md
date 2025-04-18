# Career Recommender System

An AI-powered recommendation engine that matches users to optimal career paths based on their skills and proficiency levels.

## Overview

The Career Recommender System uses machine learning to analyze user skills and suggest the most suitable career fields and specializations. It provides detailed confidence scores, identifies skill gaps, and offers personalized development recommendations.

### Key Features

-   **AI-Powered Matching**: ML models trained on extensive career data to provide accurate recommendations
-   **Multi-Level Analysis**: Recommends both broad career fields and specific specializations
-   **Skill Gap Analysis**: Identifies missing skills needed for career advancement
-   **Confidence Scoring**: Calculates detailed confidence metrics for each recommendation
-   **Flexible Interfaces**: Access via API, CLI, or containerized deployment
-   **Semantic Matching**: Advanced NLP for understanding skill relationships and synonyms

## Getting Started

### Prerequisites

-   Python 3.9+
-   Required packages (see `requirements.txt`)
-   Docker (optional, for containerized deployment)

### Quick Installation

```bash
# Clone repository
git clone <repository_url>
cd career-recommender

# Install dependencies
pip install -r requirements.txt

# Train the model (optional - pre-trained models included)
python src/train.py

# Start the API server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d
```

## Usage Examples

### REST API

Access the API at http://localhost:8000 after starting the server.

```bash
# Get career recommendations
curl -X POST \
  http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "skills": {
      "Python": 90,
      "SQL": 80,
      "Data Analysis": 85,
      "Machine Learning": 80,
      "Excel": 75
    }
  }'
```

### Command Line Interface

```bash
# Get recommendations via CLI
python src/cli.py --skills "Python 90, SQL 80, Data Analysis 85, Machine Learning 80, Excel 75"

# Output in JSON format
python src/cli.py --skills "Python 90, SQL 80" --json
```

## System Architecture

### Components

1. **Data Layer**

    - Field definitions with core skills
    - Specialization definitions with required skills
    - Skill weight and importance mappings

2. **ML Models**

    - Field classifier (Random Forest, 100% accuracy)
    - Specialization classifier (Random Forest, 99-100% accuracy)

3. **Matching Engine**

    - Fuzzy matching for misspelled skills
    - Semantic matching for conceptually similar skills
    - Confidence scoring algorithm with multi-factor analysis

4. **Interfaces**
    - RESTful API (FastAPI)
    - Command-line interface
    - Docker containerization

### How It Works

1. **Input Processing**: User skills are parsed and normalized
2. **Skill Matching**: User skills are matched to known skills using fuzzy and semantic matching
3. **Field Prediction**: ML models predict suitable career fields
4. **Specialization Matching**: System identifies matching specializations within recommended fields
5. **Confidence Calculation**: Multi-factor analysis generates confidence scores
6. **Skill Gap Analysis**: Missing skills are identified and prioritized
7. **Results Formatting**: Recommendations are packaged and returned

## Data Structure

### Skills

Skills have the following properties:

-   Name: The skill name (e.g., "Python", "Data Analysis")
-   Proficiency: User's proficiency level from 1-100

### Fields

Fields are broad career areas with associated core skills.

Example:

```json
{
    "Data Science": {
        "description": "Interdisciplinary field that uses scientific methods to extract knowledge from data",
        "core_skills": {
            "Statistics": 85,
            "Mathematics": 75,
            "Programming": 80,
            "Data Analysis": 90,
            "Machine Learning": 80
        }
    }
}
```

### Specializations

Specializations are specific roles within fields, with their own set of core skills and weights.

Example:

```json
{
    "Data Analyst": {
        "field": "Data Science",
        "description": "Collects, processes, and performs statistical analysis on large datasets",
        "core_skills": {
            "Python": 80,
            "SQL": 90,
            "Data Analysis": 95,
            "Data Visualization": 85,
            "Statistics": 80,
            "Excel": 75
        }
    }
}
```

## Supported Fields & Specializations

The system currently supports recommendations for the following fields:

-   **Computer Science**: Software Engineer, Web Developer, Mobile App Developer, DevOps, Cloud Architect
-   **Data Science**: Data Scientist, Data Engineer, Machine Learning Engineer, Business Intelligence Analyst
-   **Cybersecurity**: Security Analyst, Penetration Tester, Information Security Manager
-   **Law**: Lawyer, Legal Consultant, Patent Attorney
-   **Finance**: Financial Analyst, Investment Banker, Portfolio Manager
-   **Chemistry**: Analytical Chemist, Toxicologist, Organic Chemist
-   **Healthcare**: Physician, Nurse Practitioner, Clinical Specialist
-   **And many more fields with 65+ specializations**

## Project Structure

-   `data/`: Contains JSON files defining career fields, specializations, and skill weights
    -   `fields.json`: Definitions of career fields with core skills
    -   `specializations.json`: Definitions of specializations within fields
    -   `skill_weights.json`: Weights and importance of different skills
-   `model/`: Contains trained machine learning models
-   `src/`: Source code
    -   `train.py`: Script to train recommendation models
    -   `recommender.py`: Career recommendation engine
    -   `api.py`: FastAPI web API for serving recommendations
    -   `cli.py`: Command-line interface
-   `tests/`: Testing framework
    -   `test_recommendation_system.py`: Comprehensive test suite
-   `utils/`: Utility modules
    -   `semantic_matcher.py`: Advanced skill matching using NLP

## Testing Framework

The system includes a comprehensive testing framework:

### Automated Test Suite

```bash
# Run all recommendation system tests
pytest tests/test_recommendation_system.py

# Run tests for a specific field/specialization
pytest tests/test_recommendation_system.py::test_case_1_data_scientist_profile
```

### Test Coverage

-   **Field Recognition Tests**: Verify field identification accuracy
-   **Specialization Mapping**: Ensure specializations are correctly matched
-   **Edge Cases**: Test minimal skills, misspelled inputs, irrelevant skills
-   **Field-Specific Tests**: Dedicated test suites for each major field (Data Science, Law, etc.)

### Test Results Analysis

Each test generates a detailed JSON report with:

-   Input skills and proficiency levels
-   Matched fields with confidence scores
-   Matched specializations with confidence scores
-   Matched skills with match scores
-   Missing skills with priority levels

## Model Performance

The trained models have high accuracy:

-   Field classifier: 100% accuracy
-   Specialization classifier: 99-100% accuracy

In the latest test suite:

-   Data Science field was correctly identified with 95-100% confidence for profiles with relevant skills
-   Data Engineer specialization matched with 92% confidence for data engineering skill sets
-   Machine Learning Engineer specialization achieved 69% confidence for specialized ML skills
-   Business Intelligence Analyst specialization correctly matched BI-specific skills

## Recent Improvements

### Enhanced Data Science Recommendations

-   Improved confidence scoring for Data Science specializations
-   Enhanced matching for specialized Machine Learning roles
-   Better handling of technical Data Science terminology
-   Added comprehensive test suite for all Data Science specializations

### Chemistry Field Integration

-   Added new Chemistry field with specialized roles (Analytical Chemist, Toxicologist, Organic Chemist)
-   Expanded skill weights for chemistry domain

### Matching Algorithm Enhancements

-   Reduced fuzzy threshold from 80 to 70 for better specialized term matching
-   Implemented token-based matching for multi-word skills
-   Added bonuses for technical terminology
-   Improved handling of partial matches in specialized domains

### Training Process Improvements

-   Balanced data generation across all fields
-   Increased synthetic dataset to 50,000 profiles
-   Ensured minimum representation for each field in the training data
-   Added weighting for specialized skills during training

### Weighted Confidence Calculation

-   Specialized skills now have 50% more weight in confidence scoring
-   Improved multi-factor confidence calculation considering:
    -   Skill match quality
    -   Match score
    -   Skill importance
    -   Coverage of required skills

## Advanced Feature: SemanticMatcher

The system includes a powerful semantic matching engine:

```python
from utils.semantic_matcher import SemanticMatcher

# Match user skills against reference skills
matched_skill, similarity = SemanticMatcher.match_skill(
    "python programming",
    ["python development", "javascript", "data analysis"]
)
```

### SemanticMatcher Features

-   **Model Selection**: Multiple pre-trained models (small, medium, large)
-   **Batch Processing**: Efficient handling of large skill sets
-   **Async Support**: Asynchronous processing for API use cases
-   **Skill Clustering**: Group related skills semantically
-   **Multi-language Support**: Process skills in different languages

### Configuration Examples

```python
from utils.semantic_matcher import SemanticMatcher, ModelConfig

# Configure model with custom settings
SemanticMatcher.configure_model(ModelConfig(
    model_name="medium",  # Use a more accurate model
    warmup_on_init=True,  # Warm up model on initialization
    enable_progress_bars=True  # Show progress bars
))

# Configure matching parameters
SemanticMatcher.configure_matching({
    "similarity_threshold": 0.7,  # Require higher similarity for matches
    "partial_match_threshold": 0.5,  # Higher threshold for partial matches
    "domain_bonus_cap": 0.4  # Lower the domain bonus cap
})
```

## Testing the API with Uvicorn

The project includes a test script that interacts with the API running on Uvicorn:

```bash
# Run all tests with default settings
python test_api.py

# Run simple endpoint tests
python test_api.py --simple-test

# Test with custom skills
python test_api.py --skills "Python:90,Data Analysis:85,Machine Learning:75"

# Test with specific parameters
python test_api.py --no-semantic --threshold 70 --top-fields 5 --top-specs 10
```

### Command-line parameters

-   `--skills`: Comma-separated list of skills with optional proficiency
-   `--no-semantic`: Disable semantic matching
-   `--threshold`: Fuzzy matching threshold (0-100)
-   `--top-fields`: Number of top fields to return (1-5)
-   `--top-specs`: Number of top specializations to return (1-10)
-   `--compare-only`: Only run the matching methods comparison
-   `--port`: Port to run the API on (default: 8000)

## Extending the System

To add new fields, specializations, or skills:

1. Edit the JSON files in the `data/` directory
2. Retrain the models using `python -m src.train`

## Dependencies

-   numpy
-   scikit-learn
-   fastapi
-   uvicorn
-   sentence-transformers
-   fuzzywuzzy
-   pytest

## License

MIT License
