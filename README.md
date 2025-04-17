# Career Recommender System

A machine learning-based system that recommends career fields and specializations based on a user's skills and proficiency levels.

## Features

-   **Skill-Based Recommendations**: Match skills and proficiency levels to career fields and specializations
-   **Confidence Scoring**: Calculate confidence scores for how well user skills match different careers
-   **Missing Skills Identification**: Identify skills that users should develop for specific careers
-   **Multiple Interface Options**:
    -   RESTful API
    -   Command Line Interface
    -   Docker Deployment

## Getting Started

### Prerequisites

-   Python 3.9+
-   Docker (optional, for containerized deployment)

### Installation

#### Option 1: Local Installation

1. Clone the repository:

    ```bash
    git clone <repository_url>
    cd career-recommender
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Generate sample data:

    ```bash
    python src/data_generator.py
    ```

4. Run the API server:
    ```bash
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
    ```

#### Option 2: Docker Deployment

1. Clone the repository:

    ```bash
    git clone <repository_url>
    cd career-recommender
    ```

2. Build and run with Docker Compose:
    ```bash
    docker-compose up -d
    ```

## Usage

### API Endpoints

Once the server is running, access the API at: http://localhost:8000

Available endpoints:

-   `GET /`: API information
-   `GET /fields`: List all available career fields
-   `GET /specializations`: List all available specializations
-   `POST /recommend`: Get career recommendations based on skills

#### Example API Request

```bash
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

The CLI provides a quick way to get recommendations:

```bash
python src/cli.py --skills "Python 90, SQL 80, Data Analysis 85, Machine Learning 80, Excel 75"
```

Options:

-   `--skills`: Comma-separated list of skills with proficiency (required)
-   `--top-fields`: Number of top fields to display (default: 1)
-   `--top-specializations`: Number of top specializations to display (default: 3)
-   `--json`: Output results in JSON format

## Data Structure

### Skills

Skills have the following properties:

-   Name: The skill name (e.g. "Python", "Data Analysis")
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
-   `test_model.py`: Simple script to test the recommendation models
-   `compare_profiles.py`: Script to compare recommendations for different skill profiles

## How It Works

1. **Data Modeling**: The system uses JSON files to define career fields, specializations, and skills with their importance weights.

2. **Model Training**: The training process:

    - Loads field, specialization, and skill weight data
    - Generates 10,000 synthetic user profiles with skills relevant to their specializations
    - Trains Random Forest classifiers for field and specialization prediction
    - Achieves high accuracy (100%) on the test set

3. **Recommendation Engine**: Based on a user's skills and proficiency levels, the system:

    - Predicts the most suitable career fields
    - Recommends specializations within those fields
    - Provides confidence scores for each recommendation
    - Identifies matching and missing skills for each recommendation

4. **API**: The system exposes a FastAPI-based web API to:
    - Get career recommendations based on skills
    - Browse available fields and specializations

## Usage

1. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2. **Train the Models** (optional, pre-trained models included):

    ```bash
    python -m src.train
    ```

3. **Test the Models**:

    ```bash
    python test_model.py
    ```

4. **Compare Different Profiles**:

    ```bash
    python compare_profiles.py
    ```

5. **Start the API Server**:

    ```bash
    python -m src.api
    ```

6. **Make Recommendation Requests**:
    ```bash
    curl -X POST "http://localhost:8000/recommend" \
      -H "Content-Type: application/json" \
      -d '{"skills": {"Python": 90, "JavaScript": 85, "Data Analysis": 80}}'
    ```

## Extending the System

To add new fields, specializations, or skills:

1. Edit the JSON files in the `data/` directory
2. Retrain the models using `python -m src.train`

## Model Performance

The trained models have high accuracy:

-   Field classifier: 100% accuracy
-   Specialization classifier: 100% accuracy

These high accuracy rates are expected since the test data is generated from the same distribution as the training data.

## Customization

You can customize the recommendation system by:

1. Modifying the data files in the `data/`
