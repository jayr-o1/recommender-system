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

## Recent Improvements

### Added Chemistry Field and Specializations

-   New Chemistry field with core skills like Laboratory Techniques, Toxicology, and Analytical Chemistry
-   Added specializations: Analytical Chemist, Toxicologist, and Organic Chemist
-   Expanded skill weights with chemistry-specialized skills

### Improved Fuzzy Matching

-   Reduced default fuzzy threshold from 80 to 70 for better matching of specialized terms
-   Implemented token-based matching for multi-word skills
-   Added bonuses for specialized technical terms in matching algorithm
-   Improved handling of partial matches in specialized domains

### Enhanced Training Process

-   Balanced training data generation across all fields
-   Increased synthetic user dataset size from 10,000 to 15,000
-   Ensured minimum representation for each field in the training data
-   Added weighting for specialized skills during training

### Weighted Confidence Calculation

-   Specialized skills now have 50% more weight in confidence scoring
-   Improved multi-factor confidence calculation considering:
    -   Skill match quality
    -   Match score
    -   Skill importance
    -   Coverage of required skills
-   Better confidence scoring for specialized career paths

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

## Testing the API with Uvicorn

The project now includes a test script that interacts with the API running on Uvicorn. The `test_api.py` script can:

1. Start a Uvicorn server automatically
2. Run tests against the API endpoints
3. Test skill matching capabilities
4. Compare fuzzy vs semantic matching through the API

### Running the API tests

To run the API tests with Uvicorn:

```bash
# Run all tests with default settings
python test_api.py

# Run simple endpoint tests
python test_api.py --simple-test

# Test with custom skills
python test_api.py --skills "Python:90,Data Analysis:85,Machine Learning:75"

# Test with specific parameters
python test_api.py --no-semantic --threshold 70 --top-fields 5 --top-specs 10

# Only run the matching comparison
python test_api.py --compare-only

# Use a custom port
python test_api.py --port 8080
```

### Command-line parameters

The `test_api.py` script supports the following parameters:

-   `--skills`: Comma-separated list of skills with optional proficiency (e.g., "Python:90,Data Analysis:85")
-   `--no-semantic`: Disable semantic matching
-   `--threshold`: Fuzzy matching threshold (0-100)
-   `--top-fields`: Number of top fields to return (1-5)
-   `--top-specs`: Number of top specializations to return (1-10)
-   `--compare-only`: Only run the matching methods comparison
-   `--port`: Port to run the API on (default: 8000)
-   `--host`: Host to run the API on (default: 127.0.0.1)
-   `--no-server`: Don't start the server (assume it's already running)
-   `--simple-test`: Run the simple API tests (test_recommend_direct, test_recommend_nested, test_api_recommend)

### Starting the API with Uvicorn manually

You can also start the API with Uvicorn manually using the provided PowerShell script:

```powershell
# Start the API with development reload enabled
./start_api_uvicorn.ps1
```

Or directly with Python:

```bash
# Development mode with reload
uvicorn src.api:app --reload

# Production mode
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

# SemanticMatcher - Enhanced Skill Matching

SemanticMatcher is an advanced tool for matching and analyzing skills using semantic similarity with transformer-based models. It provides accurate matching for specialized terminology by considering semantic meaning rather than just string similarity.

## New Features

The latest update includes the following improvements:

### Model Management

-   **Model Warmup**: Added option to load and warm up the model on startup for faster initial queries
-   **Model Versioning**: Support for different model sizes (small, medium, large) and switching between models
-   **Custom Models**: Support for custom model paths and configurations
-   **Configuration Serialization**: Save and load model configurations from JSON files

### Performance Enhancements

-   **Batch Processing**: Efficient batch processing for large skill sets
-   **Async Support**: Asynchronous methods for API use cases
-   **Optimized Caching**: Improved caching strategy for embeddings and match results
-   **Progress Tracking**: Configurable progress bars for long-running operations

### Configuration Options

-   **Configurable Thresholds**: Customizable matching thresholds and weights
-   **Domain Customization**: Ability to add or modify domain-specific terminology
-   **Cross-Domain Mapping**: Enhanced cross-domain skill recognition

### Enhanced Features

-   **Skill Clustering**: Group skills based on semantic similarity
-   **Temporal Analysis**: Analyze skill trends over time
-   **Skill Development Path**: Suggest skill development paths based on career goals
-   **Multi-language Support**: Process skills in different languages with translation capabilities

### Testing & Benchmarking

-   **Unit Tests**: Comprehensive test suite for matching logic
-   **Benchmarks**: Performance comparison tools for optimization

## Usage Examples

### Basic Skill Matching

```python
from utils.semantic_matcher import SemanticMatcher

# Match a user skill against reference skills
matched_skill, similarity = SemanticMatcher.match_skill(
    "python programming",
    ["python development", "javascript", "data analysis"]
)
print(f"Matched: {matched_skill}, Similarity: {similarity:.2f}")
```

### Configuring the Model

```python
from utils.semantic_matcher import SemanticMatcher, ModelConfig

# Configure model with custom settings
SemanticMatcher.configure_model(ModelConfig(
    model_name="medium",  # Use a more accurate model
    warmup_on_init=True,  # Warm up model on initialization
    enable_progress_bars=True  # Show progress bars
))
```

### Batch Processing

```python
# Process multiple skills efficiently
skills = ["python", "javascript", "machine learning", "data visualization"]
embeddings = SemanticMatcher.get_embeddings_batch(skills)

# Use embeddings for further processing
for skill, embedding in embeddings.items():
    print(f"Processed: {skill}")
```

### Async Processing

```python
import asyncio

async def process_skills():
    skills = ["python", "javascript", "machine learning"]

    # Process skills asynchronously
    embeddings = await SemanticMatcher.get_embeddings_batch_async(skills)

    # Calculate similarities asynchronously
    similarity = await SemanticMatcher.semantic_similarity_async("python", "python programming")

    return embeddings, similarity

# Run in an async context
results = asyncio.run(process_skills())
```

### Customizing Matching Parameters

```python
# Configure matching parameters
SemanticMatcher.configure_matching({
    "similarity_threshold": 0.7,  # Require higher similarity for matches
    "partial_match_threshold": 0.5,  # Higher threshold for partial matches
    "domain_bonus_cap": 0.4  # Lower the domain bonus cap
})
```

### Skill Clustering

```python
# Cluster skills based on semantic similarity
skills = [
    "python", "javascript", "java", "c++",
    "data analysis", "machine learning", "deep learning",
    "project management", "team leadership"
]

# Auto-determine number of clusters
clusters = SemanticMatcher.cluster_skills(skills)

for cluster_name, cluster_skills in clusters.items():
    print(f"\n{cluster_name}:")
    for skill in cluster_skills:
        print(f"  - {skill}")
```

### Skill Trend Analysis

```python
# Analyze skill trends over time
skill_snapshots = {
    "2021-01": {
        "python": 60,
        "javascript": 50,
        "sql": 40
    },
    "2022-01": {
        "python": 70,
        "javascript": 60,
        "sql": 50,
        "machine learning": 30
    },
    "2023-01": {
        "python": 80,
        "javascript": 70,
        "machine learning": 50,
        "data visualization": 40
    }
}

trends = SemanticMatcher.analyze_skill_trends(skill_snapshots)

print("Emerging Skills:", trends["emerging_skills"])
print("New Skills:", trends["new_skills"])
print("Declining Skills:", trends["declining_skills"])
```

### Multi-language Support

```python
# Translate a skill between languages
translated_skill = SemanticMatcher.translate_skill(
    "machine learning",
    source_lang="en",
    target_lang="es"
)
print(f"Translated: {translated_skill}")  # "aprendizaje automático"

# Detect language of a skill
lang = SemanticMatcher.detect_skill_language("programación")
print(f"Detected language: {lang}")  # "es"
```

## Performance Benchmarks

The performance improvements can be measured using the benchmarking tools:

```bash
python benchmarks/benchmark_semantic_matcher.py
```

This will generate performance metrics and plots comparing:

-   Batch vs. individual processing
-   Effect of model warmup
-   Performance of different matching methods

## Running Tests

Run tests to verify functionality:

```bash
python -m unittest tests/test_semantic_matcher.py
```

## Dependencies

-   numpy
-   sentence-transformers
-   fuzzywuzzy
-   scikit-learn (optional, for advanced clustering)
-   matplotlib (for benchmarking)
-   langdetect (optional, for language detection)

## License

MIT License
