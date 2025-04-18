<<<<<<< HEAD
# Model Directory

This directory contains the trained machine learning models and associated metadata files for the career recommendation system.

## Large Model Files

Due to GitHub's file size limitations, the following large model files are not included in the repository:

-   `spec_clf.joblib` (205MB)
-   `field_clf.joblib` (62MB)

These files should be stored separately and can be:

1. Downloaded from our model storage server
2. Generated by running the training scripts
3. Shared via alternative file transfer methods

## Included Files

The following metadata and smaller model files are included in the repository:

-   `feature_names.joblib`: Feature names used by the models
-   `le_spec.joblib`: Label encoder for specializations
-   `le_field.joblib`: Label encoder for fields
-   `model_metadata.json`: Metadata about the models, including version and performance metrics

## Regenerating Models

To regenerate the models, run:

```
python src/train.py
```

This will create the model files in this directory.
=======
# Career Recommendation Model

This directory contains trained models for the Career Recommendation system. These models are trained using synthetic user data based on the field and specialization definitions in the `data` directory.

## Model Files

-   `field_clf.joblib`: Random Forest classifier for predicting career fields
-   `spec_clf.joblib`: Random Forest classifier for predicting specializations
-   `le_field.joblib`: Label encoder for field names
-   `le_spec.joblib`: Label encoder for specialization names
-   `feature_names.joblib`: List of features (skills) used by the models
-   `model_metadata.json`: Metadata about the trained models
>>>>>>> 05a31f53e7cf8827ee9d9d7602031ca22712bd52

## Training Process

The models are trained using synthetic user data generated from the field and specialization definitions. The training process:

1. Loads field, specialization, and skill weight data from the `data` directory
2. Generates 10,000 synthetic user profiles with skills relevant to their specializations
3. Prepares feature vectors and labels for training
4. Trains Random Forest classifiers for field and specialization prediction
5. Evaluates model accuracy on a test set
6. Saves the trained models and metadata

## How to Retrain

To retrain the models:

```bash
python -m src.train
```

This will regenerate the synthetic user data and retrain the models using the current field, specialization, and skill weight data.

## Performance

The trained models have high accuracy:

-   Field classifier: 100% accuracy
-   Specialization classifier: 100% accuracy

These high accuracy rates are expected since the test data is generated from the same distribution as the training data.

## Usage in the API

The models are automatically loaded by the `CareerRecommender` class when it's initialized. If the models are not found, the recommender will fall back to a rule-based approach.

Example API call to get recommendations:

```
POST /recommend
{
  "skills": {
    "Python": 90,
    "JavaScript": 85,
    "Data Analysis": 80,
    "Machine Learning": 75
  }
}
```

This will return field and specialization recommendations based on the provided skills.
