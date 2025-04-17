import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
from typing import Dict, List, Tuple, Any, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_path="data"):
    """
    Load fields, specializations and skill weights from data directory
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Tuple of fields, specializations, and skill weights dictionaries
        
    Raises:
        FileNotFoundError: If required data files are missing
        ValueError: If data files are not valid JSON
    """
    try:
        # Check if required files exist
        required_files = ["fields.json", "specializations.json", "skill_weights.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(data_path, file)):
                raise FileNotFoundError(f"Required data file not found: {file}")
        
        # Load fields
        with open(os.path.join(data_path, "fields.json"), "r") as f:
            fields = json.load(f)
            
        # Load specializations
        with open(os.path.join(data_path, "specializations.json"), "r") as f:
            specializations = json.load(f)
            
        # Load skill weights
        with open(os.path.join(data_path, "skill_weights.json"), "r") as f:
            skill_weights = json.load(f)
            
        # Validate data consistency
        validate_data_consistency(fields, specializations, skill_weights)
            
        return fields, specializations, skill_weights
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in data files: {e}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def validate_data_consistency(fields, specializations, skill_weights):
    """
    Validate consistency between fields, specializations, and skill weights
    
    Args:
        fields: Dictionary of fields
        specializations: Dictionary of specializations
        skill_weights: Dictionary of skill weights
        
    Raises:
        ValueError: If inconsistencies are found
    """
    # Check that all specialization fields exist in fields
    invalid_specs = []
    for spec_name, spec_data in specializations.items():
        if "field" not in spec_data:
            invalid_specs.append(f"{spec_name} (missing field)")
        elif spec_data["field"] not in fields:
            invalid_specs.append(f"{spec_name} (references non-existent field '{spec_data['field']}')")
            
    if invalid_specs:
        raise ValueError(f"Data inconsistency found in specializations: {', '.join(invalid_specs)}")
        
    # Check that all skills in specializations exist in skill_weights
    missing_skills = []
    for spec_name, spec_data in specializations.items():
        if "core_skills" in spec_data:
            for skill in spec_data["core_skills"]:
                if skill not in skill_weights:
                    missing_skills.append(f"{skill} (used in {spec_name})")
                    
    if missing_skills:
        logger.warning(f"Some skills in specializations are not in skill_weights: {', '.join(missing_skills)}")

def generate_synthetic_users(fields, specializations, skill_weights, num_users=15000):
    """
    Generate synthetic user data for training the recommender
    with balanced representation across all fields
    
    Args:
        fields: Dictionary of fields
        specializations: Dictionary of specializations  
        skill_weights: Dictionary of skill weights
        num_users: Number of synthetic users to generate
        
    Returns:
        List of synthetic user profiles
    """
    users = []
    
    # Get field names and specializations per field
    field_names = list(fields.keys())
    specs_by_field = {}
    for field in field_names:
        specs_by_field[field] = [
            spec_name for spec_name, spec_data in specializations.items()
            if spec_data.get("field") == field
        ]
    
    all_skills = list(skill_weights.keys())
    
    # Calculate users per field to ensure balanced representation
    min_users_per_field = num_users // len(field_names)
    
    # Track specialized skills
    specialized_terms = {"laboratory", "techniques", "toxicology", "analytical", 
                      "chemistry", "chemical", "organic", "synthesis", "spectroscopy",
                      "biochemistry", "instrumentation"}
    
    specialized_skills = []
    for skill in all_skills:
        if any(term in skill.lower() for term in specialized_terms):
            specialized_skills.append(skill)
    
    logger.info(f"Generating {num_users} synthetic users with {len(all_skills)} possible skills")
    logger.info(f"Ensuring at least {min_users_per_field} users per field for balanced representation")
    logger.info(f"Identified {len(specialized_skills)} specialized skills that will receive higher weight")
    
    # Generate users for each field to ensure balance
    field_user_count = {field: 0 for field in field_names}
    
    while sum(field_user_count.values()) < num_users:
        # Choose a field - prioritize underrepresented fields
        remaining_users = num_users - sum(field_user_count.values())
        underrepresented_fields = [f for f in field_names if field_user_count[f] < min_users_per_field]
        
        if underrepresented_fields:
            field_name = np.random.choice(underrepresented_fields)
        else:
            # After minimum quota met, randomly assign remaining users
            field_name = np.random.choice(field_names)
        
        # Get available specializations for this field
        field_specs = specs_by_field[field_name]
        if not field_specs:
            continue  # Skip fields with no specializations
            
        # Select a random specialization from this field
        spec_name = np.random.choice(field_specs)
        spec_data = specializations[spec_name]
        
        # Generate skills
        user_skills = {}
        
        # Add all core skills for the specialization with high proficiency (70-100)
        core_skills = spec_data.get("core_skills", {})
        for skill in core_skills:
            if skill in all_skills:  # Ensure skill exists in all_skills
                # Increase the probability of higher proficiency for specialized skills
                if any(term in skill.lower() for term in specialized_terms):
                    user_skills[skill] = np.random.randint(80, 101)  # 80-100 for specialized skills
                else:
                    user_skills[skill] = np.random.randint(70, 101)  # 70-100 for other skills
        
        # Add some random skills from the same field with medium proficiency (40-80)
        field_skills = set()
        for s_name, s_data in specializations.items():
            if s_data.get("field") == field_name and s_name != spec_name:
                field_skills.update(s_data.get("core_skills", {}).keys())
                
        # Select a subset of field skills
        num_field_skills = min(np.random.randint(2, 6), len(field_skills))
        selected_field_skills = np.random.choice(list(field_skills), num_field_skills, replace=False)
        
        for skill in selected_field_skills:
            if skill in all_skills and skill not in user_skills:  # Avoid duplicates
                user_skills[skill] = np.random.randint(40, 81)
        
        # Add some random skills with low proficiency (10-60)
        remaining_skills = [s for s in all_skills if s not in user_skills]
        num_random_skills = min(np.random.randint(3, 10), len(remaining_skills))
        selected_random_skills = np.random.choice(remaining_skills, num_random_skills, replace=False)
        
        for skill in selected_random_skills:
            user_skills[skill] = np.random.randint(10, 61)
        
        # Create user profile
        user = {
            "specialization": spec_name,
            "field": field_name,
            "skills": user_skills
        }
        
        users.append(user)
        field_user_count[field_name] += 1
    
    # Log field distribution
    logger.info("Final user distribution by field:")
    for field, count in field_user_count.items():
        logger.info(f"- {field}: {count} users ({count/num_users*100:.1f}%)")
    
    return users

def prepare_data(users, skill_weights):
    """
    Prepare feature matrix and labels from user data
    
    Args:
        users: List of user profiles
        skill_weights: Dictionary of skill weights
        
    Returns:
        Tuple of feature matrix, field labels, specialization labels, 
        label encoders, and feature names
    """
    # Get all possible skills
    all_skills = list(skill_weights.keys())
    
    # Prepare feature matrix and labels
    X = []
    y_spec = []
    y_field = []
    
    logger.info(f"Preparing data for {len(users)} users with {len(all_skills)} features")
    
    for user in users:
        # Create skill vector
        skill_vector = np.zeros(len(all_skills))
        
        for i, skill in enumerate(all_skills):
            skill_vector[i] = user["skills"].get(skill, 0) / 100.0  # Normalize to 0-1
        
        X.append(skill_vector)
        y_spec.append(user["specialization"])
        y_field.append(user["field"])
    
    # Convert to numpy arrays
    X = np.array(X)
    
    # Encode labels
    le_field = LabelEncoder()
    le_spec = LabelEncoder()
    
    y_field_encoded = le_field.fit_transform(y_field)
    y_spec_encoded = le_spec.fit_transform(y_spec)
    
    logger.info(f"Prepared data with {X.shape[0]} samples and {X.shape[1]} features")
    logger.info(f"Number of unique fields: {len(le_field.classes_)}")
    logger.info(f"Number of unique specializations: {len(le_spec.classes_)}")
    
    return X, y_field_encoded, y_spec_encoded, le_field, le_spec, all_skills

def train_models(X, y_field, y_spec):
    """
    Train Random Forest models for field and specialization prediction
    
    Args:
        X: Feature matrix
        y_field: Field labels
        y_spec: Specialization labels
        
    Returns:
        Tuple of trained field classifier and specialization classifier
    """
    # Split data
    X_train, X_test, y_field_train, y_field_test, y_spec_train, y_spec_test = train_test_split(
        X, y_field, y_spec, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Field classifier
    field_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    field_clf.fit(X_train, y_field_train)
    field_accuracy = field_clf.score(X_test, y_field_test)
    logger.info(f"Field classifier accuracy: {field_accuracy:.2f}")
    
    # Specialization classifier
    spec_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    spec_clf.fit(X_train, y_spec_train)
    spec_accuracy = spec_clf.score(X_test, y_spec_test)
    logger.info(f"Specialization classifier accuracy: {spec_accuracy:.2f}")
    
    return field_clf, spec_clf

def save_models(field_clf, spec_clf, le_field, le_spec, feature_names, model_dir="model"):
    """
    Save trained models and encoders to disk
    
    Args:
        field_clf: Trained field classifier
        spec_clf: Trained specialization classifier
        le_field: Label encoder for fields
        le_spec: Label encoder for specializations
        feature_names: List of feature names
        model_dir: Directory to save models
    """
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models and encoders
        joblib.dump(field_clf, os.path.join(model_dir, "field_clf.joblib"))
        joblib.dump(spec_clf, os.path.join(model_dir, "spec_clf.joblib"))
        joblib.dump(le_field, os.path.join(model_dir, "le_field.joblib"))
        joblib.dump(le_spec, os.path.join(model_dir, "le_spec.joblib"))
        joblib.dump(feature_names, os.path.join(model_dir, "feature_names.joblib"))
        
        # Save metadata for reference
        metadata = {
            "num_features": len(feature_names),
            "fields": list(le_field.classes_),
            "specializations": list(le_spec.classes_),
            "created_at": str(np.datetime64('now'))
        }
        
        with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Models and metadata saved to {model_dir}")
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Loading data...")
        fields, specializations, skill_weights = load_data()
        
        logger.info("Generating synthetic users for training...")
        users = generate_synthetic_users(fields, specializations, skill_weights, num_users=15000)
        logger.info(f"Generated {len(users)} synthetic user profiles")
        
        logger.info("Preparing data for training...")
        X, y_field, y_spec, le_field, le_spec, feature_names = prepare_data(users, skill_weights)
        
        logger.info("Training models...")
        field_clf, spec_clf = train_models(X, y_field, y_spec)
        
        logger.info("Saving models...")
        save_models(field_clf, spec_clf, le_field, le_spec, feature_names)
        
        logger.info("Training complete!")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise 