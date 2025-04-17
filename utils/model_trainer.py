#!/usr/bin/env python3
"""
Model trainer utilities for career recommender.
This module provides utility functions for loading and processing skill data
for use in career recommendation models.
"""

import os
import sys
import traceback
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from collections import Counter, defaultdict
import json
import pickle
import shutil
import difflib
import logging

# Import utilities
from .data_loader import load_synthetic_employee_data, load_synthetic_career_path_data

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
MODEL_PATH = "models/career_path_recommendation_model.pkl"
EMPLOYEE_DATA_PATH = "data/synthetic_employee_data.json"
CAREER_PATH_DATA_PATH = "data/synthetic_career_path_data.json"
MODEL_HISTORY_DIR = "models/history"

# Adjust paths for when running from different directories
def get_adjusted_path(path):
    """Check if path exists, if not try to find it relative to the script location."""
    if os.path.exists(path):
        return path
    
    # Try relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    adjusted = os.path.join(base_dir, path)
    if os.path.exists(adjusted):
        return adjusted
    
    # Try without data/ prefix
    filename = os.path.basename(path)
    adjusted = os.path.join(base_dir, "data", filename)
    if os.path.exists(adjusted):
        return adjusted
    
    # Could not find the file, return original path
    return path

def report_progress(message, percent=None):
    """Report progress for long-running operations."""
    # In a real implementation, this might update a progress bar
    # or send a notification to a user interface
    if percent is not None:
        print(f"{message} - {percent}% complete")
    else:
        print(message)

def ensure_directory_exists(directory_path):
    """Ensure that the directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        if not os.path.exists(directory_path):
            raise RuntimeError(f"Failed to create directory: {directory_path}")
    return directory_path

def save_model_components(model_components, verbose=False):
    """
    Save model components to disk, with backup.
    
    Args:
        model_components (dict): Dictionary containing model components.
        verbose (bool): If True, print progress messages.
        
    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        # First, ensure the directories exist
        ensure_directory_exists(os.path.dirname(MODEL_PATH))
        ensure_directory_exists(MODEL_HISTORY_DIR)
        
        # Update the timestamp if not already set
        if 'trained_at' not in model_components:
            model_components['trained_at'] = datetime.now().isoformat()
            
        # Create a backup of the existing model if it exists
        if os.path.exists(MODEL_PATH):
            # Get current timestamp for the backup filename
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
            backup_path = os.path.join(MODEL_HISTORY_DIR, f"career_path_recommendation_model_backup_{timestamp}.pkl")
            
            # Copy the existing model to the backup location
            shutil.copy2(MODEL_PATH, backup_path)
            
            if verbose:
                print(f"Created backup of existing model at {backup_path}")
                
            # Also save metadata about the model for easier tracking
            metadata = {
                "original_model": MODEL_PATH,
                "backup_path": backup_path,
                "backup_time": timestamp,
                "trained_at": model_components.get('trained_at', 'Unknown'),
                "accuracy": model_components.get('accuracy', 0),
                "feedback_entries_used": model_components.get('feedback_entries_used', 0)
            }
            
            metadata_path = os.path.join(MODEL_HISTORY_DIR, "..", "model_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            if verbose:
                print(f"Saved model metadata to {metadata_path}")
        
        # Save the model components
        joblib.dump(model_components, MODEL_PATH)
        
        if verbose:
            print(f"Saved model components to {MODEL_PATH}")
            
        return True
    except Exception as e:
        if verbose:
            print(f"Error saving model components: {str(e)}")
            traceback.print_exc()
        return False

def load_career_fields():
    """
    Load career fields dictionary.
    This function no longer tries to import from recommender.py and directly uses its own dictionary.
    """
    # Return a minimal default dictionary
    return {
        "Computer Science": {
            "roles": [
                "Software Engineer", "Data Scientist", "Machine Learning Engineer",
                "Cybersecurity Analyst", "Cloud Architect", "UI/UX Designer", 
                "AI Research Scientist", "Full-Stack Developer"
            ],
            "skills": [
                "Python", "Java", "C++", "JavaScript", "SQL", 
                "Machine Learning", "Cloud Computing", "Web Development"
            ]
        },
        "Marketing": {
            "roles": [
                "Digital Marketing Specialist", "Brand Manager", "Market Research Analyst",
                "Content Strategist", "Social Media Manager"
            ],
            "skills": [
                "Social Media Marketing", "SEO", "Content Marketing",
                "Data Analytics", "Brand Management", "Market Research"
            ]
        },
        "Finance": {
            "roles": [
                "Investment Banker", "Portfolio Manager", "Risk Manager",
                "Financial Advisor"
            ],
            "skills": [
                "Financial Modeling", "Valuation", "Risk Management",
                "Market Analysis", "Financial Research"
            ]
        },
        "Law": {
            "roles": [
                "Corporate Lawyer", "Litigation Attorney", "Intellectual Property Lawyer",
                "Criminal Defense Attorney", "Family Law Attorney", "Immigration Lawyer",
                "Real Estate Attorney", "Tax Attorney", "Employment Law Attorney"
            ],
            "skills": [
                "Legal Research", "Legal Writing", "Contract Drafting", "Contract Negotiation",
                "Due Diligence", "Trial Advocacy", "Litigation", "Case Strategy",
                "Client Representation", "Regulatory Compliance", "Legal Analysis"
            ]
        }
    }

def prepare_features(data, target):
    """
    Prepare features for training with advanced text preprocessing
    """
    # Find the skills column - could be 'Skills' (employee data) or 'Required Skills' (career path data)
    skills_column = None
    if 'Skills' in data.columns:
        skills_column = 'Skills'
    elif 'Required Skills' in data.columns:
        skills_column = 'Required Skills'
    else:
        raise ValueError(f"No skills column found in data. Available columns: {data.columns.tolist()}")
    
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in data. Available columns: {data.columns.tolist()}")
    
    # Extract features and target
    X = data[skills_column].fillna('')
    y = data[target].fillna('')
    
    # Remove empty targets and skills
    valid = (y != '') & (X != '')
    return X[valid], y[valid]

def identify_popular_specializations(data, threshold=5):
    """
    Identify popular specializations that have enough training examples.
    
    Args:
        data: DataFrame containing training data
        threshold: Minimum number of examples needed to be considered popular
        
    Returns:
        dict: Dictionary mapping fields to their popular specializations
    """
    # Group by field and specialization to count occurrences
    counts = data.groupby(['Field', 'Specialization']).size().reset_index(name='count')
    
    # Filter to specializations that have at least threshold examples
    popular = counts[counts['count'] >= threshold]
    
    # Create a dictionary mapping fields to their popular specializations
    result = {}
    for field, group in popular.groupby('Field'):
        result[field] = group['Specialization'].tolist()
        
    return result

def filter_to_popular_specializations(data, popular_specs=None, min_examples=5):
    """
    Filter dataset to include only popular specializations with sufficient training examples.
    
    Args:
        data: DataFrame containing training data
        popular_specs: Optional pre-defined dictionary of popular specializations
        min_examples: Minimum examples needed if popular_specs not provided
        
    Returns:
        DataFrame: Filtered dataset with only popular specializations
    """
    if popular_specs is None:
        popular_specs = identify_popular_specializations(data, min_examples)
    
    # If no popular specializations were found or all are empty, return original data
    if not popular_specs or all(len(specs) == 0 for specs in popular_specs.values()):
        print(f"No popular specializations found with threshold {min_examples}. Using all data.")
        return data
    
    # Create a mask for rows to keep
    mask = pd.Series(False, index=data.index)
    
    # Include rows where the field+specialization combination is in our popular list
    for field, specializations in popular_specs.items():
        if specializations:  # Only if we have specializations for this field
            field_mask = (data['Field'] == field) & (data['Specialization'].isin(specializations))
            mask = mask | field_mask
    
    # Filter the data
    filtered_data = data[mask].copy()
    
    # If filtering resulted in too little data, return the original
    if len(filtered_data) < 50 or len(filtered_data) < 0.1 * len(data):
        print(f"Filtered data too small ({len(filtered_data)} rows). Using all {len(data)} rows instead.")
        return data
        
    print(f"Filtered from {len(data)} to {len(filtered_data)} examples (focusing on popular specializations)")
    return filtered_data

def train_enhanced_model(X, y, verbose=False):
    """
    Train a model with enhanced feature extraction and parameter tuning
    
    Args:
        X: Series or array of skill strings
        y: Series or array of target labels
        verbose: Whether to print progress messages
    """
    if verbose:
        print("Starting enhanced model training...")
    
    # Create TF-IDF vectorizer with improved parameters
    tfidf = TfidfVectorizer(
        max_features=150,       # Increased features
        min_df=2,               # Minimum document frequency
        max_df=0.9,             # Maximum document frequency
        ngram_range=(1, 2),     # Include bigrams
        stop_words='english'    # Remove common English stop words
    )
    X_tfidf = tfidf.fit_transform(X)
    
    if verbose:
        print(f"Extracted {X_tfidf.shape[1]} features from skills text")
    
    # Choose appropriate n_estimators based on dataset size
    n_trees = min(200, max(100, int(len(X) / 10)))
    
    # Create RandomForest with improved parameters
    model = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42
    )
    
    if verbose:
        print(f"Training RandomForest with {n_trees} trees...")
    
    # Train the model
    model.fit(X_tfidf, y)
    
    # Get training accuracy
    y_pred = model.predict(X_tfidf)
    accuracy = accuracy_score(y, y_pred)
    
    if verbose:
        print(f"Model training completed with accuracy: {accuracy:.4f}")
        
        # Show class distribution
        class_counts = Counter(y)
        total = sum(class_counts.values())
        print("\nClass distribution:")
        for cls, count in class_counts.most_common():
            percentage = (count / total) * 100
            print(f"  {cls}: {count} examples ({percentage:.1f}%)")
    
    return model, tfidf, accuracy

def cross_validate_model(X, y, folds=5):
    """
    Perform cross-validation to evaluate model quality
    """
    # Create a pipeline with all preprocessing steps
    tfidf = TfidfVectorizer(max_features=150, ngram_range=(1, 2))
    
    # Use simple model for cross-validation to save time
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    # Convert y to numpy array if it's a pandas Series
    if isinstance(y, pd.Series):
        y = y.values
    
    # Perform stratified k-fold cross-validation
    try:
        cv = StratifiedKFold(n_splits=min(folds, len(set(y))), shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        return scores
    except Exception as e:
        print(f"Cross-validation error: {e}")
        return [0.0]  # Return dummy score on error

def predict_field(skills_str, components):
    """
    Predict field based on skills.
    
    Args:
        skills_str (str): Comma-separated list of skills
        components (dict): Model components
        
    Returns:
        dict: Prediction results with field and confidence
    """
    try:
        # Extract model components
        field_model = components.get('field_model')
        field_vectorizer = components.get('field_vectorizer')
        
        if field_model is None or field_vectorizer is None:
            # Missing required components
            print("Missing field model components")
            return {
                'field': "Computer Science",  # Default to Computer Science instead of generic "Technology"
                'confidence': 0.3,
                'error': "Missing model components"
            }
        
        # Convert skills to features
        X = field_vectorizer.transform([skills_str])
        
        # Make prediction
        field = field_model.predict(X)[0]
        
        # Get prediction probabilities
        proba = field_model.predict_proba(X)[0]
        confidence = max(proba)
        
        # If confidence is very low, try to use field mapping from specialization skills
        # This helps when we have domain-specific skills that might not be in training data
        if confidence < 0.4 or field == "Other":
            # Try to find matches in specialization skills
            specialization_skills = components.get('specialization_skills', {})
            if specialization_skills:
                skills_list = [s.strip() for s in skills_str.split(',')]
                field_match_scores = {}
                
                # Check each field's specializations for skill matches
                for field_name, specializations in specialization_skills.items():
                    field_match_scores[field_name] = 0
                    for spec_name, spec_skills in specializations.items():
                        # Count how many skills match
                        matched_skills = set(skills_list).intersection(set(spec_skills))
                        if matched_skills:
                            # Add more weight to exact field-specific skill matches
                            field_match_scores[field_name] += len(matched_skills) * 2
                
                # If we found any matches, use the field with the most matches
                if field_match_scores and max(field_match_scores.values()) > 0:
                    best_field = max(field_match_scores.items(), key=lambda x: x[1])[0]
                    # If our best field has significantly more matches than the predicted field
                    # or the predicted field is "Other", override
                    if field == "Other" or field_match_scores.get(field, 0) < field_match_scores[best_field] * 0.7:
                        field = best_field
                        confidence = max(0.5, confidence)  # Boost confidence but not above original if high
                        print(f"Overrode field prediction to {field} based on skill matching")
        
        # Get alternate fields as the top 3 predicted fields
        predicted_classes = field_model.classes_
        sorted_indices = np.argsort(proba)[::-1]  # Sort in descending order
        alternate_fields = [predicted_classes[i] for i in sorted_indices[1:4]]  # Next 3 fields after the top one
        
        return {
            'field': field,
            'confidence': float(confidence),
            'alternate_fields': alternate_fields
        }
    except Exception as e:
        # Fallback to default
        print(f"Error in field prediction: {str(e)}")
        return {
            'field': "Computer Science",  # Change default fallback field
            'confidence': 0.2,
            'error': str(e)
        }

def predict_specialization(skills_str, field, components):
    """
    Predict specialization within a field based on skills.
    
    Args:
        skills_str (str): Comma-separated list of skills
        field (str): The field to predict specialization within
        components (dict): Model components
        
    Returns:
        dict: Prediction results with specialization and confidence
    """
    try:
        # First, check if we have a dedicated specialization matcher
        if 'specialization_matcher' in components and 'specialization_vectorizer' in components:
            try:
                # This is the new specialization matcher trained specifically on specialization_skills.json
                model = components['specialization_matcher']
                vectorizer = components['specialization_vectorizer']
                
                # Transform input skills
                X = vectorizer.transform([skills_str])
                
                # Get prediction and probabilities
                prediction = model.predict(X)[0]
                probas = model.predict_proba(X)[0]
                
                # Get class indices sorted by probability
                classes = model.classes_
                class_probas = dict(zip(classes, probas))
                
                # Get top specializations
                top_specs = []
                for spec, prob in sorted(class_probas.items(), key=lambda x: x[1], reverse=True)[:5]:
                    top_specs.append({
                        'specialization': spec,
                        'confidence': float(prob)
                    })
                
                return {
                    'specialization': prediction,
                    'confidence': float(class_probas[prediction]),
                    'top_specializations': top_specs
                }
            except Exception as e:
                print(f"Error using specialization matcher: {str(e)}")
                # Fall back to traditional method below
                pass
                
        # Traditional field-specific specialization prediction
        # Get the specialization encoder and model for this field
        if field not in components.get('specialization_encoders', {}):
            # If this field doesn't have a specific encoder, try to find the closest match
            field_options = list(components.get('specialization_encoders', {}).keys())
            if not field_options:
                return {
                    'specialization': f"{field} Specialist",
                    'confidence': 0.5
                }
            
            # Find closest matching field
            field = find_closest_match(field, field_options)
        
        # Get the encoder and model
        encoder = components['specialization_encoders'].get(field)
        model = components['specialization_models'].get(field)
        vectorizer = components['specialization_vectorizers'].get(field)
        
        if not encoder or not model or not vectorizer:
            return {
                'specialization': f"{field} Specialist",
                'confidence': 0.5
            }
        
        # Transform input
        X = vectorizer.transform([skills_str])
        
        # Get prediction
        prediction_idx = model.predict(X)[0]
        
        # Get probabilities
        probas = model.predict_proba(X)[0]
        confidence = probas[prediction_idx]
        
        # Decode specialization name
        if hasattr(encoder, 'inverse_transform'):
            specialization = encoder.inverse_transform([prediction_idx])[0]
        else:
            # Fallback for LabelEncoder
            specialization = encoder.classes_[prediction_idx]
        
        # Get alternate specializations
        alternate_specializations = []
        if hasattr(encoder, 'classes_'):
            # Get top N predictions
            top_n = 3
            top_indices = probas.argsort()[-top_n:][::-1]
            
            for idx in top_indices:
                if idx != prediction_idx:  # Skip the primary prediction
                    alt_spec = encoder.classes_[idx]
                    alt_confidence = probas[idx]
                    alternate_specializations.append({
                        'specialization': alt_spec,
                        'confidence': float(alt_confidence)
                    })
        
        # Return full prediction results
        return {
            'specialization': specialization,
            'confidence': float(confidence),
            'alternate_specializations': alternate_specializations
        }
    except Exception as e:
        # If any error occurs, return a default response
        print(f"Error predicting specialization: {str(e)}")
        return {
            'specialization': f"{field} Specialist",
            'confidence': 0.5
        }

def load_specialization_skills(path=None):
    """
    Load specialization skills mapping from JSON file.
    
    Args:
        path: Path to specialization skills JSON file (optional)
        
    Returns:
        dict: Mapping of specialization names to required skills
    """
    if not path:
        # Use default path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, "data", "specialization_skills.json")
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded specialization skills from {path}")
        
        # Check if it's the expected format
        if isinstance(data, dict):
            return data
        
        # Try alternate formats
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Convert list of dicts to dict of specialization -> skills
            result = {}
            for item in data:
                if 'specialization' in item and 'skills' in item:
                    result[item['specialization']] = item['skills']
                elif 'title' in item and 'required_skills' in item:
                    result[item['title']] = item['required_skills']
            
            logger.info(f"Converted list format to specialization mapping")
            return result
        
        logger.warning(f"Unexpected data format in {path}")
        return {}
        
    except Exception as e:
        logger.error(f"Error loading specialization skills: {str(e)}")
        # Return fallback data for testing
        return _get_fallback_specializations()

def load_skill_weights(path=None):
    """
    Load skill weight data for specialization matching.
    
    Args:
        path (str): Path to skill weights JSON file
        
    Returns:
        dict: Mapping of specializations to skill weights
    """
    try:
        if path is None:
            # Try multiple potential file locations
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            potential_paths = [
                os.path.join(base_dir, "data", "skill_weights.json"),
                os.path.join(base_dir, "skill_weights.json"),
                "skill_weights.json"
            ]
            
            # Try each path until we find one that exists
            for potential_path in potential_paths:
                if os.path.exists(potential_path):
                    path = potential_path
                    break
        
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                weights = json.load(f)
                
            # Check if we need to apply dynamic adjustments based on trends
            trend_adjustment_file = os.path.join(os.path.dirname(path), "skill_weights_metadata.json")
            if os.path.exists(trend_adjustment_file):
                try:
                    with open(trend_adjustment_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Apply trend-based adjustments if available
                    if 'trends' in metadata:
                        weights = _apply_trend_adjustments(weights, metadata['trends'])
                        
                    # Apply recency adjustments to skills if last_updated is available
                    if 'last_updated' in metadata:
                        weights = _apply_recency_adjustments(weights, metadata['last_updated'])
                        
                except Exception as e:
                    logger.warning(f"Error applying trend adjustments: {str(e)}")
            
            return weights
        else:
            # If no file exists, return fallback weights
            logger.warning(f"Skill weights file not found at {path if path else 'any location'}. Using fallback weights.")
            return _generate_fallback_weights(_get_fallback_specializations())
    except Exception as e:
        logger.warning(f"Error loading skill weights: {str(e)}")
        return _generate_fallback_weights(_get_fallback_specializations())

def _apply_trend_adjustments(weights, trends):
    """
    Apply trend-based adjustments to skill weights.
    
    Args:
        weights (dict): Original skill weights
        trends (dict): Trend data from metadata
        
    Returns:
        dict: Adjusted skill weights
    """
    adjusted_weights = {spec: dict(spec_weights) for spec, spec_weights in weights.items()}
    
    # Process trending skills (boost weights)
    for skill, boost in trends.get('trending_skills', {}).items():
        # Find all specializations that have this skill
        for spec in adjusted_weights:
            if skill in adjusted_weights[spec]:
                # Boost the weight, but cap at 1.0
                current_weight = adjusted_weights[spec][skill]
                adjusted_weights[spec][skill] = min(1.0, current_weight * (1.0 + boost))
    
    # Process declining skills (reduce weights)
    for skill, reduction in trends.get('declining_skills', {}).items():
        # Find all specializations that have this skill
        for spec in adjusted_weights:
            if skill in adjusted_weights[spec]:
                # Reduce the weight, but keep it above 0.1
                current_weight = adjusted_weights[spec][skill]
                adjusted_weights[spec][skill] = max(0.1, current_weight * (1.0 - reduction))
    
    # Add emerging skills if not already present
    for skill, initial_weight in trends.get('emerging_skills', {}).items():
        # Find relevant specializations for this emerging skill
        for spec in adjusted_weights:
            # Use skill relationship or category mapping to decide if this emerging skill
            # should be added to a specialization
            if _should_add_emerging_skill(skill, spec, adjusted_weights[spec]):
                if skill not in adjusted_weights[spec]:
                    adjusted_weights[spec][skill] = initial_weight
    
    return adjusted_weights

def _should_add_emerging_skill(skill, specialization, existing_skills):
    """
    Determine if an emerging skill should be added to a specialization.
    
    Args:
        skill (str): The emerging skill
        specialization (str): The specialization
        existing_skills (dict): Existing skills for the specialization
        
    Returns:
        bool: True if the skill should be added
    """
    # Map of tech domains and related skills
    domain_skills = {
        'web_development': ['JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Vue', 'Web Development'],
        'data_science': ['Python', 'R', 'Data Analysis', 'Machine Learning', 'Statistics', 'Data Visualization'],
        'cloud': ['AWS', 'Azure', 'GCP', 'Cloud Computing', 'Containerization', 'DevOps'],
        'mobile': ['Android', 'iOS', 'Swift', 'Kotlin', 'React Native', 'Flutter'],
        'ai': ['Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision', 'TensorFlow', 'PyTorch'],
        'security': ['Cybersecurity', 'Network Security', 'Penetration Testing', 'Security Compliance'],
        'devops': ['CI/CD', 'Docker', 'Kubernetes', 'Jenkins', 'GitLab CI', 'DevOps'],
        'database': ['SQL', 'NoSQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'Database Design']
    }
    
    # Emerging skill mappings to domains
    emerging_skill_domains = {
        'WebAssembly': ['web_development'],
        'JAMstack': ['web_development'],
        'Next.js': ['web_development'],
        'Svelte': ['web_development'],
        'AutoML': ['data_science', 'ai'],
        'MLOps': ['data_science', 'ai', 'devops'],
        'GitOps': ['devops', 'cloud'],
        'FinOps': ['cloud'],
        'Serverless': ['cloud'],
        'Edge Computing': ['cloud'],
        'Blockchain': ['security'],
        'Zero Trust': ['security'],
        'Low-Code': ['web_development'],
        'Generative AI': ['ai'],
        'LLMs': ['ai', 'data_science'],
        'Vector Databases': ['database', 'ai'],
        'AR/VR Development': ['mobile'],
        'Quantum Computing': ['ai'],
        'GPT Integration': ['ai', 'web_development'],
        'Prompt Engineering': ['ai']
    }
    
    # Check if the emerging skill belongs to a domain relevant to this specialization
    skill_domains = emerging_skill_domains.get(skill, [])
    
    # Count how many skills from each domain are in the specialization
    domain_counts = {}
    for domain, domain_skill_list in domain_skills.items():
        count = sum(1 for s in domain_skill_list if s in existing_skills)
        domain_counts[domain] = count
        
    # Check if any of the skill's domains have a significant presence in the specialization
    for domain in skill_domains:
        if domain in domain_counts and domain_counts[domain] >= 2:
            return True
    
    # Special cases based on specialization name
    specialization_lower = specialization.lower()
    if 'web' in specialization_lower and 'web_development' in skill_domains:
        return True
    if 'data' in specialization_lower and ('data_science' in skill_domains or 'database' in skill_domains):
        return True
    if 'cloud' in specialization_lower and 'cloud' in skill_domains:
        return True
    if 'mobile' in specialization_lower and 'mobile' in skill_domains:
        return True
    if 'ai' in specialization_lower or 'intelligence' in specialization_lower and 'ai' in skill_domains:
        return True
    if 'security' in specialization_lower and 'security' in skill_domains:
        return True
    if 'devops' in specialization_lower and 'devops' in skill_domains:
        return True
    if 'database' in specialization_lower and 'database' in skill_domains:
        return True
    
    return False

def _apply_recency_adjustments(weights, last_updated):
    """
    Apply recency-based adjustments to skill weights.
    
    Args:
        weights (dict): Original skill weights
        last_updated (dict): Last updated timestamps for skills
        
    Returns:
        dict: Adjusted skill weights
    """
    adjusted_weights = {spec: dict(spec_weights) for spec, spec_weights in weights.items()}
    
    # Get current date for comparison
    current_date = datetime.now()
    
    for spec in adjusted_weights:
        for skill in list(adjusted_weights[spec].keys()):
            # Check if we have recency data for this skill
            if skill in last_updated:
                try:
                    # Parse the timestamp
                    last_updated_date = datetime.fromisoformat(last_updated[skill])
                    
                    # Calculate days since last update
                    days_since_update = (current_date - last_updated_date).days
                    
                    # Apply decay for older skills (more than 180 days without update)
                    if days_since_update > 180:
                        # Calculate decay factor (from 1.0 down to 0.7 for very old skills)
                        decay_factor = max(0.7, 1.0 - ((days_since_update - 180) / 1000))
                        adjusted_weights[spec][skill] *= decay_factor
                except:
                    # Skip if date parsing fails
                    pass
    
    return adjusted_weights

def save_model_data(data, filename, folder=None):
    """
    Save model data to JSON file.
    
    Args:
        data: Data to save
        filename: Name of the file to save to
        folder: Optional folder path
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not folder:
        # Use default path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder = os.path.join(base_dir, "data")
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception as e:
            logger.error(f"Could not create folder {folder}: {str(e)}")
            return False
    
    # Save data
    path = os.path.join(folder, filename)
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved model data to {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model data: {str(e)}")
        return False

def preprocess_skills(skills):
    """
    Preprocess skills to standardize format and remove variations.
    
    Args:
        skills: List of skill names or dict of skill:proficiency pairs
        
    Returns:
        dict: Processed skills with standardized names as keys
    """
    # Standardize input format
    if isinstance(skills, list):
        skill_dict = {skill: 70 for skill in skills}  # Default proficiency
    elif isinstance(skills, dict):
        skill_dict = skills.copy()
    else:
        return {}
    
    processed = {}
    
    # Process each skill
    for skill, proficiency in skill_dict.items():
        # Standardize case and whitespace
        std_skill = skill.lower().strip()
        
        # Handle common abbreviations and aliases
        aliases = {
            "js": "javascript",
            "ts": "typescript",
            "py": "python",
            "ui/ux": "ui design",
            "react": "react.js",
            "node": "node.js",
            "vue": "vue.js",
            "postgres": "postgresql",
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "oop": "object oriented programming"
        }
        
        if std_skill in aliases:
            std_skill = aliases[std_skill]
        
        # Add to processed dict (use higher proficiency if duplicated)
        if std_skill in processed:
            processed[std_skill] = max(processed[std_skill], proficiency)
        else:
            processed[std_skill] = proficiency
    
    return processed

def extract_top_skills_by_field(specializations_skills, top_n=10):
    """
    Extract top skills by field from specialization skills.
    
    Args:
        specializations_skills: Mapping of specialization to required skills
        top_n: Number of top skills to extract per field
        
    Returns:
        dict: Mapping of field to top skills
    """
    # Initialize counters for skills by field
    field_skills = defaultdict(lambda: defaultdict(int))
    
    # Map specializations to fields
    field_map = {
        "Software Developer": "Computer Science",
        "Web Developer": "Computer Science",
        "Mobile Developer": "Computer Science",
        "Data Scientist": "Computer Science",
        "Machine Learning Engineer": "Computer Science",
        "DevOps Engineer": "Computer Science",
        "Cloud Engineer": "Computer Science",
        "Security Engineer": "Computer Science",
        "Project Manager": "Business",
        "Product Manager": "Business",
        "Business Analyst": "Business",
        "Financial Analyst": "Finance",
        "Investment Banker": "Finance",
        "Accountant": "Finance",
        "Marketing Specialist": "Marketing",
        "Digital Marketer": "Marketing",
        "Content Marketer": "Marketing",
        "UI Designer": "Design",
        "UX Designer": "Design",
        "Graphic Designer": "Design",
        "HR Specialist": "Human Resources",
        "Recruiter": "Human Resources",
        "Talent Manager": "Human Resources"
    }
    
    # Count occurrences of skills in each field
    for specialization, skills in specializations_skills.items():
        # Determine field
        field = field_map.get(specialization, "Other")
        
        # Count skills
        for skill in skills:
            field_skills[field][skill] += 1
    
    # Get top skills for each field
    top_skills_by_field = {}
    for field, skill_counts in field_skills.items():
        # Sort by count (descending)
        sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N skills
        top_skills_by_field[field] = [skill for skill, count in sorted_skills[:top_n]]
    
    return top_skills_by_field

def _get_fallback_specializations():
    """
    Provide fallback specialization skills for testing.
    
    Returns:
        dict: Mapping of specialization names to required skills
    """
    return {
        "Software Developer": [
            "Python", "JavaScript", "Java", "C#", "C++", "Git", 
            "Data Structures", "Algorithms", "Problem Solving", 
            "Object Oriented Programming", "Software Architecture"
        ],
        "Web Developer": [
            "HTML", "CSS", "JavaScript", "React.js", "Angular", "Vue.js", 
            "Node.js", "PHP", "REST API", "Web Security", "Responsive Design"
        ],
        "Data Scientist": [
            "Python", "R", "SQL", "Statistics", "Machine Learning", 
            "Data Visualization", "Data Cleaning", "Big Data", 
            "Natural Language Processing", "Deep Learning"
        ],
        "UX Designer": [
            "User Research", "Wireframing", "Prototyping", "UI Design", 
            "Usability Testing", "Figma", "Adobe XD", "Information Architecture", 
            "Visual Design", "Design Thinking"
        ],
        "Product Manager": [
            "Product Strategy", "Market Research", "User Stories", 
            "Roadmapping", "Agile", "Stakeholder Management", 
            "Competitive Analysis", "Prioritization", "Data Analysis", "Leadership"
        ],
        "Digital Marketer": [
            "SEO", "SEM", "Content Marketing", "Social Media Marketing", 
            "Email Marketing", "Analytics", "CRM", "A/B Testing", 
            "Campaign Management", "Brand Strategy"
        ]
    }

def _generate_fallback_weights(specializations):
    """
    Generate fallback skill weights based on specializations.
    
    Args:
        specializations: Mapping of specialization names to required skills
        
    Returns:
        dict: Mapping of specialization names to skill weights
    """
    weights = {}
    
    for spec, skills in specializations.items():
        skill_weights = {}
        
        # Assign weights to skills
        for i, skill in enumerate(skills):
            # First few skills get higher weights
            if i < 3:
                weight = 0.9  # Critical skills
            elif i < 6:
                weight = 0.7  # Important skills
            else:
                weight = 0.5  # Useful skills
                
            skill_weights[skill] = weight
            
        weights[spec] = skill_weights
    
    return weights

def are_skills_related(skill1, skill2):
    """
    Determine if two skills are related to each other for fuzzy matching.
    
    Args:
        skill1: First skill
        skill2: Second skill
        
    Returns:
        bool: True if skills are related, False otherwise
    """
    # Normalize skills
    s1 = skill1.lower().replace("-", " ").replace(".", " ")
    s2 = skill2.lower().replace("-", " ").replace(".", " ")
    
    # Exact match
    if s1 == s2:
        return True
    
    # One is substring of the other
    if s1 in s2 or s2 in s1:
        return True
    
    # Fuzzy matching using difflib
    similarity = difflib.SequenceMatcher(None, s1, s2).ratio()
    
    # High similarity threshold
    if similarity > 0.8:
        return True
    
    # Check for compound skills
    if (s1 in ["javascript", "js"] and s2 in ["javascript", "js"]) or \
       (s1 in ["typescript", "ts"] and s2 in ["typescript", "ts"]) or \
       (s1 in ["python", "py"] and s2 in ["python", "py"]) or \
       (s1 in ["java", "jvm"] and s2 in ["java", "jvm"]) or \
       (s1 in ["ruby", "rb"] and s2 in ["ruby", "rb"]) or \
       (s1 in ["csharp", "c#", "dotnet", ".net"] and s2 in ["csharp", "c#", "dotnet", ".net"]) or \
       (s1 in ["c++", "cpp"] and s2 in ["c++", "cpp"]) or \
       (s1 in ["react", "react.js", "reactjs"] and s2 in ["react", "react.js", "reactjs"]) or \
       (s1 in ["vue", "vue.js", "vuejs"] and s2 in ["vue", "vue.js", "vuejs"]) or \
       (s1 in ["angular", "angular.js", "angularjs"] and s2 in ["angular", "angular.js", "angularjs"]) or \
       (s1 in ["node", "node.js", "nodejs"] and s2 in ["node", "node.js", "nodejs"]) or \
       (s1 in ["ml", "machine learning"] and s2 in ["ml", "machine learning"]) or \
       (s1 in ["ai", "artificial intelligence"] and s2 in ["ai", "artificial intelligence"]) or \
       (s1 in ["nlp", "natural language processing"] and s2 in ["nlp", "natural language processing"]):
        return True
        
    return False

def calculate_skill_match_percentage(user_skills_str, specialization, specialization_skills, skill_weights=None, proficiency_levels=None):
    """
    Calculate how well a user's skills match a specific specialization.
    
    Args:
        user_skills_str (str or dict): User's skills as comma-separated string or proficiency dict
        specialization (str): Target specialization to match against
        specialization_skills (dict): Dictionary mapping specializations to required skills
        skill_weights (dict, optional): Dictionary of skill importance weights
        proficiency_levels (dict, optional): Dictionary mapping skills to proficiency levels
        
    Returns:
        dict: Dictionary with match metrics and skill lists
    """
    # Parse user skills from string if needed
    if isinstance(user_skills_str, str):
        user_skills = [skill.strip() for skill in user_skills_str.split(',') if skill.strip()]
        user_skills_proficiency = {skill: 70 for skill in user_skills}  # Default proficiency
    else:
        # Assume it's already a dictionary of skills with proficiency
        user_skills_proficiency = user_skills_str
        user_skills = list(user_skills_proficiency.keys())
    
    # Apply any provided proficiency levels
    if proficiency_levels:
        for skill, level in proficiency_levels.items():
            if skill in user_skills_proficiency:
                user_skills_proficiency[skill] = level
    
    # Get required skills for this specialization
    required_skills = specialization_skills.get(specialization, [])
    if not required_skills:
        # Try case-insensitive match
        for spec, skills in specialization_skills.items():
            if spec.lower() == specialization.lower():
                required_skills = skills
                break
    
    if not required_skills:
        return {
            'match_percentage': 0.0,
            'skill_coverage': 0.0,
            'proficiency_score': 0.0,
            'missing_skills': [],
            'matched_skills': [],
            'partially_matched_skills': []
        }
    
    # Get weights for this specialization
    spec_weights = {}
    if skill_weights and specialization in skill_weights:
        spec_weights = skill_weights[specialization]
    
    # Calculate match
    matched_skills = []
    partially_matched = []
    missing_skills = []
    total_weight = 0.0
    matched_weight = 0.0
    
    for req_skill in required_skills:
        # Get weight for this skill (default to 0.5 if not specified)
        skill_weight = spec_weights.get(req_skill, 0.5) 
        total_weight += skill_weight
        
        # Check for exact match
        if req_skill in user_skills:
            matched_skills.append(req_skill)
            # Apply proficiency as a factor (0.0-1.0)
            proficiency_factor = user_skills_proficiency.get(req_skill, 70) / 100.0
            matched_weight += skill_weight * proficiency_factor
            continue
        
        # Check for fuzzy/similar matches
        matched = False
        for user_skill in user_skills:
            if are_skills_related(req_skill, user_skill):
                partially_matched.append(req_skill)
                # For partial matches, apply partial weight and proficiency
                proficiency_factor = user_skills_proficiency.get(user_skill, 70) / 100.0
                matched_weight += skill_weight * 0.7 * proficiency_factor  # 70% match for related skills
                matched = True
                break
        
        # If not matched, add to missing skills
        if not matched:
            missing_skills.append(req_skill)
    
    # Calculate match percentage (weighted by importance)
    match_percentage = (matched_weight / total_weight * 100) if total_weight > 0 else 0.0
    
    # Calculate simple skill coverage (percentage of skills covered)
    skill_coverage = ((len(matched_skills) + len(partially_matched)) / len(required_skills) * 100) if required_skills else 0.0
    
    # Calculate average proficiency score for matched skills
    proficiency_values = [user_skills_proficiency.get(skill, 70) for skill in matched_skills]
    proficiency_score = sum(proficiency_values) / len(proficiency_values) if proficiency_values else 0.0
    
    return {
        'match_percentage': match_percentage,
        'skill_coverage': skill_coverage, 
        'proficiency_score': proficiency_score,
        'missing_skills': missing_skills,
        'matched_skills': matched_skills,
        'partially_matched_skills': partially_matched
    }

def identify_missing_skills(user_skills, target_specialization, model_components):
    """
    Identify missing skills for a target specialization, with special handling for legal specializations.
    
    Args:
        user_skills (str or list): Comma-separated skills string or list of skills
        target_specialization (str): Target specialization to analyze
        model_components (dict): Dictionary of model components
        
    Returns:
        tuple: (missing_skills, match_percentage, matched_skills)
    """
    try:
        # Load specialization skills
        specialization_skills = model_components.get('specialization_skills')
        if not specialization_skills:
            specialization_skills = load_specialization_skills()
            
        # Ensure user_skills is a list
        if isinstance(user_skills, str):
            skills_list = [skill.strip() for skill in user_skills.split(',')]
        else:
            skills_list = user_skills
            
        user_skills_set = set(skills_list)
        
        # If the target specialization is a legal specialization, perform special handling
        legal_specializations = [
            "Corporate Lawyer", "Litigation Attorney", "Intellectual Property Lawyer",
            "Criminal Defense Attorney", "Family Law Attorney", "Immigration Lawyer", 
            "Real Estate Attorney", "Tax Attorney", "Employment Law Attorney", "Environmental Lawyer"
        ]
        
        # Special handling for Criminal Defense Attorney
        if target_specialization == "Criminal Defense Attorney":
            # Key skills for criminal defense
            criminal_law_skills = {
                "Criminal Law", "Criminal Procedure", "Trial Advocacy", 
                "Evidence Law", "Cross-Examination", "Constitutional Law",
                "Defense Investigation", "Plea Negotiations", "Sentencing Advocacy",
                "Criminal Defense", "Jury Selection", "Witness Preparation",
                "Courtroom Demeanor", "Criminal Appeals", "Forensic Evidence Analysis",
                "Motion Practice", "Search and Seizure", "Miranda Rights"
            }
            
            # Calculate how many criminal law skills the user has
            user_criminal_skills = user_skills_set.intersection(criminal_law_skills)
            criminal_focus_score = len(user_criminal_skills) / len(criminal_law_skills)
            
            # Specific skills that are particularly valuable for Criminal Defense
            high_value_skills = {
                "Trial Advocacy", "Cross-Examination", "Constitutional Law",
                "Plea Negotiations", "Criminal Law", "Criminal Procedure"
            }
            
            # Check for high-value skills presence
            high_value_present = user_skills_set.intersection(high_value_skills)
            high_value_bonus = len(high_value_present) * 0.05  # 5% bonus per high-value skill
            
            # If user has significant criminal law skills, boost match percentage
            if criminal_focus_score >= 0.3 or len(high_value_present) >= 2:
                print(f"Strong criminal law focus detected with score: {criminal_focus_score:.2f}")
                print(f"High-value skills present: {high_value_present}")
                
                # Get the required skills for Criminal Defense Attorney
                required_skills = set(specialization_skills.get(target_specialization, []))
                
                # Calculate direct matches
                matched_skills = user_skills_set.intersection(required_skills)
                
                # Calculate missing skills - prioritize high-value missing skills
                missing_skills = required_skills - user_skills_set
                
                # Sort missing skills to prioritize high-value skills first
                sorted_missing = []
                # First add high-value missing skills
                for skill in missing_skills:
                    if skill in high_value_skills:
                        sorted_missing.append(skill)
                # Then add remaining missing skills
                for skill in missing_skills:
                    if skill not in high_value_skills:
                        sorted_missing.append(skill)
                
                # Calculate match percentage with boost for criminal focus and high-value skills
                base_match_percentage = len(matched_skills) / len(required_skills) * 100 if required_skills else 0
                boosted_match_percentage = min(100, base_match_percentage * (1 + criminal_focus_score + high_value_bonus))
                
                return sorted_missing, boosted_match_percentage, list(matched_skills)
                
        # Special handling for Corporate Lawyer
        elif target_specialization == "Corporate Lawyer":
            # Key skills for corporate law
            corporate_law_skills = {
                "Contract Drafting", "Contract Negotiation", "Corporate Governance",
                "Mergers & Acquisitions", "Securities Law", "Business Law", 
                "Corporate Compliance", "Due Diligence", "Commercial Transactions",
                "Corporate Finance", "Regulatory Compliance", "Shareholder Agreements",
                "Business Entity Formation", "Corporate Restructuring", "Licensing"
            }
            
            # Calculate how many corporate law skills the user has
            user_corporate_skills = user_skills_set.intersection(corporate_law_skills)
            corporate_focus_score = len(user_corporate_skills) / len(corporate_law_skills)
            
            # Specific skills that are particularly valuable for Corporate Law
            high_value_skills = {
                "Contract Drafting", "Contract Negotiation", "Corporate Governance",
                "Mergers & Acquisitions", "Due Diligence", "Securities Law"
            }
            
            # Check for high-value skills presence
            high_value_present = user_skills_set.intersection(high_value_skills)
            high_value_bonus = len(high_value_present) * 0.05  # 5% bonus per high-value skill
            
            # If user has significant corporate law skills, boost match percentage
            if corporate_focus_score >= 0.3 or len(high_value_present) >= 2:
                print(f"Strong corporate law focus detected with score: {corporate_focus_score:.2f}")
                print(f"High-value skills present: {high_value_present}")
                
                # Get the required skills for Corporate Lawyer
                required_skills = set(specialization_skills.get(target_specialization, []))
                
                # Calculate direct matches
                matched_skills = user_skills_set.intersection(required_skills)
                
                # Calculate missing skills - prioritize high-value missing skills
                missing_skills = required_skills - user_skills_set
                
                # Sort missing skills to prioritize high-value skills first
                sorted_missing = []
                # First add high-value missing skills
                for skill in missing_skills:
                    if skill in high_value_skills:
                        sorted_missing.append(skill)
                # Then add remaining missing skills
                for skill in missing_skills:
                    if skill not in high_value_skills:
                        sorted_missing.append(skill)
                
                # Calculate match percentage with boost for corporate focus and high-value skills
                base_match_percentage = len(matched_skills) / len(required_skills) * 100 if required_skills else 0
                boosted_match_percentage = min(100, base_match_percentage * (1 + corporate_focus_score + high_value_bonus))
                
                return sorted_missing, boosted_match_percentage, list(matched_skills)
                
        # Special handling for Intellectual Property Lawyer
        elif target_specialization == "Intellectual Property Lawyer":
            # Key skills for IP law
            ip_law_skills = {
                "Patent Law", "Trademark Law", "Copyright Law", "IP Litigation",
                "IP Portfolio Management", "Licensing Agreements", "IP Strategy",
                "Trade Secrets", "IP Enforcement", "IP Due Diligence"
            }
            
            # Calculate how many IP law skills the user has
            user_ip_skills = user_skills_set.intersection(ip_law_skills)
            ip_focus_score = len(user_ip_skills) / len(ip_law_skills)
            
            # If user has significant IP law skills, boost match percentage
            if ip_focus_score >= 0.4:
                print(f"Strong intellectual property law focus detected with score: {ip_focus_score:.2f}")
                
                # Get the required skills for IP Lawyer
                required_skills = set(specialization_skills.get(target_specialization, []))
                
                # Calculate direct matches
                matched_skills = user_skills_set.intersection(required_skills)
                
                # Calculate missing skills
                missing_skills = required_skills - user_skills_set
                
                # Calculate match percentage with boost for IP focus
                base_match_percentage = len(matched_skills) / len(required_skills) * 100 if required_skills else 0
                boosted_match_percentage = min(100, base_match_percentage * (1 + ip_focus_score))
                
                return list(missing_skills), boosted_match_percentage, list(matched_skills)
        
        # Standard processing for all other specializations
        if target_specialization in specialization_skills:
            # Direct lookup in the specialization skills dictionary
            required_skills = set(specialization_skills[target_specialization])
            
            # Calculate matches
            matched_skills = user_skills_set.intersection(required_skills)
            missing_skills = required_skills - user_skills_set
            
            # Calculate match percentage
            match_percentage = len(matched_skills) / len(required_skills) * 100 if required_skills else 0
            
            return list(missing_skills), match_percentage, list(matched_skills)
        else:
            # If specialization not found, check for similar specializations
            best_match = None
            best_similarity = 0
            
            for specialization in specialization_skills.keys():
                # Simple string similarity check
                similarity = calculate_string_similarity(target_specialization.lower(), specialization.lower())
                
                if similarity > best_similarity and similarity > 0.7:  # Threshold for considering a match
                    best_similarity = similarity
                    best_match = specialization
            
            if best_match:
                # Use the best match instead
                required_skills = set(specialization_skills[best_match])
                
                # Calculate matches
                matched_skills = user_skills_set.intersection(required_skills)
                missing_skills = required_skills - user_skills_set
                
                # Calculate match percentage
                match_percentage = len(matched_skills) / len(required_skills) * 100 if required_skills else 0
                
                return list(missing_skills), match_percentage, list(matched_skills)
        
        # If no specific specialization is matched, return generic suggestions
        generic_tech_skills = ["Problem Solving", "Research", "Analysis", "Communication", "Project Management"]
        generic_missing = [skill for skill in generic_tech_skills if skill not in user_skills_set]
        
        return generic_missing, 0, []
        
    except Exception as e:
        print(f"Error identifying missing skills: {str(e)}")
        return [], 0, []


def calculate_string_similarity(str1, str2):
    """
    Calculate simple string similarity between two strings.
    
    Args:
        str1 (str): First string
        str2 (str): Second string
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Basic implementation using set operations
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    
    # Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0

def recommend_career_path(skills_str, model_path=MODEL_PATH):
    """
    Recommend career path based on skills.
    
    Args:
        skills_str (str): Comma-separated list of skills
        model_path (str): Path to the model file
        
    Returns:
        dict: Recommendation results
    """
    try:
        # Load model components
        model_file = get_adjusted_path(model_path)
        if not os.path.exists(model_file):
            return {
                'status': 'error', 
                'message': f'Model file not found at {model_file}'
            }
        
        components = joblib.load(model_file)
        
        # Parse user skills
        user_skills = [skill.strip() for skill in skills_str.split(',') if skill.strip()]
    
        # Stage 1: Field Recommendation
        field_info = predict_field(skills_str, components)
        field = field_info['field']
    
        # Stage 2: Specialization Recommendation
        specialization_info = predict_specialization(skills_str, field, components)
        
        if isinstance(specialization_info, dict):
            specialization = specialization_info.get('specialization')
            spec_confidence = specialization_info.get('confidence', 0.7)
        else:
            # Handle legacy format (tuple of specialization and confidence)
            specialization = specialization_info[0] if isinstance(specialization_info, tuple) else specialization_info
            spec_confidence = specialization_info[1] if isinstance(specialization_info, tuple) else 0.7
    
        # Stage 3: Skill Gap Analysis
        missing_skills_info = identify_missing_skills(skills_str, specialization, components)
        
        if isinstance(missing_skills_info, dict):
            missing_skills = missing_skills_info.get('missing_skills', [])
        else:
            # Handle legacy format (set of skills)
            missing_skills = list(missing_skills_info) if missing_skills_info else []
        
        # Prepare the response
        return {
            'status': 'success',
            'recommended_field': field,
            'field_confidence': round(field_info.get('confidence', 0.7) * 100, 2),
            'recommended_specialization': specialization,
            'specialization_confidence': round(spec_confidence * 100, 2),
            'missing_skills': missing_skills,
            'existing_skills': user_skills,
            'model_version': components.get('version', '1.0'),
            'alternate_fields': field_info.get('alternate_fields', []),
            'alternate_specializations': specialization_info.get('top_specializations', [])
        }
    except Exception as e:
        print(f"Error in career path recommendation: {str(e)}")
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }

def initial_model_training(verbose=True):
    """
    Perform initial model training using synthetic data.
        
    Returns:
        bool: True if training was successful, False otherwise
    """
    try:
        if verbose:
            print("Loading synthetic data for model training...")
            
        # Load the employee and career path data
        employee_data = load_synthetic_employee_data()
        career_path_data = load_synthetic_career_path_data()
        
        if employee_data is None or career_path_data is None:
            if verbose:
                print("Failed to load synthetic data. Please generate data first.")
            return False
            
        if verbose:
            print(f"Loaded {len(employee_data)} employee records and {len(career_path_data)} career path records")
        
        # Identify popular specializations
        popular_specs = identify_popular_specializations(career_path_data, threshold=5)
        
        if verbose:
            print("\nPopular specializations by field:")
            for field, specs in popular_specs.items():
                print(f"  {field}: {len(specs)} specializations")
                for spec in specs[:5]:  # Show first 5 for each field
                    print(f"    - {spec}")
                if len(specs) > 5:
                    print(f"    - ... and {len(specs)-5} more")
        
        # Filter data to focus on popular specializations
        filtered_career_data = filter_to_popular_specializations(career_path_data, popular_specs)
        
        # Train field prediction model
        if verbose:
            print("\n=== Training Field Prediction Model ===")
        
        X_field, y_field = prepare_features(filtered_career_data, 'Field')
        field_model, field_vectorizer, field_accuracy = train_enhanced_model(X_field, y_field, verbose=verbose)
        
        # Train specialization prediction model - one model per field
        if verbose:
            print("\n=== Training Specialization Prediction Models ===")
            
        specialization_models = {}
        specialization_vectorizers = {}
        specialization_accuracies = {}
        
        # Get all unique fields
        fields = filtered_career_data['Field'].unique()
        
        for field in fields:
            if verbose:
                print(f"\nTraining model for field: {field}")
                
            # Filter data for this field
            field_data = filtered_career_data[filtered_career_data['Field'] == field]
            
            # Check if we have enough data
            if len(field_data) < 5:
                if verbose:
                    print(f"Not enough data for field {field}. Skipping...")
                continue
                
            # Prepare features
            X_spec, y_spec = prepare_features(field_data, 'Specialization')
            
            if len(X_spec) < 5:
                if verbose:
                    print(f"Not enough valid data for field {field}. Skipping...")
                continue
            
            # Train the model
            spec_model, spec_vectorizer, spec_accuracy = train_enhanced_model(
                X_spec, y_spec, verbose=verbose
            )
            
            # Store the model
            specialization_models[field] = spec_model
            specialization_vectorizers[field] = spec_vectorizer
            specialization_accuracies[field] = spec_accuracy
        
        # Prepare to create the final model components
        model_components = {
            'field_model': field_model,
            'field_vectorizer': field_vectorizer,
            'field_accuracy': field_accuracy,
            'specialization_models': specialization_models,
            'specialization_vectorizers': specialization_vectorizers,
            'specialization_accuracies': specialization_accuracies,
            'popular_specializations': popular_specs,
            'trained_at': datetime.now().isoformat()
        }
        
        # Save the model
        if save_model_components(model_components, verbose):
            if verbose:
                print("\n=== Model Training Complete ===")
                print(f"Field prediction accuracy: {field_accuracy:.4f}")
                
                # Calculate average specialization accuracy
                if specialization_accuracies:
                    avg_spec_accuracy = sum(specialization_accuracies.values()) / len(specialization_accuracies)
                    print(f"Average specialization prediction accuracy: {avg_spec_accuracy:.4f}")
                    
                print(f"Model saved to {MODEL_PATH}")
            
            return True
        else:
            if verbose:
                print("Failed to save model components.")
            return False
        
    except Exception as e:
        if verbose:
            print(f"Error during model training: {str(e)}")
            traceback.print_exc()
        return False

def create_minimal_dataset():
    """Create a minimal synthetic dataset for testing when no real data is available."""
    print("Creating minimal synthetic dataset for training...")
    
    # Create a simple dataset with a few fields and specializations
    employee_data = pd.DataFrame({
        'Employee ID': [f'EMP{i:03d}' for i in range(1, 21)],
        'Name': ['Test User ' + str(i) for i in range(1, 21)],
        'Age': [30 + i % 15 for i in range(1, 21)],
        'Years Experience': [5 + i % 10 for i in range(1, 21)],
        'Skills': [
            'Python, Data Analysis, Machine Learning' if i % 4 == 0 else
            'JavaScript, React, Web Development' if i % 4 == 1 else
            'Leadership, Project Management, Communication' if i % 4 == 2 else
            'Healthcare, Patient Care, Medical Knowledge' 
            for i in range(1, 21)
        ],
        'Career Goal': [
            'Data Scientist' if i % 4 == 0 else
            'Frontend Developer' if i % 4 == 1 else
            'Project Manager' if i % 4 == 2 else
            'Registered Nurse'
            for i in range(1, 21)
        ],
        'Current Role': [
            'Data Analyst' if i % 4 == 0 else
            'Web Developer' if i % 4 == 1 else
            'Team Lead' if i % 4 == 2 else
            'Nursing Assistant'
            for i in range(1, 21)
        ],
        'Field': [
            'Computer Science' if i % 4 == 0 else
            'Computer Science' if i % 4 == 1 else
            'Business' if i % 4 == 2 else
            'Healthcare'
            for i in range(1, 21)
        ],
        'Specialization': [
            'Data Scientist' if i % 4 == 0 else
            'Frontend Developer' if i % 4 == 1 else
            'Project Manager' if i % 4 == 2 else
            'Registered Nurse'
            for i in range(1, 21)
        ]
    })
    
    print(f"Created minimal dataset with {len(employee_data)} records")
    return employee_data

def fine_tune_specialization_mapping(specialization_to_field):
    """
    Fine-tune the specialization to field mapping with manual corrections
    for important specializations to ensure more accurate recommendations.
    
    Args:
        specialization_to_field (dict): Original mapping from specialization to field
        
    Returns:
        dict: Enhanced mapping with manual corrections
    """
    # Create a copy to avoid modifying the original
    enhanced_mapping = specialization_to_field.copy()
    
    # Define specific data science related specializations that should map to Computer Science
    data_science_specializations = [
        "Data Scientist", 
        "Data Analyst", 
        "Machine Learning Engineer",
        "Data Engineer",
        "Big Data Specialist",
        "Business Intelligence Analyst",
        "AI Research Scientist",
        "NLP Engineer",
        "Computer Vision Engineer"
    ]
    
    # Ensure data science specializations map to Computer Science
    for specialization in data_science_specializations:
        enhanced_mapping[specialization] = "Computer Science"
    
    # Other important specialization corrections
    specific_mappings = {
        # Software development specializations
        "Software Engineer": "Computer Science",
        "Frontend Developer": "Computer Science",
        "Backend Developer": "Computer Science",
        "Full-Stack Developer": "Computer Science",
        "Mobile App Developer": "Computer Science",
        "Game Developer": "Computer Science",
        
        # Business specializations
        "Project Manager": "Business",
        "Business Analyst": "Business",
        "Marketing Specialist": "Business",
        "Human Resources Specialist": "Business",
        
        # Healthcare specializations
        "Registered Nurse": "Healthcare",
        "Medical Doctor": "Healthcare",
        "Physical Therapist": "Healthcare",
        
        # Correct common misclassifications
        "Biotechnology": "Biology" 
    }
    
    # Apply specific mappings
    for specialization, field in specific_mappings.items():
        enhanced_mapping[specialization] = field
    
    return enhanced_mapping

def update_model_with_feedback(feedback_data, verbose=False):
    """
    Update the model with user feedback for continuous improvement.
    
    Args:
        feedback_data (list): List of feedback entries with format:
            {
                'user_id': str,
                'skills': list,
                'recommended_field': str,
                'recommended_specialization': str,
                'user_selected_field': str,
                'user_selected_specialization': str,
                'feedback_score': int,  # 1-5 rating
                'timestamp': str,
                'additional_comments': str
            }
        verbose (bool): Whether to print verbose output
        
    Returns:
        bool: True if model was updated successfully, False otherwise
    """
    if not feedback_data:
        logger.warning("No feedback data provided for model update")
        return False
        
    try:
        # Load existing model components
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), MODEL_PATH)
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}, cannot update")
            return False
            
        model_components = joblib.load(model_path)
        
        # Process feedback data
        if verbose:
            print(f"Processing {len(feedback_data)} feedback entries for model update")
            
        # Extract valuable feedback: entries with low scores or where user selection differs
        valuable_feedback = []
        for entry in feedback_data:
            # Calculate if this is valuable feedback (incorrect prediction or low score)
            is_field_incorrect = entry.get('recommended_field') != entry.get('user_selected_field')
            is_specialization_incorrect = entry.get('recommended_specialization') != entry.get('user_selected_specialization')
            is_low_score = entry.get('feedback_score', 5) < 3
            
            if is_field_incorrect or is_specialization_incorrect or is_low_score:
                valuable_feedback.append(entry)
                
        if verbose:
            print(f"Found {len(valuable_feedback)} valuable feedback entries for model improvement")
            
        if not valuable_feedback:
            logger.info("No valuable feedback entries found for model update")
            return True  # No update needed, but not an error
            
        # Create training data from feedback
        field_feedback = []
        specialization_feedback = []
        
        for entry in valuable_feedback:
            skills = entry.get('skills', [])
            if not skills:
                continue
                
            skills_str = ", ".join(skills)
            
            # Add to field feedback if field was incorrect or score was low
            if entry.get('user_selected_field'):
                field_feedback.append({
                    'Skills': skills_str,
                    'Field': entry.get('user_selected_field')
                })
                
            # Add to specialization feedback if specialization was incorrect or score was low
            if entry.get('user_selected_specialization') and entry.get('user_selected_field'):
                specialization_feedback.append({
                    'Skills': skills_str,
                    'Field': entry.get('user_selected_field'),
                    'Specialization': entry.get('user_selected_specialization')
                })
                
        # Convert to DataFrames
        field_df = pd.DataFrame(field_feedback) if field_feedback else None
        specialization_df = pd.DataFrame(specialization_feedback) if specialization_feedback else None
        
        # Update field model if we have field feedback
        if field_df is not None and len(field_df) > 0:
            if verbose:
                print(f"Updating field model with {len(field_df)} feedback entries")
                
            # Get existing field model and vectorizer
            field_model = model_components.get('field_model')
            field_vectorizer = model_components.get('field_vectorizer')
            
            if field_model and field_vectorizer:
                # Vectorize new data
                X_field = field_vectorizer.transform(field_df['Skills'])
                y_field = field_df['Field']
                
                # Update the model (partial_fit if RandomForest doesn't have it, we need to train a new model)
                if hasattr(field_model, 'partial_fit'):
                    # If model supports incremental learning
                    field_model.partial_fit(X_field, y_field)
                else:
                    # Otherwise, we need to combine with original data and retrain
                    # This is a simplified approach - in practice, you'd want to store original training data
                    # Get original predictions as a proxy for original training data
                    original_features = field_vectorizer.get_feature_names_out()
                    dummy_data = [" ".join(original_features)]
                    X_dummy = field_vectorizer.transform(dummy_data)
                    
                    # Combine dummy data with new data 
                    # (this is a placeholder - normally you'd have actual original training data)
                    X_combined = np.vstack([X_dummy, X_field])
                    y_combined = np.concatenate([[list(set(y_field))[0]], y_field])
                    
                    # Train a new model (with increased weight on feedback data)
                    field_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    field_model.fit(X_combined, y_combined)
                    
                # Update the model in components
                model_components['field_model'] = field_model
                
        # Update specialization model if we have specialization feedback
        if specialization_df is not None and len(specialization_df) > 0:
            if verbose:
                print(f"Updating specialization model with {len(specialization_df)} feedback entries")
                
            # Get existing specialization model components
            spec_models = model_components.get('specialization_models', {})
            spec_vectorizers = model_components.get('specialization_vectorizers', {})
            
            # Group by field
            for field, field_group in specialization_df.groupby('Field'):
                if field in spec_models and field in spec_vectorizers:
                    # Get model and vectorizer for this field
                    spec_model = spec_models[field]
                    spec_vectorizer = spec_vectorizers[field]
                    
                    # Vectorize new data
                    X_spec = spec_vectorizer.transform(field_group['Skills'])
                    y_spec = field_group['Specialization']
                    
                    # Update the model
                    if hasattr(spec_model, 'partial_fit'):
                        # If model supports incremental learning
                        spec_model.partial_fit(X_spec, y_spec)
                    else:
                        # Otherwise, train a new model (simplified approach)
                        # In practice, you'd want to store original training data
                        spec_model = RandomForestClassifier(n_estimators=100, random_state=42)
                        spec_model.fit(X_spec, y_spec)
                        
                    # Update the model in components
                    spec_models[field] = spec_model
                else:
                    # Create new model and vectorizer for this field if doesn't exist
                    if verbose:
                        print(f"Creating new specialization model for field: {field}")
                        
                    # Create vectorizer and transform skills
                    spec_vectorizer = TfidfVectorizer(max_features=1000)
                    X_spec = spec_vectorizer.fit_transform(field_group['Skills'])
                    y_spec = field_group['Specialization']
                    
                    # Train model
                    spec_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    spec_model.fit(X_spec, y_spec)
                    
                    # Add to components
                    spec_models[field] = spec_model
                    spec_vectorizers[field] = spec_vectorizer
                    
            # Update the components
            model_components['specialization_models'] = spec_models
            model_components['specialization_vectorizers'] = spec_vectorizers
            
        # Update skill weights based on feedback
        try:
            # Load existing skill weights
            skill_weights = load_skill_weights()
            
            # Update weights based on feedback
            for entry in valuable_feedback:
                if entry.get('user_selected_specialization') and entry.get('skills'):
                    spec = entry.get('user_selected_specialization')
                    
                    if spec in skill_weights:
                        # For each skill in this specialty, slightly boost weights for those
                        # that appeared in successful recommendations
                        boost_factor = 0.05
                        for skill in entry.get('skills', []):
                            if skill in skill_weights[spec]:
                                # Boost the weight but cap at 1.0
                                current_weight = skill_weights[spec][skill]
                                skill_weights[spec][skill] = min(1.0, current_weight + boost_factor)
                                
            # Save updated skill weights
            skill_weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                            "data", "skill_weights.json")
            
            with open(skill_weights_path, 'w') as f:
                json.dump(skill_weights, f, indent=4)
                
            if verbose:
                print(f"Updated skill weights saved to {skill_weights_path}")
        except Exception as e:
            logger.warning(f"Error updating skill weights: {str(e)}")
            
        # Update metadata with feedback statistics
        model_components['feedback_entries_used'] = model_components.get('feedback_entries_used', 0) + len(valuable_feedback)
        model_components['last_feedback_update'] = datetime.now().isoformat()
        
        # Calculate and update model accuracy metrics if possible
        if field_df is not None and len(field_df) > 0:
            field_pred = predict_field_batch(field_df['Skills'], model_components)
            if isinstance(field_pred, list):
                field_accuracy = sum(1 for pred, actual in zip(field_pred, field_df['Field']) if pred == actual) / len(field_pred)
                model_components['field_accuracy'] = field_accuracy
        
        # Save updated model
        if verbose:
            print("Saving updated model with feedback improvements")
            
        success = save_model_components(model_components, verbose=verbose)
        
        return success
    except Exception as e:
        logger.error(f"Error updating model with feedback: {str(e)}")
        if verbose:
            traceback.print_exc()
        return False
        
def predict_field_batch(skills_list, components):
    """
    Predict fields for a batch of skills entries.
    
    Args:
        skills_list (list): List of skills strings
        components (dict): Model components
        
    Returns:
        list: Predicted fields
    """
    if not skills_list or not components:
        return []
        
    try:
        field_model = components.get('field_model')
        field_vectorizer = components.get('field_vectorizer')
        
        if not field_model or not field_vectorizer:
            return []
            
        # Vectorize skills
        X = field_vectorizer.transform(skills_list)
        
        # Predict fields
        predictions = field_model.predict(X)
        
        return list(predictions)
    except Exception as e:
        logger.error(f"Error in batch field prediction: {str(e)}")
        return []