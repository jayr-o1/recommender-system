#!/usr/bin/env python3
"""
Train a specialization matcher model based on specialization_skills.json
"""

import os
import json
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_specialization_skills():
    """Load the specialization skills from JSON file."""
    skills_file_path = os.path.join(os.path.dirname(__file__), "data/specialization_skills.json")
    try:
        with open(skills_file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading specialization skills: {str(e)}")
        return {}

def generate_training_data(specialization_skills, samples_per_specialization=100):
    """
    Generate synthetic training data based on the specialization skills.
    Each sample will have a random subset of skills from the specialization.
    """
    data = []
    
    for specialization, skills in specialization_skills.items():
        if not skills:
            continue
            
        for _ in range(samples_per_specialization):
            # Determine how many skills to include (between 30% and 90% of total)
            num_skills = random.randint(max(1, int(len(skills) * 0.3)), max(2, int(len(skills) * 0.9)))
            
            # Select random skills
            selected_skills = random.sample(skills, num_skills)
            
            # Add some noise - 10% chance to add each unrelated skill
            for other_specialization, other_skills in specialization_skills.items():
                if other_specialization != specialization:
                    for skill in other_skills:
                        if skill not in selected_skills and random.random() < 0.1:
                            # Only add if it doesn't create too much noise
                            if len([s for s in selected_skills if s in other_skills]) < num_skills * 0.3:
                                selected_skills.append(skill)
            
            # Create a comma-separated skill string
            skill_str = ", ".join(selected_skills)
            
            data.append({
                'skills': skill_str,
                'specialization': specialization
            })
    
    return pd.DataFrame(data)

def train_specialization_model(df):
    """Train a model to predict specialization based on skills."""
    # Prepare features and targets
    X = df['skills']
    y = df['specialization']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create feature extractor
    vectorizer = TfidfVectorizer(
        max_features=500,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate on test data
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
    # Return the model and vectorizer
    return {
        'specialization_matcher': model,
        'specialization_vectorizer': vectorizer,
        'training_accuracy': accuracy
    }

def predict_specialization_from_skills(skills_str, model_components):
    """Predict specialization from skills string using the trained model."""
    vectorizer = model_components['specialization_vectorizer']
    model = model_components['specialization_matcher']
    
    # Transform input
    X = vectorizer.transform([skills_str])
    
    # Get prediction and probabilities
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    # Get class indices sorted by probability
    sorted_indices = np.argsort(proba)[::-1]
    classes = model.classes_
    
    # Create results
    results = []
    for idx in sorted_indices[:5]:  # Get top 5 predictions
        results.append({
            'specialization': classes[idx],
            'confidence': float(proba[idx])
        })
    
    return {
        'specialization': prediction,
        'confidence': float(proba[np.where(classes == prediction)[0][0]]),
        'top_specializations': results
    }

def save_model(model_components):
    """Save the model components."""
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, "specialization_matcher_model.pkl")
    joblib.dump(model_components, model_path)
    print(f"Model saved to {model_path}")
    
    # Also update main model
    main_model_path = os.path.join(models_dir, "career_path_recommendation_model.pkl")
    if os.path.exists(main_model_path):
        main_model = joblib.load(main_model_path)
        main_model.update({
            'specialization_matcher': model_components['specialization_matcher'],
            'specialization_vectorizer': model_components['specialization_vectorizer']
        })
        joblib.dump(main_model, main_model_path)
        print(f"Updated main model at {main_model_path}")

def test_model(model_components):
    """Test the model with a few examples."""
    test_skills = [
        "Python, Data Analysis, Statistical Modeling, Machine Learning, Data Visualization",
        "JavaScript, React.js, HTML, CSS, Frontend Development, UI Design",
        "AWS, Docker, Kubernetes, CI/CD, DevOps, Infrastructure as Code"
    ]
    
    print("\nTesting model with examples:")
    for skills in test_skills:
        result = predict_specialization_from_skills(skills, model_components)
        print(f"\nSkills: {skills}")
        print(f"Predicted: {result['specialization']} (Confidence: {result['confidence']:.4f})")
        print("Top specializations:")
        for spec in result['top_specializations'][:3]:
            print(f"- {spec['specialization']} ({spec['confidence']:.4f})")

def main():
    # Load specialization skills
    specialization_skills = load_specialization_skills()
    print(f"Loaded {len(specialization_skills)} specializations with skills")
    
    # Generate training data
    df = generate_training_data(specialization_skills)
    print(f"Generated {len(df)} training samples")
    
    # Train model
    model_components = train_specialization_model(df)
    
    # Test model
    test_model(model_components)
    
    # Save model
    save_model(model_components)

if __name__ == "__main__":
    main() 