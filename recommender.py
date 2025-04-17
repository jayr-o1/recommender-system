#!/usr/bin/env python3
"""
Main module for career path recommendation system.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Define paths
MODEL_PATH = "models/career_path_recommendation_model.pkl"
EMPLOYEE_DATA_PATH = "data/synthetic_employee_data.json"
CAREER_PATH_DATA_PATH = "data/synthetic_career_path_data.json"

# Import utilities
from utils.model_trainer import predict_field, predict_specialization, identify_missing_skills, calculate_skill_match_percentage
from weighted_recommender import WeightedSkillRecommender

def load_model_and_data():
    """
    Load the model and data files.
    
    Returns:
        tuple: (model_components, employee_data, career_path_data)
    """
    try:
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, checking alternative path...")
            model_path = os.path.join(os.path.dirname(__file__), "models", "career_path_recommendation_model.pkl")
            
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path} either.")
            return None, None, None
            
        model_components = joblib.load(model_path)
        
        # Validate that necessary components exist in the model
        required_components = ['field_model', 'field_vectorizer']
        for component in required_components:
            if component not in model_components:
                print(f"Missing required component: {component}")
                if component == 'field_vectorizer' and 'field_tfidf' in model_components:
                    # Handle legacy naming
                    model_components['field_vectorizer'] = model_components['field_tfidf']
                    print("Using field_tfidf as field_vectorizer")
        
        # Load employee data
        employee_data_path = os.path.join(os.path.dirname(__file__), EMPLOYEE_DATA_PATH)
        try:
            employee_data = pd.read_json(employee_data_path, orient='records')
        except:
            print(f"Could not load employee data from {employee_data_path}")
            employee_data = None
            
        # Load career path data
        career_path_data_path = os.path.join(os.path.dirname(__file__), CAREER_PATH_DATA_PATH)
        try:
            career_path_data = pd.read_json(career_path_data_path, orient='records')
        except:
            print(f"Could not load career path data from {career_path_data_path}")
            career_path_data = None
            
        return model_components, employee_data, career_path_data
    except Exception as e:
        print(f"Error loading model and data: {str(e)}")
        return None, None, None

def recommend_field_and_career_paths(skills, user_id=None):
    """
    Recommend a field and possible career paths based on skills.
    
    Args:
        skills (list or str): List of skills or comma-separated string of skills
        user_id (str, optional): User ID for feedback tracking
        
    Returns:
        dict: Recommendation results including field, specialization, and skills to develop
    """
    # Standardize skills input
    if isinstance(skills, str):
        skills_list = [skill.strip() for skill in skills.split(',') if skill.strip()]
    else:
        skills_list = [skill.strip() for skill in skills if skill.strip()]
    
    if not skills_list:
        return {
            'status': 'error',
            'message': 'No valid skills provided'
        }
    
    # Prepare skills for weighted recommender (default proficiency level of 70%)
    user_skills = {skill: 70 for skill in skills_list}
    
    try:
        # Try using the weighted recommender for more accurate results
        recommender = WeightedSkillRecommender()
        weighted_result = recommender.recommend(user_skills, top_n=5)
        
        if weighted_result.get('success', False):
            recommendations = weighted_result.get('recommendations', {})
            top_fields = recommendations.get('top_fields', [])
            top_specializations = recommendations.get('top_specializations', [])
            
            if top_fields and top_specializations:
                field = top_fields[0].get('field', '')
                field_confidence = top_fields[0].get('confidence', 0.7)
                
                career_paths = []
                popular_specializations = []
                
                # Format career paths from top specializations
                for spec_info in top_specializations:
                    specialization = spec_info.get('specialization', '')
                    match_score = spec_info.get('match_percentage', 0)
                    missing_skills = spec_info.get('missing_skills', [])
                    
                    # Override field for specific specializations
                    spec_field = field
                    if specialization == "Quantitative Analyst":
                        spec_field = "Finance"
                    
                    career_path = {
                        'field': spec_field,
                        'specialization': specialization,
                        'match_score': match_score,
                        'missing_skills': missing_skills
                    }
                    career_paths.append(career_path)
                    popular_specializations.append(specialization)
                
                # Fix primary field if primary specialization is Quantitative Analyst
                primary_field = field
                primary_spec = top_specializations[0].get('specialization', '')
                if primary_spec == "Quantitative Analyst":
                    primary_field = "Finance"
                
                return {
                    'status': 'success',
                    'field': primary_field,
                    'field_confidence': field_confidence,
                    'primary_specialization': primary_spec,
                    'specialization_confidence': top_specializations[0].get('match_percentage', 0) / 100,
                    'career_paths': career_paths,
                    'popular_specializations': popular_specializations,
                    'explanation': recommendations.get('explanation', '')
                }
    except Exception as e:
        # Log the exception and fall back to the traditional model
        print(f"Error using weighted recommender: {str(e)}")
    
    # Fall back to traditional model if weighted recommender fails
    # Load model components
    components = load_model_and_data()[0]
    if components is None:
        return {
            'status': 'error',
            'message': 'Model not found or failed to load'
        }
    
    # Join skills into a string
    skills_str = ", ".join(skills_list)
    
    # Predict field
    field_result = predict_field(skills_str, components)
    
    if isinstance(field_result, dict):
        field = field_result.get('field')
        field_confidence = field_result.get('confidence', 0)
    else:
        # Backward compatibility with older prediction function
        field = field_result
        field_confidence = 0.7
    
    # Predict specialization
    specialization_result = predict_specialization(skills_str, field, components)
    
    if isinstance(specialization_result, dict):
        specialization = specialization_result.get('specialization')
        specialization_confidence = specialization_result.get('confidence', 0)
        alternate_specializations = specialization_result.get('alternate_specializations', [])
    else:
        # Backward compatibility with older prediction function
        specialization = specialization_result[0] if isinstance(specialization_result, tuple) else specialization_result
        specialization_confidence = 0.7
        alternate_specializations = []
    
    # Get popular specializations for this field
    popular_specializations = components.get('popular_specializations', {}).get(field, [])
    if specialization not in popular_specializations and popular_specializations:
        # If the predicted specialization isn't in the popular list, add it
        popular_specializations = [specialization] + [s for s in popular_specializations if s != specialization]
    
    # Identify missing skills
    missing_skills_result = identify_missing_skills(skills_str, specialization, components)
    
    # Get career paths
    career_paths = []
    
    # First add the primary recommendation
    primary_path = {
        'field': field,
        'specialization': specialization, 
        'match_score': min(100, int(specialization_confidence * 100)),
        'missing_skills': missing_skills_result.get('missing_skills', []) if isinstance(missing_skills_result, dict) else missing_skills_result,
    }
    career_paths.append(primary_path)
    
    # Add alternate recommendations from popular specializations
    for alt_spec in popular_specializations[:2]:  # Add top 2 alternatives
        if alt_spec != specialization:
            alt_missing_skills = identify_missing_skills(skills_str, alt_spec, components)
            if isinstance(alt_missing_skills, dict):
                alt_missing_skills = alt_missing_skills.get('missing_skills', [])
            
            # Calculate a slightly lower match score for alternatives
            alt_score = min(95, int(field_confidence * 85))
            
            alt_path = {
                'field': field,
                'specialization': alt_spec,
                'match_score': alt_score,
                'missing_skills': alt_missing_skills
            }
            career_paths.append(alt_path)
    
    return {
        'status': 'success',
        'field': field,
        'field_confidence': field_confidence,
        'primary_specialization': specialization,
        'specialization_confidence': specialization_confidence,
        'career_paths': career_paths,
        'popular_specializations': popular_specializations[:5]  # Include top 5 popular specializations
    }

def recommend_career_from_resume(file_path, user_id=None):
    """
    Recommends a field, top 3 career paths, required skills, lacking skills based on the skills in a resume.

    Args:
        file_path (str): Path to the resume file.
        user_id (str, optional): User ID for personalized recommendations.

    Returns:
        tuple: A tuple containing (recommendations, skills, experience)
    """
    # Parse the resume
    resume_data = parse_resume(file_path)

    # Extract skills and experiences
    skills = resume_data["Skills"]
    experiences = resume_data["Experiences"]

    # Calculate total experience (for informational purposes only)
    total_experience = calculate_total_experience(experiences)
    experience_str = f"{int(total_experience)}+ years"  # Format experience as "X+ years"

    # Get recommendations (no longer passing experience)
    recommendations = recommend_field_and_career_paths(skills, user_id)

    return recommendations, ", ".join(skills), experience_str

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
        model_file = os.path.join(os.path.dirname(__file__), model_path)
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
        
        if isinstance(field_info, dict):
            field = field_info['field']
            field_confidence = field_info['confidence']
        else:
            # Handle legacy format
            field = field_info
            field_confidence = 0.7
        
        # Stage 2: Specialization Recommendation
        specialization_info = predict_specialization(skills_str, field, components)
        
        if isinstance(specialization_info, dict):
            specialization = specialization_info.get('specialization')
            specialization_confidence = specialization_info.get('confidence', 0.7)
            top_specializations = specialization_info.get('top_specializations', [])
        else:
            # Handle legacy format (tuple of specialization and confidence)
            specialization = specialization_info[0] if isinstance(specialization_info, tuple) else specialization_info
            specialization_confidence = specialization_info[1] if isinstance(specialization_info, tuple) else 0.7
            top_specializations = []
        
        # Create specialization recommendations list (format for output)
        specialization_recommendations = []
        
        # Use top_specializations from model prediction if available
        if top_specializations:
            for spec_info in top_specializations:
                specialization_recommendations.append({
                    'specialization': spec_info['specialization'],
                    'confidence': round(spec_info['confidence'] * 100, 2)
                })
        else:
            # Fallback to legacy format or create defaults
            # Add primary recommendation
            specialization_recommendations.append({
                'specialization': specialization,
                'confidence': round(specialization_confidence * 100, 2)
            })
            
            # Get popular specializations for this field
            popular_specializations = components.get('popular_specializations', {})
            field_specializations = popular_specializations.get(field, [])
            
            # Add additional recommendations with decreasing confidence
            used_specializations = {specialization}
            
            # Try to use field-specific specializations
            for spec in field_specializations:
                if spec not in used_specializations and len(specialization_recommendations) < 3:
                    # Calculate a slightly lower confidence for each alternative
                    alt_confidence = round(max(10, specialization_confidence * 100 * (0.9 - 0.1 * (len(specialization_recommendations) - 1))), 2)
                    specialization_recommendations.append({
                        'specialization': spec,
                        'confidence': alt_confidence
                    })
                    used_specializations.add(spec)
            
            # If we still don't have 3, add generic ones
            while len(specialization_recommendations) < 3:
                generic_spec = f"{field} Specialist {len(specialization_recommendations)}"
                if generic_spec not in used_specializations:
                    # Calculate a lower confidence for generic recommendations
                    generic_confidence = round(max(5, specialization_confidence * 100 * 0.6 * (0.9 - 0.1 * (len(specialization_recommendations) - 1))), 2)
                    specialization_recommendations.append({
                        'specialization': generic_spec,
                        'confidence': generic_confidence
                    })
                    used_specializations.add(generic_spec)
        
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
            'field_confidence': round(field_confidence * 100, 2),
            'recommended_specialization': specialization,
            'specialization_confidence': round(specialization_confidence * 100, 2),
            'missing_skills': missing_skills,
            'existing_skills': user_skills,
            'model_version': components.get('version', '1.0'),
            'alternate_fields': field_info.get('alternate_fields', []) if isinstance(field_info, dict) else [],
            'specialization_recommendations': specialization_recommendations
        }
    except Exception as e:
        print(f"Error in career path recommendation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        } 

def get_career_path_match_scores(skills_list):
    """
    Get matching scores for all career paths based on skills.
    
    Args:
        skills_list (list): List of skills
        
    Returns:
        dict: Dictionary mapping career path names to match scores (percentage)
    """
    # Prepare skills as comma-separated string if it's a list
    if isinstance(skills_list, list):
        skills_str = ", ".join(skills_list)
    else:
        skills_str = skills_list
    
    # Load data
    employee_data_path = os.path.join(os.path.dirname(__file__), EMPLOYEE_DATA_PATH)
    try:
        employee_data = pd.read_json(employee_data_path, orient='records')
    except:
        print(f"Could not load employee data from {employee_data_path}")
        return {}
    
    career_path_data_path = os.path.join(os.path.dirname(__file__), CAREER_PATH_DATA_PATH)
    try:
        career_path_data = pd.read_json(career_path_data_path, orient='records')
    except:
        print(f"Could not load career path data from {career_path_data_path}")
        return {}
    
    # Calculate match score for each career path
    match_scores = {}
    for _, row in career_path_data.iterrows():
        career_path = row['Career Path']
        required_skills = row['Required Skills']
        score = calculate_skill_match_percentage(skills_str, row['Specialization'], required_skills)
        match_scores[career_path] = score
    
    return match_scores 