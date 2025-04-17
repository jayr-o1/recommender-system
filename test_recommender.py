#!/usr/bin/env python3
"""
Test script for the Career Recommender System.
This script verifies that the recommender system works correctly after being moved.
"""

import os
import sys
import json
from weighted_recommender import WeightedSkillRecommender
from recommender import recommend_career_path, recommend_field_and_career_paths

def test_basic_recommender():
    """Test the basic career recommender functionality"""
    
    print("\n=== Testing Basic Recommender ===")
    
    # Example skills
    skills = "Python, Machine Learning, Data Analysis, SQL, Statistics"
    
    # Get recommendations
    try:
        recommendations = recommend_career_path(skills)
        print(f"Recommended field: {recommendations['field']}")
        print(f"Recommended specialization: {recommendations['specialization']}")
        print(f"Confidence score: {recommendations['confidence']:.2f}")
        print(f"Missing skills: {', '.join(recommendations['missing_skills'][:3])}")
        print("Basic recommender test: SUCCESS")
    except Exception as e:
        print(f"Basic recommender test: FAILED - {str(e)}")

def test_weighted_recommender():
    """Test the weighted recommender functionality"""
    
    print("\n=== Testing Weighted Recommender ===")
    
    # Example skills with proficiency levels
    user_skills = {
        "Python": 90,
        "Machine Learning": 75,
        "Data Analysis": 85,
        "SQL": 80,
        "Statistics": 70,
        "Deep Learning": 60
    }
    
    # Get recommendations
    try:
        recommender = WeightedSkillRecommender()
        results = recommender.recommend(user_skills, top_n=3)
        
        if results.get('success', False):
            recommendations = results.get('recommendations', {})
            top_fields = recommendations.get('top_fields', [])
            top_specializations = recommendations.get('top_specializations', [])
            
            print(f"Top field: {top_fields[0]['field']} ({top_fields[0]['match_percentage']:.1f}%)")
            
            print("\nTop specializations:")
            for i, spec in enumerate(top_specializations[:3], 1):
                print(f"{i}. {spec['specialization']} ({spec['match_percentage']:.1f}%)")
                print(f"   Missing skills: {', '.join([s for s in spec['missing_skills'][:3]])}")
            
            print("Weighted recommender test: SUCCESS")
        else:
            print(f"Weighted recommender test: FAILED - {results.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"Weighted recommender test: FAILED - {str(e)}")

def test_advanced_recommendation():
    """Test the advanced recommendation functionality"""
    
    print("\n=== Testing Advanced Recommendation ===")
    
    # Example skills
    skills = ["Python", "JavaScript", "React", "Node.js", "HTML", "CSS"]
    
    # Get recommendations
    try:
        recommendations = recommend_field_and_career_paths(skills)
        
        print(f"Primary field: {recommendations['primary_field']}")
        print(f"Primary specialization: {recommendations['primary_specialization']}")
        
        print("\nAlternative career paths:")
        for i, path in enumerate(recommendations['alternative_paths'][:2], 1):
            print(f"{i}. {path['field']} - {path['specialization']} ({path['match_score']:.1f}%)")
        
        print("Advanced recommendation test: SUCCESS")
    except Exception as e:
        print(f"Advanced recommendation test: FAILED - {str(e)}")

def check_data_files():
    """Check that all required data files exist"""
    
    print("\n=== Checking Data Files ===")
    
    required_files = [
        "data/specialization_skills.json",
        "data/skill_weights.json",
        "data/skill_weights_metadata.json"
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    if all_exist:
        print("Data files check: SUCCESS")
    else:
        print("Data files check: FAILED - Some required files are missing")

if __name__ == "__main__":
    print("=== Career Recommender System Test ===")
    
    # Check that data files exist
    check_data_files()
    
    # Test basic recommender
    test_basic_recommender()
    
    # Test weighted recommender
    test_weighted_recommender()
    
    # Test advanced recommendation
    test_advanced_recommendation()
    
    print("\nTest completed.") 