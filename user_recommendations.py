#!/usr/bin/env python3
"""
Script to generate recommendations using sample proficiency data.
"""

import os
import json
import joblib
from recommender import recommend_career_path
from utils.model_trainer import identify_missing_skills, load_specialization_skills, load_skill_weights, calculate_skill_match_percentage

def main():
    # Load sample proficiency data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    proficiency_file = os.path.join(data_dir, "sample_proficiency_data.json")
    
    with open(proficiency_file, 'r') as f:
        proficiency_data = json.load(f)
        
    # Load model components for identify_missing_skills
    model_path = os.path.join(os.path.dirname(__file__), "models/career_path_recommendation_model.pkl")
    components = joblib.load(model_path)
    
    # Load specialization skills and weights
    specialization_skills = load_specialization_skills()
    skill_weights = load_skill_weights()
    
    # Process each user
    for user_id, skills in proficiency_data.items():
        print(f"\n{'='*80}")
        print(f"RECOMMENDATIONS FOR {user_id.upper()}")
        print(f"{'='*80}")
        
        # Create skills string for recommendation
        skills_str = ", ".join(skills.keys())
        
        # Get recommendations
        result = recommend_career_path(skills_str)
        
        # Print field recommendation
        print(f"\nField Recommendation: {result['recommended_field']} (Confidence: {result['field_confidence']:.2f}%)")
        
        # Print specialization recommendations
        print("\nTop Specialization Recommendations:")
        print("-" * 40)
        
        # Get top 3 specializations
        top_specializations = result['specialization_recommendations'][:3]
        
        for i, spec in enumerate(top_specializations, 1):
            print(f"{i}. {spec['specialization']} (Confidence: {spec['confidence']:.2f}%)")
            
            # Get missing skills for this specialization
            missing_skills_result = identify_missing_skills(
                skills_str, 
                spec['specialization'], 
                components, 
                skills
            )
            
            # Handle case where missing_skills_result is None
            if missing_skills_result is None:
                # Calculate match info directly
                match_info = calculate_skill_match_percentage(
                    skills, 
                    spec['specialization'], 
                    specialization_skills,
                    skill_weights
                )
                
                match_percentage = match_info['match_percentage']
                skill_coverage = match_info['skill_coverage']
                proficiency_score = match_info['proficiency_score']
                matched_skills = match_info['matched_skills']
                missing_skills = match_info['missing_skills']
            else:
                # Extract the metrics from the result
                match_percentage = missing_skills_result.get('match_percentage', 0)
                skill_coverage = missing_skills_result.get('skill_coverage', 0)
                proficiency_score = missing_skills_result.get('proficiency_score', 0)
                matched_skills = missing_skills_result.get('matched_skills', [])
                missing_skills = missing_skills_result.get('missing_skills', [])
            
            print(f"   Match Score: {match_percentage:.2f}% | Skill Coverage: {skill_coverage:.2f}% | Proficiency: {proficiency_score:.2f}%")
            
            # Print matched skills (top 5)
            if matched_skills:
                print("\n   Matched Skills:")
                for skill in matched_skills[:5]:
                    proficiency = skills.get(skill, 0)
                    print(f"   ✓ {skill} (Proficiency: {proficiency}%)")
            
            # Print missing skills (top 5)
            if missing_skills:
                print("\n   Missing Skills to Develop:")
                for skill in missing_skills[:5]:
                    # Get weight if available
                    weight = skill_weights.get(spec['specialization'], {}).get(skill, 0.5)
                    importance = "High" if weight >= 0.8 else "Medium" if weight >= 0.5 else "Low"
                    print(f"   - {skill} (Importance: {importance})")
            
            print()
        
        # Print existing skills
        print("Current Skills Profile:")
        print("-" * 40)
        # Sort skills by proficiency level (descending)
        sorted_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)
        for skill, proficiency in sorted_skills:
            tier = "Expert" if proficiency >= 85 else "Advanced" if proficiency >= 70 else "Intermediate" if proficiency >= 50 else "Beginner"
            print(f"- {skill}: {proficiency}% ({tier})")

if __name__ == "__main__":
    main() 