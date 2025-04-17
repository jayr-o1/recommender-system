#!/usr/bin/env python3
import argparse
import json
import sys
import os
import textwrap
import logging
import traceback
from typing import Dict, List, Any, Optional

# Add parent directory to path so we can import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommender import CareerRecommender
from utils.skill_processor import SkillProcessor
from .train import train_models, load_data, prepare_data, save_models, generate_synthetic_users

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_command(args):
    """
    Train the recommender model with given parameters
    
    Args:
        args: Command line arguments
    """
    try:
        # Load data
        logger.info("Loading data...")
        fields, specializations, skill_weights = load_data(args.data_path)
        
        # Generate synthetic users for training
        logger.info("Generating synthetic users for training...")
        users = generate_synthetic_users(
            fields, 
            specializations, 
            skill_weights, 
            num_users=args.num_users
        )
        logger.info(f"Generated {len(users)} synthetic user profiles")
        
        # Prepare data
        logger.info("Preparing data for training...")
        X, y_field, y_spec, le_field, le_spec, feature_names = prepare_data(users, skill_weights)
        
        # Train models
        logger.info("Training models...")
        field_clf, spec_clf = train_models(X, y_field, y_spec)
        
        # Save models
        logger.info(f"Saving models to {args.model_path}...")
        save_models(field_clf, spec_clf, le_field, le_spec, feature_names, args.model_path)
        
        logger.info("Training complete!")
        return 0
    except Exception as e:
        logger.error(f"Error training model: {e}")
        logger.error(traceback.format_exc())
        return 1

def process_skills(skills_str):
    """
    Process skills string into dictionary
    
    Args:
        skills_str: String of skills and proficiency levels (e.g., "Python:90,Java:80")
        
    Returns:
        Dictionary of skills and proficiency levels
    """
    skills = {}
    
    # Handle different formats
    if ":" in skills_str:
        # Format: skill1:90,skill2:80
        pairs = skills_str.split(",")
        for pair in pairs:
            if ":" in pair:
                name, proficiency = pair.split(":", 1)
                name = name.strip()
                try:
                    proficiency = int(proficiency.strip())
                    skills[name] = min(max(proficiency, 1), 100)  # Ensure 1-100 range
                except ValueError:
                    logger.warning(f"Invalid proficiency for {name}: {proficiency}. Using default 50.")
                    skills[name] = 50
    else:
        # Format: skill1 90, skill2 80
        parts = skills_str.split(",")
        for part in parts:
            words = part.strip().split()
            if len(words) >= 2:
                # Last word is proficiency, rest is skill name
                name = " ".join(words[:-1])
                try:
                    proficiency = int(words[-1])
                    skills[name] = min(max(proficiency, 1), 100)  # Ensure 1-100 range
                except ValueError:
                    logger.warning(f"Invalid proficiency for {name}: {words[-1]}. Using default 50.")
                    skills[name] = 50
            elif len(words) == 1:
                # Just skill name, use default proficiency
                skills[words[0]] = 50
                
    return skills

def recommend_command(args):
    """
    Get career recommendations based on given skills
    
    Args:
        args: Command line arguments
    """
    try:
        # Initialize recommender
        recommender = CareerRecommender(
            data_path=args.data_path,
            model_path=args.model_path,
            fuzzy_threshold=args.fuzzy_threshold
        )
        
        # Process skills
        skills = process_skills(args.skills)
        
        if not skills:
            logger.error("No valid skills provided")
            return 1
        
        print(f"\nProcessing recommendations for:")
        for skill, level in skills.items():
            print(f"- {skill}: {level}")
        
        print(f"\nUsing fuzzy threshold: {args.fuzzy_threshold}")
        
        # Get full recommendation
        try:
            result = recommender.full_recommendation(
                skills=skills,
                top_fields=args.top_fields,
                top_specs=args.top_specs
            )
            
            # Display results
            print(f"\n===== CAREER RECOMMENDATIONS =====")
            
            # Fields
            print(f"\nTOP FIELDS:")
            for i, field in enumerate(result["fields"], 1):
                print(f"{i}. {field['field']} (Confidence: {field['confidence']}%)")
                print(f"   Description: {field['description']}")
            
            # Specializations
            print(f"\nTOP SPECIALIZATIONS:")
            for i, spec in enumerate(result["specializations"], 1):
                print(f"{i}. {spec['specialization']} (Field: {spec['field']}, Confidence: {spec['confidence']}%)")
                print(f"   Description: {spec['description']}")
                
                # Show matched skills
                if spec.get('matched_skill_details'):
                    print(f"   Matched Skills:")
                    for skill in spec['matched_skill_details']:
                        print(f"    - {skill['skill']} (Your level: {skill.get('proficiency', 'N/A')}, Required level: {skill.get('weight', 'N/A')})")
                        if 'matched_to' in skill:
                            print(f"      (Matched to: {skill['matched_to']}, Score: {skill.get('match_score', 'N/A')})")
                
                # Show missing skills
                if spec.get('missing_skills'):
                    print(f"   Missing Skills:")
                    for skill in spec['missing_skills']:
                        if isinstance(skill, dict) and 'skill' in skill:
                            print(f"    - {skill['skill']} (Required level: {skill.get('weight', 'N/A')})")
                        elif isinstance(skill, str):
                            print(f"    - {skill}")
                print()
                
            return 0
        except ValueError as e:
            if "Models not loaded" in str(e):
                logger.warning("ML models not loaded, falling back to rule-based recommendations")
                # Could implement fallback logic here
                raise
            else:
                raise
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        logger.error(traceback.format_exc())
        return 1

def main():
    """
    Command line interface for the Career Recommender system
    """
    parser = argparse.ArgumentParser(description="Career Recommender CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the recommender model")
    train_parser.add_argument("--data-path", default="data", help="Path to data directory")
    train_parser.add_argument("--model-path", default="model", help="Path to save models")
    train_parser.add_argument("--num-users", type=int, default=15000, 
                             help="Number of synthetic users to generate")
    train_parser.set_defaults(func=train_command)
    
    # Recommend command
    recommend_parser = subparsers.add_parser("recommend", help="Get career recommendations")
    recommend_parser.add_argument("--data-path", default="data", help="Path to data directory")
    recommend_parser.add_argument("--model-path", default="model", help="Path to model directory")
    recommend_parser.add_argument("--skills", required=True, 
                                help="Comma-separated list of skills with proficiency (skill:prof)")
    recommend_parser.add_argument("--top-fields", type=int, default=3, 
                                help="Number of top fields to return")
    recommend_parser.add_argument("--top-specs", type=int, default=5, 
                                help="Number of top specializations to return")
    recommend_parser.add_argument("--fuzzy-threshold", type=int, default=70,
                                help="Threshold for fuzzy matching (0-100)")
    recommend_parser.set_defaults(func=recommend_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # Run the appropriate function
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main()) 