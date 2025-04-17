#!/usr/bin/env python3
"""
Consolidated workflow for HR career recommendation system.
This script handles data generation, model training, and recommendations in a single unified workflow.
It consolidates all data sources into a single efficient structure for faster processing.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from utils.data_generator import SyntheticDataGenerator
from utils.model_trainer import initial_model_training, save_model_components, calculate_skill_match_percentage
from recommender import recommend_field_and_career_paths, get_career_path_match_scores

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("consolidated_workflow")

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONSOLIDATED_DATA_PATH = os.path.join(DATA_DIR, "consolidated_data.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "career_path_recommendation_model.pkl")

def ensure_dir_exists(dir_path):
    """Ensure directory exists, create if not."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Created directory: {dir_path}")

def consolidate_all_data(force_regenerate=False):
    """
    Consolidate all data sources into a single unified structure.
    
    Args:
        force_regenerate (bool): Force regeneration of data even if consolidated file exists
        
    Returns:
        dict: Consolidated data
    """
    # Check if consolidated data already exists and we're not forcing regeneration
    if os.path.exists(CONSOLIDATED_DATA_PATH) and not force_regenerate:
        logger.info(f"Loading existing consolidated data from {CONSOLIDATED_DATA_PATH}")
        try:
            with open(CONSOLIDATED_DATA_PATH, 'r') as f:
                consolidated = json.load(f)
            return consolidated
        except Exception as e:
            logger.warning(f"Error loading existing consolidated data: {str(e)}")
            logger.info("Will regenerate consolidated data")
    
    # Paths for all data files
    data_files = {
        "employee_data": os.path.join(DATA_DIR, "synthetic_employee_data.json"),
        "career_path_data": os.path.join(DATA_DIR, "synthetic_career_path_data.json"),
        "specialization_skills": os.path.join(DATA_DIR, "specialization_skills.json"),
        "skill_weights": os.path.join(DATA_DIR, "skill_weights.json"),
        "employee_weighted_data": os.path.join(DATA_DIR, "synthetic_employee_weighted_data.json"),
        "career_path_weighted_data": os.path.join(DATA_DIR, "synthetic_career_path_weighted_data.json"),
        "proficiency_test_data": os.path.join(DATA_DIR, "proficiency_test_data.json")
    }
    
    # Check if required files exist
    missing_files = [f for f, p in data_files.items() if not os.path.exists(p)]
    if missing_files:
        logger.warning(f"Missing data files: {', '.join(missing_files)}")
        logger.info("Some data may be incomplete")
    
    # Initialize consolidated data structure
    consolidated = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "source_files": {}
        },
        "data": {},
        "indexes": {}
    }
    
    # Load and process each data file
    for data_type, file_path in data_files.items():
        if not os.path.exists(file_path):
            logger.warning(f"Skipping missing file: {file_path}")
            continue
        
        try:
            # Load data based on type
            if data_type in ["employee_data", "career_path_data", 
                            "employee_weighted_data", "career_path_weighted_data",
                            "proficiency_test_data"]:
                # These are list-based JSON files
                with open(file_path, 'r') as f:
                    data = json.load(f)
                consolidated["data"][data_type] = data
                
                # Create indexes for faster lookups
                if data_type == "employee_data":
                    consolidated["indexes"]["employee_by_id"] = {
                        item["Employee ID"]: idx for idx, item in enumerate(data)
                    }
                    consolidated["indexes"]["employees_by_field"] = {}
                    consolidated["indexes"]["employees_by_specialization"] = {}
                    
                    for idx, item in enumerate(data):
                        field = item.get("Field")
                        specialization = item.get("Specialization")
                        
                        if field not in consolidated["indexes"]["employees_by_field"]:
                            consolidated["indexes"]["employees_by_field"][field] = []
                        consolidated["indexes"]["employees_by_field"][field].append(idx)
                        
                        if specialization not in consolidated["indexes"]["employees_by_specialization"]:
                            consolidated["indexes"]["employees_by_specialization"][specialization] = []
                        consolidated["indexes"]["employees_by_specialization"][specialization].append(idx)
                
                elif data_type == "career_path_data":
                    consolidated["indexes"]["career_path_by_id"] = {
                        item["ID"]: idx for idx, item in enumerate(data)
                    }
                    consolidated["indexes"]["career_paths_by_field"] = {}
                    consolidated["indexes"]["career_paths_by_specialization"] = {}
                    
                    for idx, item in enumerate(data):
                        field = item.get("Field")
                        specialization = item.get("Specialization")
                        
                        if field not in consolidated["indexes"]["career_paths_by_field"]:
                            consolidated["indexes"]["career_paths_by_field"][field] = []
                        consolidated["indexes"]["career_paths_by_field"][field].append(idx)
                        
                        if specialization not in consolidated["indexes"]["career_paths_by_specialization"]:
                            consolidated["indexes"]["career_paths_by_specialization"][specialization] = []
                        consolidated["indexes"]["career_paths_by_specialization"][specialization].append(idx)
            
            elif data_type in ["specialization_skills", "skill_weights"]:
                # These are dictionary-based JSON files
                with open(file_path, 'r') as f:
                    data = json.load(f)
                consolidated["data"][data_type] = data
            
            # Record the source file in metadata
            consolidated["metadata"]["source_files"][data_type] = os.path.abspath(file_path)
            logger.info(f"Loaded {data_type} from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading {data_type} from {file_path}: {str(e)}")
    
    # Add useful statistics to metadata
    consolidated["metadata"]["stats"] = {
        "employee_count": len(consolidated["data"].get("employee_data", [])),
        "career_path_count": len(consolidated["data"].get("career_path_data", [])),
        "specialization_count": len(consolidated["data"].get("specialization_skills", {})),
        "fields": list(consolidated["indexes"].get("career_paths_by_field", {}).keys())
    }
    
    # Save consolidated data
    ensure_dir_exists(os.path.dirname(CONSOLIDATED_DATA_PATH))
    with open(CONSOLIDATED_DATA_PATH, 'w') as f:
        json.dump(consolidated, f, indent=2)
    logger.info(f"Saved consolidated data to {CONSOLIDATED_DATA_PATH}")
    
    return consolidated

def generate_data(args):
    """Generate synthetic data."""
    logger.info("=== Generating Synthetic Data ===")
    
    # Set up paths
    ensure_dir_exists(DATA_DIR)
    
    employee_file = os.path.join(DATA_DIR, "synthetic_employee_data.json")
    career_file = os.path.join(DATA_DIR, "synthetic_career_path_data.json")
    
    logger.info(f"Initializing data generator with seed {args.seed}...")
    generator = SyntheticDataGenerator(seed=args.seed)
    
    # Generate data
    logger.info(f"Generating {args.employee_count} employee records and {args.career_path_count} career path records...")
    generator.generate_datasets(
        employee_count=args.employee_count,
        career_path_count=args.career_path_count,
        employee_file=employee_file,
        career_file=career_file,
        append=not args.replace
    )
    
    logger.info("Data generation complete!")
    
    # Consolidate data
    if args.consolidate:
        logger.info("Consolidating data...")
        consolidate_all_data(force_regenerate=True)
    
    return employee_file, career_file

def train_model_with_consolidated_data(args):
    """Train the career recommendation model using consolidated data."""
    logger.info("=== Training Career Recommendation Model ===")
    
    # Ensure we have consolidated data
    consolidated = consolidate_all_data()
    
    success = initial_model_training(verbose=not args.quiet)
    
    if success:
        logger.info("Model training completed successfully!")
    else:
        logger.error("Model training failed. Check logs for details.")
    
    return success

def parse_skill_proficiency(skills_input):
    """
    Parse skill-proficiency pairs from input string.
    
    Accepts formats:
    - "Skill1 Proficiency1, Skill2 Proficiency2, ..."
    - Multi-line format with one pair per line
    
    Args:
        skills_input (str): String containing skill-proficiency pairs
        
    Returns:
        tuple: (skills_list, proficiency_dict) where skills_list is list of skills
               and proficiency_dict maps skills to proficiency levels
    """
    skills_list = []
    proficiency_dict = {}
    
    # Handle both comma-separated and multi-line formats
    lines = skills_input.replace(',', '\n').split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Split by whitespace, assuming last element is proficiency and rest is skill name
        parts = line.strip().split()
        if len(parts) >= 2:
            # Last part is proficiency
            try:
                proficiency = int(parts[-1])
                # Everything else is the skill name
                skill = ' '.join(parts[:-1])
                skills_list.append(skill)
                proficiency_dict[skill] = proficiency
            except ValueError:
                # If last part is not a number, treat whole line as skill with default proficiency
                logger.warning(f"Could not parse proficiency from '{line}', using default value 70")
                skills_list.append(line)
                proficiency_dict[line] = 70
        elif len(parts) == 1:
            # Just a skill name, use default proficiency
            skill = parts[0]
            skills_list.append(skill)
            proficiency_dict[skill] = 70
    
    return skills_list, proficiency_dict

def get_recommendations_from_consolidated(skills_input, consolidated_data=None, model_path=MODEL_PATH):
    """
    Generate recommendations from consolidated data.
    
    Args:
        skills_input (str): Comma-separated skills string with optional proficiency values
        consolidated_data (dict, optional): Preloaded consolidated data
        model_path (str, optional): Path to the model file
        
    Returns:
        tuple: (field, specialization, top_matches)
    """
    logger.info("=== Generating Career Recommendations ===")
    
    # Parse skill-proficiency pairs
    skills_list, proficiency_dict = parse_skill_proficiency(skills_input)
    
    logger.info(f"Skills with proficiency: {', '.join([f'{s} ({p}%)' for s, p in proficiency_dict.items()])}")
    
    # Ensure we have consolidated data
    if consolidated_data is None:
        consolidated_data = consolidate_all_data()
    
    # Calculate match scores directly from consolidated data first to enhance model decision
    user_skills_set = set(skills_list)
    match_scores = {}
    field_match_scores = {}
    field_matched_skills_count = {}  # Track number of matched skills per field
    
    if consolidated_data and "data" in consolidated_data:
        for cp in consolidated_data["data"].get("career_path_data", []):
            career_path = cp["Career Path"]
            career_field = cp["Field"]
            required_skills = set(cp["Required Skills"].split(", "))
            
            if len(required_skills) > 0:
                # Calculate weighted match considering proficiency
                matched_skills = user_skills_set.intersection(required_skills)
                if matched_skills:
                    # Calculate average proficiency of matched skills
                    avg_proficiency = sum(proficiency_dict.get(skill, 70) for skill in matched_skills) / len(matched_skills)
                    # Factor in both coverage and proficiency
                    coverage = len(matched_skills) / len(required_skills)
                    # Weight: 70% coverage, 30% proficiency
                    score = (0.7 * coverage * 100) + (0.3 * avg_proficiency)
                    match_scores[career_path] = score
                    
                    # Track field-level match scores for showing alternative fields
                    if career_field not in field_match_scores:
                        field_match_scores[career_field] = []
                        field_matched_skills_count[career_field] = set()
                    
                    field_match_scores[career_field].append(score)
                    field_matched_skills_count[career_field].update(matched_skills)
                else:
                    match_scores[career_path] = 0.0
    
    # Calculate average match score per field
    field_avg_scores = {f: sum(scores)/len(scores) for f, scores in field_match_scores.items()}
    
    # Calculate total matched skills count per field
    field_skill_counts = {f: len(skills) for f, skills in field_matched_skills_count.items()}
    
    # Sort fields by their average match score
    sorted_fields = sorted(field_avg_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Sort fields by the number of matched skills (more matched skills indicates better field alignment)
    sorted_fields_by_skills = sorted(field_skill_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get recommended field and career paths from the model
    recommendation = recommend_field_and_career_paths(skills_list)
    model_field = recommendation['field']
    model_specialization = recommendation['primary_specialization']
    career_paths = recommendation['career_paths']
    
    # Get top matches overall
    top_matches = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Collect career paths by field from top matches
    field_top_paths = {}
    field_missing_skills = {}
    
    # Get all career paths sorted by match score
    all_career_paths = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build recommendations for each field
    for career_path, score in all_career_paths:
        for cp in consolidated_data["data"].get("career_path_data", []):
            if cp["Career Path"] == career_path:
                cp_field = cp["Field"]
                cp_specialization = cp["Specialization"]
                
                if cp_field not in field_top_paths:
                    field_top_paths[cp_field] = []
                    field_missing_skills[cp_field] = {}
                
                # Check if this specialization is already in the list
                spec_exists = any(spec == cp_specialization for spec, _ in field_top_paths[cp_field])
                
                # Only add if specialization is not already in the list and we have fewer than 3
                if not spec_exists and len(field_top_paths[cp_field]) < 3:
                    field_top_paths[cp_field].append((cp_specialization, score))
                    
                    # Get missing skills for this specialization
                    required_skills = set(cp["Required Skills"].split(", "))
                    missing = required_skills - user_skills_set
                    field_missing_skills[cp_field][cp_specialization] = missing
    
    # Override the field if it's "Other" and we have better matches from our analysis
    # or if the top field by skill count has significantly more matching skills
    field = model_field
    specialization = model_specialization
    
    # Logic to override the field if needed
    if model_field == "Other" and sorted_fields:
        # Use the field with the highest average match score
        field = sorted_fields[0][0]
        logger.info(f"Overriding 'Other' field with {field} based on skill matches")
    elif sorted_fields_by_skills and len(sorted_fields_by_skills) > 0:
        # If the top field by skill count has significantly more matching skills (30% more)
        top_field_by_skills = sorted_fields_by_skills[0][0]
        top_skill_count = sorted_fields_by_skills[0][1]
        
        if model_field in field_skill_counts:
            model_skill_count = field_skill_counts[model_field]
            if top_field_by_skills != model_field and top_skill_count > model_skill_count * 1.3:
                field = top_field_by_skills
                logger.info(f"Overriding model field {model_field} with {field} based on significantly more matching skills")
    
    # If we changed the field, we should also update the specialization
    if field != model_field and field in field_top_paths:
        # Use the specialization with the highest match score from that field
        specialization = field_top_paths[field][0][0]
        logger.info(f"Updated specialization to {specialization} based on field change")
    
    # Format results in structured format
    print("\n" + "="*60)
    print(f"CAREER RECOMMENDATIONS BASED ON YOUR SKILLS AND PROFICIENCY")
    print("="*60)
    
    # Primary field and specialization (from model prediction, potentially overridden)
    print(f"\nPRIMARY RECOMMENDATION:")
    print(f"  Field: {field}")
    print(f"  Specialization: {specialization}")

    # Direct approach to find matching skills for the primary recommendation
    primary_matched_skills = []
    found_specialization = False

    # Search all career paths for our specialization
    for cp in consolidated_data["data"].get("career_path_data", []):
        if cp["Field"] == field and cp["Specialization"] == specialization:
            found_specialization = True
            required_skills = set(cp["Required Skills"].split(", "))
            primary_matched_skills = sorted(list(user_skills_set.intersection(required_skills)))
            break

    # If not found, try similar specialization names
    if not found_specialization:
        closest_match = None
        closest_score = 0
        for cp in consolidated_data["data"].get("career_path_data", []):
            if cp["Field"] == field:
                # Use string similarity to find closest match
                spec_name = cp["Specialization"].lower()
                target_name = specialization.lower()
                
                # Simple matching score based on common words
                spec_words = set(spec_name.split())
                target_words = set(target_name.split())
                common_words = spec_words.intersection(target_words)
                
                if common_words:
                    score = len(common_words) / max(len(spec_words), len(target_words))
                    if score > closest_score:
                        closest_score = score
                        closest_match = cp
        
        # If we found a similar specialization, use it
        if closest_match and closest_score > 0.5:
            required_skills = set(closest_match["Required Skills"].split(", "))
            primary_matched_skills = sorted(list(user_skills_set.intersection(required_skills)))
            print(f"  (Using similar specialization: {closest_match['Specialization']})")

    # Default to Frontend Developer if Web Developer not found
    if not primary_matched_skills and specialization == "Web Developer":
        for cp in consolidated_data["data"].get("career_path_data", []):
            if cp["Field"] == "Computer Science" and cp["Specialization"] == "Frontend Developer":
                required_skills = set(cp["Required Skills"].split(", "))
                primary_matched_skills = sorted(list(user_skills_set.intersection(required_skills)))
                print(f"  (Using Frontend Developer skills as fallback)")
                break

    # Show matching skills for primary recommendation
    if primary_matched_skills:
        print(f"  Matching Skills:")
        for skill in sorted(primary_matched_skills)[:7]:
            print(f"    + {skill} ({proficiency_dict.get(skill, 70)}%)")
        if len(primary_matched_skills) > 7:
            print(f"    + and {len(primary_matched_skills)-7} more...")

    # Get missing skills for primary recommendation
    primary_missing_skills = []
    for cp_info in career_paths:
        if cp_info['specialization'] == specialization:
            primary_missing_skills = cp_info.get('missing_skills', [])
            break
    
    if primary_missing_skills:
        print(f"  Missing Skills:")
        for skill in sorted(primary_missing_skills)[:7]:
            print(f"    - {skill}")
        if len(primary_missing_skills) > 7:
            print(f"    - and {len(primary_missing_skills)-7} more...")
    
    # Show top matches by field
    print("\nTOP RECOMMENDATIONS BY FIELD:")
    
    # Get list of fields to show (include primary field + top matches)
    fields_to_show = []
    
    # Always add the primary field first if it has recommendations
    if field in field_top_paths:
        fields_to_show.append((field, field_avg_scores.get(field, 0)))
    
    # Add other fields that aren't the primary field
    for field_name, avg_score in sorted_fields:
        if field_name != field and field_name in field_top_paths and len(fields_to_show) < 3:
            fields_to_show.append((field_name, avg_score))
    
    # Show recommendations for fields
    for field_name, avg_score in fields_to_show:
        print(f"\n  {field_name} ({avg_score:.2f}% average match):")
        
        for i, (spec, score) in enumerate(field_top_paths[field_name], 1):
            print(f"    {i}. {spec} ({score:.2f}% match)")
            
            # Find required skills and matched skills for this specialization
            matched_skills = []
            for cp in consolidated_data["data"].get("career_path_data", []):
                if cp["Field"] == field_name and cp["Specialization"] == spec:
                    required_skills = set(cp["Required Skills"].split(", "))
                    matched_skills = sorted(list(user_skills_set.intersection(required_skills)))
                    break
            
            # Show matching skills
            if matched_skills:
                print(f"       Matching Skills:")
                for skill in matched_skills[:5]:
                    print(f"         + {skill} ({proficiency_dict.get(skill, 70)}%)")
                if len(matched_skills) > 5:
                    print(f"         + and {len(matched_skills)-5} more...")
            
            # Show missing skills
            if spec in field_missing_skills[field_name] and field_missing_skills[field_name][spec]:
                missing = sorted(list(field_missing_skills[field_name][spec]))[:5]
                print(f"       Missing Skills:")
                for skill in missing:
                    print(f"         - {skill}")
                if len(field_missing_skills[field_name][spec]) > 5:
                    print(f"         - and {len(field_missing_skills[field_name][spec])-5} more...")
    
    # Show proficiency improvement suggestions
    print("\nPROFICIENCY IMPROVEMENT SUGGESTIONS:")
    for career_path, _ in top_matches[:2]:  # Focus on top 2 matches
        # Use a dictionary to prevent duplicates, with skill as key
        low_proficiency_skills_dict = {}
        for cp in consolidated_data["data"].get("career_path_data", []):
            if cp["Career Path"] == career_path:
                career_field = cp["Field"]
                career_spec = cp["Specialization"]
                required_skills = set(cp["Required Skills"].split(", "))
                for skill in user_skills_set.intersection(required_skills):
                    proficiency = proficiency_dict.get(skill, 70)
                    if proficiency < 70:  # Consider skills with proficiency below 70% as areas to improve
                        # Only add if not already in dict or has lower proficiency than existing entry
                        if skill not in low_proficiency_skills_dict or proficiency < low_proficiency_skills_dict[skill]:
                            low_proficiency_skills_dict[skill] = proficiency
        
        if low_proficiency_skills_dict:
            print(f"\n  For {career_field} - {career_spec}:")
            # Convert dict to list of tuples for sorting
            sorted_skills = sorted(low_proficiency_skills_dict.items(), key=lambda x: x[1])[:3]
            for skill, prof in sorted_skills:  # Show top 3 lowest proficiency skills
                print(f"    - Improve {skill} (current: {prof}%)")
    
    print("\n" + "="*60)
    return field, specialization, top_matches

def identify_field_from_skills(skills_str, specialization_skills=None, threshold=0.15):
    """
    Identify the field of specialization based on the provided skills.
    
    Args:
        skills_str (str): Comma-separated list of skills
        specialization_skills (dict, optional): Dictionary mapping specializations to their skills
        threshold (float, optional): Minimum match threshold
        
    Returns:
        str: Identified field or None
    """
    try:
        # Load specialization skills
        if specialization_skills is None:
            specialization_skills = load_specialization_skills()
            
        # Initialize the field mapping
        field_mapping = {
            "Technology": ["Software Developer", "Web Developer", "Mobile Developer", "Software Engineer", 
                         "Full-Stack Developer", "DevOps Engineer", "Cloud Engineer", "Cloud Architect", 
                         "Security Engineer", "Cybersecurity Analyst", "Mobile App Developer"],
            "Data Science": ["Data Scientist", "Machine Learning Engineer", "AI Research Scientist", 
                           "NLP Engineer", "Data Engineer", "Research Scientist", "Deep Learning Engineer"],
            "Design": ["UI Designer", "UX Designer", "Graphic Designer", "UI/UX Designer"],
            "Business": ["Project Manager", "Product Manager", "Business Analyst"],
            "Finance": ["Financial Analyst", "Investment Banker", "Accountant", "Financial Advisor", 
                      "Portfolio Manager", "Risk Manager", "Hedge Fund Analyst", "Quantitative Analyst", 
                      "Compliance Officer", "Trader", "Auditor"],
            "Marketing": ["Marketing Specialist", "Digital Marketer", "Content Marketer", 
                       "Digital Marketing Specialist", "Brand Manager", "Market Research Analyst", 
                       "Content Strategist", "Social Media Manager"],
            "HR": ["HR Specialist", "Recruiter", "Talent Manager", "HR Manager", "Talent Acquisition Specialist", 
                 "Learning and Development Specialist", "Compensation and Benefits Analyst", 
                 "Employee Relations Specialist"],
            "Law": ["Corporate Lawyer", "Litigation Attorney", "Intellectual Property Lawyer", 
                  "Criminal Defense Attorney", "Family Law Attorney", "Immigration Lawyer", 
                  "Real Estate Attorney", "Tax Attorney", "Employment Law Attorney", "Environmental Lawyer"]
        }
        
        # Extract skills from the input
        skills_list = [skill.strip() for skill in skills_str.split(',')]
        
        # Check if we have specific legal skills
        legal_keywords = ["law", "legal", "attorney", "counsel", "litigation", "regulation", "compliance", 
                        "court", "trial", "contract", "patent", "trademark", "copyright", "criminal", 
                        "constitutional", "defense"]
        
        legal_skills_count = sum(1 for skill in skills_list if any(keyword in skill.lower() for keyword in legal_keywords))
        
        # Quick check for legal domain - if many legal keywords, prioritize Law field
        if legal_skills_count >= 3:
            # Special case for criminal law specialization
            criminal_law_keywords = ["criminal law", "criminal procedure", "trial advocacy", "evidence law", 
                                 "cross-examination", "defense investigation", "constitutional law"]
            
            criminal_skills_count = sum(1 for skill in skills_list 
                                     if any(keyword.lower() in skill.lower() for keyword in criminal_law_keywords))
            
            if criminal_skills_count >= 3:
                print(f"Detected Criminal Defense specialty with {criminal_skills_count} matching skills")
                return "Law", "Criminal Defense Attorney"
                
            # Check for other legal specialties
            corporate_law_keywords = ["contract", "corporate", "governance", "merger", "acquisition", "securities", "business law"]
            ip_law_keywords = ["patent", "trademark", "copyright", "intellectual property", "ip litigation"]
            
            corporate_skills_count = sum(1 for skill in skills_list 
                                      if any(keyword.lower() in skill.lower() for keyword in corporate_law_keywords))
            ip_skills_count = sum(1 for skill in skills_list 
                               if any(keyword.lower() in skill.lower() for keyword in ip_law_keywords))
            
            if corporate_skills_count >= 3:
                return "Law", "Corporate Lawyer"
            elif ip_skills_count >= 3:
                return "Law", "Intellectual Property Lawyer"
            
            return "Law", None
        
        # Calculate matches for each field
        field_matches = {}
        
        # Calculate matches for each specialization
        specialization_matches = {}
        
        for specialization, required_skills in specialization_skills.items():
            required_skills_set = set(required_skills)
            user_skills_set = set(skills_list)
            
            # Calculate skill matches
            matches = user_skills_set.intersection(required_skills_set)
            match_score = len(matches) / len(required_skills_set) if required_skills_set else 0
            
            # Store the match score
            specialization_matches[specialization] = {
                'score': match_score,
                'matched_skills': list(matches)
            }
            
            # Find the field for this specialization
            for field, specializations in field_mapping.items():
                if specialization in specializations:
                    # Add to field score
                    if field not in field_matches:
                        field_matches[field] = {'score': 0, 'specializations': []}
                    
                    field_matches[field]['score'] += match_score
                    if match_score > 0:
                        field_matches[field]['specializations'].append({
                            'name': specialization,
                            'score': match_score,
                            'matched_skills': list(matches)
                        })
                    break
        
        # Find the field with the highest match score
        best_field = None
        best_score = 0
        
        for field, match_info in field_matches.items():
            if match_info['score'] > best_score:
                best_score = match_info['score']
                best_field = field
                
        # Find the best specialization in the best field
        best_specialization = None
        best_specialization_score = 0
        
        if best_field:
            # Sort specializations in the best field by score
            specializations = field_matches[best_field]['specializations']
            specializations.sort(key=lambda x: x['score'], reverse=True)
            
            if specializations:
                best_specialization = specializations[0]['name']
                best_specialization_score = specializations[0]['score']
        
        # Only return a field if the match score is above the threshold
        if best_score > threshold:
            return best_field, best_specialization
        
        # If no significant matches, check for programming languages or technology terms
        tech_terms = [
            "python", "java", "javascript", "typescript", "c#", "c++", "go", "rust", "php", "ruby",
            "sql", "nosql", "aws", "azure", "gcp", "react", "angular", "vue", "node", "django",
            "flask", "spring", "docker", "kubernetes", "terraform", "git", "cloud", "devops", "agile",
            "api", "frontend", "backend", "fullstack", "database"
        ]
        
        data_science_terms = [
            "ml", "ai", "machine learning", "artificial intelligence", "data science", "deep learning", 
            "nlp", "natural language processing", "neural networks", "tensorflow", "pytorch", "keras",
            "data analysis", "statistics", "pandas", "numpy", "scikit-learn", "data mining", 
            "computer vision", "reinforcement learning"
        ]
        
        tech_count = sum(1 for skill in skills_list if any(term in skill.lower() for term in tech_terms))
        data_science_count = sum(1 for skill in skills_list if any(term in skill.lower() for term in data_science_terms))
        
        if data_science_count >= 3:
            return "Data Science", None
        elif tech_count >= 3:
            return "Technology", None
        
        return None, None
    
    except Exception as e:
        print(f"Error in field identification: {str(e)}")
        return None, None


def recommend_specializations(user_skills, field, components, top_n=5):
    """
    Recommend specializations based on user skills and identified field.
    
    Args:
        user_skills (str): Comma-separated list of user skills
        field (str): Identified field of interest
        components (dict): Model components
        top_n (int): Number of top specializations to return
        
    Returns:
        list: Recommended specializations with match details
    """
    try:
        # Load specialization-specific skills
        specialization_skills = components.get('specialization_skills')
        if not specialization_skills:
            specialization_skills = load_specialization_skills()
        
        # Create a mapping of fields to specializations
        field_mapping = {
            "Technology": ["Software Developer", "Web Developer", "Mobile Developer", "Software Engineer", 
                         "Full-Stack Developer", "DevOps Engineer", "Cloud Engineer", "Cloud Architect", 
                         "Security Engineer", "Cybersecurity Analyst", "Mobile App Developer"],
            "Data Science": ["Data Scientist", "Machine Learning Engineer", "AI Research Scientist",
                           "NLP Engineer", "Data Engineer", "Research Scientist", "Deep Learning Engineer"],
            "Design": ["UI Designer", "UX Designer", "Graphic Designer", "UI/UX Designer"],
            "Business": ["Project Manager", "Product Manager", "Business Analyst"],
            "Finance": ["Financial Analyst", "Investment Banker", "Accountant", "Financial Advisor", 
                      "Portfolio Manager", "Risk Manager", "Hedge Fund Analyst", "Quantitative Analyst", 
                      "Compliance Officer", "Trader", "Auditor"],
            "Marketing": ["Marketing Specialist", "Digital Marketer", "Content Marketer", 
                       "Digital Marketing Specialist", "Brand Manager", "Market Research Analyst", 
                       "Content Strategist", "Social Media Manager"],
            "HR": ["HR Specialist", "Recruiter", "Talent Manager", "HR Manager", "Talent Acquisition Specialist", 
                 "Learning and Development Specialist", "Compensation and Benefits Analyst", 
                 "Employee Relations Specialist"],
            "Law": ["Corporate Lawyer", "Litigation Attorney", "Intellectual Property Lawyer", 
                  "Criminal Defense Attorney", "Family Law Attorney", "Immigration Lawyer", 
                  "Real Estate Attorney", "Tax Attorney", "Employment Law Attorney", "Environmental Lawyer"]
        }
        
        # Get specializations for the identified field
        field_specializations = field_mapping.get(field, [])
        
        # We'll hold user skills in a list format
        skills_list = [skill.strip() for skill in user_skills.split(',')]
        user_skills_set = set(skills_list)
        
        # Check for criminal law skills
        criminal_law_indicators = [
            "Criminal Law", "Criminal Procedure", "Trial Advocacy", "Cross-Examination", 
            "Constitutional Law", "Defense Investigation", "Evidence Law"
        ]
        
        criminal_skills_match = sum(1 for skill in skills_list if any(
            indicator.lower() in skill.lower() for indicator in criminal_law_indicators))
        
        if criminal_skills_match >= 3 and "Law" == field:
            # Prioritize Criminal Defense Attorney for criminal law skills
            print(f"Detected strong criminal law focus with {criminal_skills_match} matching skills")
            # Calculate match percentage for Criminal Defense Attorney
            criminal_defense_skills = set(specialization_skills.get("Criminal Defense Attorney", []))
            matched_skills = user_skills_set.intersection(criminal_defense_skills)
            match_percentage = len(matched_skills) / len(criminal_defense_skills) * 100 if criminal_defense_skills else 0
            
            # Prioritize Criminal Defense Attorney in recommendations
            if "Criminal Defense Attorney" in field_specializations:
                field_specializations.remove("Criminal Defense Attorney")
                field_specializations.insert(0, "Criminal Defense Attorney")
        
        # Calculate match percentages for each specialization in the field
        specialization_matches = []
        
        for specialization in field_specializations:
            # Get required skills for the specialization
            required_skills = specialization_skills.get(specialization, [])
            
            required_skills_set = set(required_skills)
            
            # Calculate match percentage
            if required_skills_set:
                # Calculate direct matches
                matched_skills = user_skills_set.intersection(required_skills_set)
                match_percentage = len(matched_skills) / len(required_skills_set) * 100
                
                # Calculate missing skills
                missing_skills = required_skills_set - user_skills_set
                
                # Add to matches
                specialization_matches.append({
                    'specialization': specialization,
                    'match_percentage': match_percentage,
                    'matched_skills': list(matched_skills),
                    'missing_skills': list(missing_skills)
                })
        
        # Sort by match percentage
        specialization_matches.sort(key=lambda x: x['match_percentage'], reverse=True)
        
        # Return top N specializations
        return specialization_matches[:top_n] if top_n else specialization_matches
        
    except Exception as e:
        print(f"Error recommending specializations: {str(e)}")
        return []

def main():
    """Main function to run the consolidated workflow."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Consolidated workflow for HR career recommendation system')
    parser.add_argument('--generate-data', action='store_true',
                        help='Generate synthetic data')
    parser.add_argument('--train-model', action='store_true',
                        help='Train the recommendation model')
    parser.add_argument('--recommend', action='store_true',
                        help='Generate career recommendations')
    parser.add_argument('--consolidate', action='store_true',
                        help='Consolidate all data sources')
    parser.add_argument('--run-all', action='store_true',
                        help='Run the complete workflow (generate data, train model, recommend)')
    parser.add_argument('--employee-count', type=int, default=1000, 
                        help='Number of employee records to generate')
    parser.add_argument('--career-path-count', type=int, default=800, 
                        help='Number of career path records to generate')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--replace', action='store_true', default=True,
                        help='Replace existing data files')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    parser.add_argument('--skills', type=str, default=None,
                        help='Skill-proficiency pairs (e.g., "Python 80, Java 60, SQL 75")')
    parser.add_argument('--skills-file', type=str, default=None,
                        help='File containing skill-proficiency pairs (one pair per line)')
    parser.add_argument('--model-path', type=str, default=MODEL_PATH,
                        help='Path to the model file to use for recommendations')
    
    args = parser.parse_args()
    
    # If no specific action is specified, run everything
    if not any([args.generate_data, args.train_model, args.recommend, args.consolidate, args.run_all]):
        args.run_all = True
    
    # Run specified actions
    consolidated_data = None
    
    if args.consolidate or args.run_all:
        logger.info("Consolidating all data sources...")
        consolidated_data = consolidate_all_data(force_regenerate=True)
    
    if args.generate_data or args.run_all:
        generate_data(args)
        # Refresh consolidated data after generation
        consolidated_data = consolidate_all_data(force_regenerate=True)
    
    if args.train_model or args.run_all:
        train_model_with_consolidated_data(args)
    
    if args.recommend or args.run_all:
        # Get skills input, either from command line or file
        skills_input = None
        
        if args.skills_file:
            try:
                with open(args.skills_file, 'r') as f:
                    skills_input = f.read()
            except Exception as e:
                logger.error(f"Error reading skills file: {str(e)}")
        
        if args.skills:
            skills_input = args.skills
        
        if not skills_input:
            # Default skills with proficiency
            skills_input = "Python 85, Machine Learning 75, Data Analysis 80, Statistics 70, SQL 90, Communication 65"
        
        get_recommendations_from_consolidated(skills_input, consolidated_data, model_path=args.model_path)

if __name__ == "__main__":
    main() 