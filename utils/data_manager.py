"""
Data management utilities for career recommendation system.
This module provides functions to manage specialization skills data.
"""

import os
import json
import traceback
from .model_trainer import get_adjusted_path

def load_specialization_skills():
    """
    Load specialization-specific skills from the JSON file.
    
    Returns:
        dict: Dictionary mapping specializations to their required skills
    """
    skills_file_path = get_adjusted_path("data/specialization_skills.json")
    try:
        if os.path.exists(skills_file_path):
            with open(skills_file_path, 'r') as file:
                return json.load(file)
        else:
            print(f"Specialization skills file not found at {skills_file_path}")
            return {}
    except Exception as e:
        print(f"Error loading specialization skills: {str(e)}")
        traceback.print_exc()
        return {}

def save_specialization_skills(skills_data):
    """
    Save specialization-specific skills to the JSON file.
    
    Args:
        skills_data (dict): Dictionary mapping specializations to their required skills
        
    Returns:
        bool: True if saving was successful, False otherwise
    """
    skills_file_path = get_adjusted_path("data/specialization_skills.json")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(skills_file_path), exist_ok=True)
        
        # Write the data to the file with pretty formatting
        with open(skills_file_path, 'w') as file:
            json.dump(skills_data, file, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving specialization skills: {str(e)}")
        traceback.print_exc()
        return False

def add_specialization_skills(specialization, skills):
    """
    Add skills for a specific specialization to the data file.
    
    Args:
        specialization (str): The specialization name
        skills (list): List of skills required for the specialization
        
    Returns:
        bool: True if the operation was successful, False otherwise
    """
    try:
        # Load existing data
        skills_data = load_specialization_skills()
        
        # Add or update the specialization skills
        skills_data[specialization] = skills
        
        # Save back to file
        return save_specialization_skills(skills_data)
    except Exception as e:
        print(f"Error adding specialization skills: {str(e)}")
        traceback.print_exc()
        return False

def get_specialization_skills(specialization):
    """
    Get skills for a specific specialization.
    
    Args:
        specialization (str): The specialization name
        
    Returns:
        list: List of skills required for the specialization, or empty list if not found
    """
    skills_data = load_specialization_skills()
    return skills_data.get(specialization, [])

def remove_specialization(specialization):
    """
    Remove a specialization from the skills data file.
    
    Args:
        specialization (str): The specialization name to remove
        
    Returns:
        bool: True if the operation was successful, False otherwise
    """
    try:
        # Load existing data
        skills_data = load_specialization_skills()
        
        # Remove the specialization if it exists
        if specialization in skills_data:
            del skills_data[specialization]
            
            # Save back to file
            return save_specialization_skills(skills_data)
        
        return True  # Return True if specialization wasn't in the file to begin with
    except Exception as e:
        print(f"Error removing specialization: {str(e)}")
        traceback.print_exc()
        return False

def add_bulk_specializations(specializations_data):
    """
    Add multiple specializations with their skills at once.
    
    Args:
        specializations_data (dict): Dictionary mapping specializations to their skills
        
    Returns:
        bool: True if the operation was successful, False otherwise
    """
    try:
        # Load existing data
        skills_data = load_specialization_skills()
        
        # Update with new data
        skills_data.update(specializations_data)
        
        # Save back to file
        return save_specialization_skills(skills_data)
    except Exception as e:
        print(f"Error adding bulk specializations: {str(e)}")
        traceback.print_exc()
        return False

def get_all_specializations():
    """
    Get a list of all specializations in the skills data file.
    
    Returns:
        list: List of all specialization names
    """
    skills_data = load_specialization_skills()
    return list(skills_data.keys()) 