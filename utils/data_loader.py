"""
Data loader module for the career recommendation system.
This module provides functions to load synthetic employee and career path data.
"""

import os
import pandas as pd

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

def load_synthetic_employee_data(file_path=None):
    """
    Load synthetic employee data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file (default: use predefined path)
        
    Returns:
        pandas.DataFrame: Employee data or None if loading fails
    """
    try:
        if file_path is None:
            file_path = get_adjusted_path("data/synthetic_employee_data.json")
            
        if not os.path.exists(file_path):
            print(f"Employee data file not found at {file_path}")
            return None
            
        data = pd.read_json(file_path, orient='records')
        return data
    except Exception as e:
        print(f"Error loading synthetic employee data: {str(e)}")
        return None

def load_synthetic_career_path_data(file_path=None):
    """
    Load synthetic career path data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file (default: use predefined path)
        
    Returns:
        pandas.DataFrame: Career path data or None if loading fails
    """
    try:
        if file_path is None:
            file_path = get_adjusted_path("data/synthetic_career_path_data.json")
            
        if not os.path.exists(file_path):
            print(f"Career path data file not found at {file_path}")
            return None
            
        data = pd.read_json(file_path, orient='records')
        return data
    except Exception as e:
        print(f"Error loading synthetic career path data: {str(e)}")
        return None 