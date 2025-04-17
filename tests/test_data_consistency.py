import json
import os
import sys
import pytest
import shutil
from typing import Dict, Any

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommender import CareerRecommender
from src.data_generator import DataGenerator
from utils.skill_processor import SkillProcessor
from src.train import validate_data_consistency


@pytest.fixture
def test_data_path():
    """Path to test data directory"""
    path = "./tests/data_consistency"
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture
def data_generator(test_data_path):
    """Create a data generator that outputs to the test data path"""
    return DataGenerator(test_data_path)


@pytest.fixture
def valid_data(data_generator):
    """Generate valid test data"""
    data_generator.generate_all()
    
    # Load the generated data
    data_path = data_generator.output_dir
    
    with open(os.path.join(data_path, "fields.json"), "r") as f:
        fields = json.load(f)
        
    with open(os.path.join(data_path, "specializations.json"), "r") as f:
        specializations = json.load(f)
        
    with open(os.path.join(data_path, "skill_weights.json"), "r") as f:
        skill_weights = json.load(f)
        
    return {
        "fields": fields,
        "specializations": specializations,
        "skill_weights": skill_weights,
        "data_path": data_path
    }


def test_validate_data_consistency(valid_data):
    """Test that valid data passes consistency check"""
    # This should not raise an exception
    validate_data_consistency(
        valid_data["fields"], 
        valid_data["specializations"], 
        valid_data["skill_weights"]
    )


def test_invalid_specialization_field(valid_data, test_data_path):
    """Test that a specialization with an invalid field reference is caught"""
    # Create a deep copy of the specializations to avoid modifying the original
    specializations = {}
    for k, v in valid_data["specializations"].items():
        specializations[k] = v.copy()
    
    # Modify a specialization to have an invalid field
    specialization_name = list(specializations.keys())[0]
    specializations[specialization_name]["field"] = "NonexistentField"
    
    # Try to validate the data - should raise an exception
    with pytest.raises(ValueError, match="Data inconsistency found in specializations"):
        validate_data_consistency(
            valid_data["fields"], 
            specializations, 
            valid_data["skill_weights"]
        )


def test_missing_field_reference(valid_data, test_data_path):
    """Test that a specialization with a missing field field is caught"""
    # Create a deep copy of the specializations
    specializations = {}
    for k, v in valid_data["specializations"].items():
        specializations[k] = v.copy()
    
    # Modify a specialization to have a missing field
    specialization_name = list(specializations.keys())[0]
    del specializations[specialization_name]["field"]
    
    # Try to validate the data - should raise an exception
    with pytest.raises(ValueError, match="Data inconsistency found in specializations"):
        validate_data_consistency(
            valid_data["fields"], 
            specializations, 
            valid_data["skill_weights"]
        )


def test_skill_processor_basic_matching():
    """Test the SkillProcessor's basic matching capabilities"""
    processor = SkillProcessor()
    
    # Test with a simple set of skills
    reference_skills = ["Python", "JavaScript", "SQL", "Machine Learning"]
    reference_tuple = tuple(reference_skills)
    
    # Direct match
    assert processor.match_skill("Python", reference_tuple) == "Python"
    
    # Case insensitive match
    assert processor.match_skill("python", reference_tuple) == "Python"
    
    # Fuzzy match
    assert processor.match_skill("Pythn", reference_tuple, threshold=0.7) == "Python"
    
    # No match
    assert processor.match_skill("Rust", reference_tuple) is None


def test_missing_data_file():
    """Test handling of missing data files"""
    # Create a temp directory path
    temp_path = "./temp_test_directory"
    
    # Make sure it doesn't exist first
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    
    # Create an empty directory
    os.makedirs(temp_path)
    
    try:
        # Try to create a recommender with no data files - should raise RuntimeError
        # that wraps the FileNotFoundError
        with pytest.raises(RuntimeError, match="Error loading data: Field data file not found"):
            CareerRecommender(temp_path)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)


def test_input_validation(valid_data):
    """Test input validation in recommender"""
    recommender = CareerRecommender(valid_data["data_path"])
    
    # Test with empty skills
    with pytest.raises(ValueError, match="No skills provided"):
        recommender.recommend_field({})
        
    # Test with invalid field in recommend_specializations
    with pytest.raises(ValueError, match="Unknown field"):
        recommender.recommend_specializations({"Python": 90}, field="NonexistentField")


def test_skill_processor_error_handling():
    """Test error handling in skill processor"""
    processor = SkillProcessor()
    
    # Test with invalid input
    with pytest.raises(ValueError, match="Skills text must be a non-empty string"):
        processor.parse_skills_input("")
        
    # Test with empty reference skills
    with pytest.raises(ValueError, match="Reference skills list cannot be empty"):
        processor.standardize_skills({"Python": 90}, [])


def test_overlapping_skills():
    """Test the new overlapping skills function"""
    processor = SkillProcessor()
    
    user_skills = {
        "Python": 90,
        "JavaScript": 80,
        "Data Analysis": 85,
        "Machine Learning": 70
    }
    
    target_skills = {
        "python": 85,  # Note lowercase to test case insensitivity
        "SQL": 90,
        "Data Analysis": 80,
        "Statistics": 75
    }
    
    overlapping = processor.get_overlapping_skills(user_skills, target_skills)
    
    assert len(overlapping) == 2
    assert "python" in overlapping
    assert "Data Analysis" in overlapping
    assert overlapping["python"]["proficiency"] == 90
    assert overlapping["python"]["importance"] == 85
    assert overlapping["Data Analysis"]["proficiency"] == 85
    assert overlapping["Data Analysis"]["importance"] == 80 