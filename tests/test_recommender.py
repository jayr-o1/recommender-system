import sys
import os
import pytest
from typing import Dict, Any

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommender import CareerRecommender
from src.data_generator import DataGenerator
from utils.skill_processor import SkillProcessor


@pytest.fixture
def recommender():
    """Create a recommender with test data"""
    # Generate test data
    generator = DataGenerator("./tests/data")
    generator.generate_all()
    
    # Create recommender with test data
    return CareerRecommender("./tests/data")


@pytest.fixture
def skill_processor():
    """Create a skill processor for testing"""
    return SkillProcessor()


def test_data_generator():
    """Test that the data generator creates valid data files"""
    generator = DataGenerator("./tests/data")
    
    # Generate data
    fields = generator.generate_fields()
    specializations = generator.generate_specializations()
    skill_weights = generator.generate_skill_weights()
    
    # Check that data was generated
    assert len(fields) > 0, "No fields were generated"
    assert len(specializations) > 0, "No specializations were generated"
    assert len(skill_weights) > 0, "No skill weights were generated"
    
    # Check structure
    for field_name, field_data in fields.items():
        assert "description" in field_data
        assert "core_skills" in field_data
        
    for spec_name, spec_data in specializations.items():
        assert "field" in spec_data
        assert "description" in spec_data
        assert "core_skills" in spec_data


def test_recommender_initialization(recommender):
    """Test that the recommender initializes correctly"""
    # Check that fields are loaded
    assert len(recommender.fields) > 0, "No fields were loaded"
    assert len(recommender.specializations) > 0, "No specializations were loaded"
    assert len(recommender.skill_weights) > 0, "No skill weights were loaded"


def test_rule_based_recommendation(recommender):
    """Test rule-based recommendation works when models aren't loaded"""
    # Setup test data
    data_science_skills = {
        "Python": 90,
        "SQL": 80,
        "Data Analysis": 85,
        "Statistics": 75,
        "Machine Learning": 80
    }
    
    # We'll use the _get_matching_skills_for_field method which is used
    # for rule-based recommendation when ML models aren't available
    matches = recommender._get_matching_skills_for_field(data_science_skills, "Data Science")
    assert matches > 0, "Should have matched some skills for Data Science field"
    
    # For web development, there should be fewer matches
    web_matches = recommender._get_matching_skills_for_field(data_science_skills, "Web Development")
    assert web_matches < matches, "Should have fewer matches for Web Development field"


def test_get_skill_details(recommender):
    """Test getting skill details for specializations"""
    # Test skills for Data Analyst
    data_analyst_skills = {
        "Python": 80,
        "SQL": 90,
        "Data Analysis": 85,
        "Excel": 75
    }
    
    # Get skill details
    matched, missing = recommender._get_skill_details(data_analyst_skills, "Data Analyst")
    
    # Should have matched some skills
    assert len(matched) > 0, "Should have matched some skills for Data Analyst"
    
    # First matched skill should have expected properties
    if matched:
        first_skill = matched[0]
        assert "skill" in first_skill
        assert "proficiency" in first_skill
        assert "weight" in first_skill
        assert first_skill["skill"] in data_analyst_skills


def test_input_validation_and_preparation(recommender):
    """Test input validation and preparation for ML prediction"""
    # With empty skills
    with pytest.raises(ValueError, match="No skills provided"):
        recommender.recommend_field({})
    
    # With unknown field
    with pytest.raises(ValueError, match="Unknown field"):
        recommender.recommend_specializations({"Python": 90}, field="NonexistentField")
    
    # Test prepare_input when feature_names exist
    if hasattr(recommender, 'feature_names') and recommender.feature_names:
        # Should convert skills to a feature vector
        vector = recommender.prepare_input({"Python": 90})
        assert vector.shape[1] == len(recommender.feature_names)


def test_skill_processor(skill_processor):
    """Test skill processing utilities"""
    # Test parsing text input
    skills_text = "Python 90, SQL 80, Data Analysis 85, Machine Learning"
    parsed = skill_processor.parse_skills_input(skills_text)
    
    assert parsed["Python"] == 90
    assert parsed["SQL"] == 80
    assert parsed["Data Analysis"] == 85
    assert parsed["Machine Learning"] == 70  # Default proficiency
    
    # Test skill matching
    reference_skills = ["Python", "JavaScript", "HTML", "CSS", "SQL", "Data Analysis"]
    reference_tuple = tuple(reference_skills)
    
    matched = skill_processor.match_skill("python", reference_tuple)
    assert matched == "Python"
    
    matched = skill_processor.match_skill("Javascrpt", reference_tuple)
    assert matched == "JavaScript"  # Fuzzy matching
    
    matched = skill_processor.match_skill("C++", reference_tuple)
    assert matched is None  # No match
    
    # Test standardize_skills
    user_skills = {
        "python": 90,  # lowercase
        "javascrpt": 80,  # misspelled
        "C++": 85  # not in reference
    }
    
    standardized, unmatched = skill_processor.standardize_skills(user_skills, reference_skills)
    assert "Python" in standardized
    assert "JavaScript" in standardized
    assert "C++" in unmatched 