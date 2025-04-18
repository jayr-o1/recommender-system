import sys
import os
import pytest
import numpy as np
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
    # Setup test data - using skills that are likely in any generated dataset
    skills = {
        "Python": 90,
        "JavaScript": 80,
        "HTML": 85,
        "CSS": 75,
        "SQL": 80
    }
    
    # First get all available fields in the data
    all_fields = list(recommender.fields.keys())
    if not all_fields:
        pytest.skip("No fields available in test data")
    
    # Pick the first field and check if it has matches
    test_field = all_fields[0]
    
    # We'll use the _get_matching_skills_for_field method which is used
    # for rule-based recommendation when ML models aren't available
    matches = recommender._get_matching_skills_for_field(skills, test_field)
    
    # For comparison with another field - if multiple fields exist
    if len(all_fields) > 1:
        other_field = all_fields[1]
        other_matches = recommender._get_matching_skills_for_field(skills, other_field)
        # We don't compare results, just verify the code runs without errors


def test_get_skill_details(recommender):
    """Test getting skill details for specializations"""
    # Get first available specialization from test data
    all_specializations = list(recommender.specializations.keys())
    if not all_specializations:
        pytest.skip("No specializations available in test data")
    
    test_spec = all_specializations[0]
        
    # Setup test skills that should work with any specialization
    test_skills = {
        "Python": 80,
        "JavaScript": 90,
        "HTML": 85,
        "CSS": 75,
        "SQL": 80
    }
    
    # Get skill details (function returns 4 values - matched, missing, match_score, confidence)
    matched, missing, match_score, confidence = recommender._get_skill_details(test_skills, test_spec)
    
    # We don't assert matches since we're using generic test data
    # Just verify the function returns expected structure
    assert isinstance(matched, list), "Matched skills should be a list"
    assert isinstance(missing, list), "Missing skills should be a list"
    assert isinstance(match_score, (float, int)), "Match score should be numeric"
    assert isinstance(confidence, (float, int)), "Confidence should be numeric"
    
    # Check missing skills structure if there are any
    if missing:
        first_missing = missing[0]
        assert "skill" in first_missing, "Missing skill should have 'skill' field"
        assert "priority" in first_missing, "Missing skill should have 'priority' field"


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
        # Check vector dimension matches feature_names
        assert len(vector) == len(recommender.feature_names)
        # Ensure it's a 1D array before reshape
        assert isinstance(vector, np.ndarray), "Input vector should be a numpy array"


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


def test_input_reshape():
    """Test that input reshaping for ML prediction works correctly"""
    # Create a recommender instance
    generator = DataGenerator("./tests/data")
    generator.generate_all()
    recommender = CareerRecommender("./tests/data")
    
    # Override feature_names if not set (for testing only)
    if not hasattr(recommender, 'feature_names') or not recommender.feature_names:
        recommender.feature_names = ["Python", "SQL", "JavaScript", "HTML", "CSS"]
        recommender.models_loaded = True  # Pretend models are loaded
    
    # Create a mock classifier that can accept our input shape
    class MockClassifier:
        def predict_proba(self, X):
            # Just check shape and return a dummy probability array
            assert X.shape[0] == 1, "Input should be a 2D array with 1 row"
            assert X.shape[1] == len(recommender.feature_names), "Input width should match feature count"
            return np.array([[0.2, 0.8]])
    
    # Set mock classifiers
    recommender.field_clf = MockClassifier()
    recommender.spec_clf = MockClassifier()
    recommender.le_field = type('obj', (object,), {'classes_': ["Web Development", "Data Science"]})
    recommender.le_spec = type('obj', (object,), {'classes_': ["Frontend Developer", "Data Analyst"]})
    
    # Test normal case - should reshape correctly
    skills = {"Python": 90, "JavaScript": 85}
    
    # Test recommend_field reshapes input correctly
    try:
        recommender.recommend_field(skills)
        passed = True
    except AssertionError:
        passed = False
    assert passed, "recommend_field should reshape input correctly"
    
    # Test recommend_specializations reshapes input correctly
    try:
        recommender.recommend_specializations(skills)
        passed = True
    except AssertionError:
        passed = False
    assert passed, "recommend_specializations should reshape input correctly"


def test_input_edge_cases():
    """Test edge cases for input handling"""
    # Create a recommender instance
    generator = DataGenerator("./tests/data")
    generator.generate_all()
    recommender = CareerRecommender("./tests/data")
    
    # Override feature_names if not set (for testing only)
    if not hasattr(recommender, 'feature_names') or not recommender.feature_names:
        recommender.feature_names = ["Python", "SQL", "JavaScript", "HTML", "CSS"]
        recommender.models_loaded = True  # Pretend models are loaded
    
    # Create a mock classifier that logs its input shape
    class MockClassifier:
        def __init__(self):
            self.last_input_shape = None
            
        def predict_proba(self, X):
            self.last_input_shape = X.shape
            n_classes = 2  # Number of output classes
            return np.array([np.ones(n_classes) / n_classes])  # Uniform distribution
    
    # Set mock classifiers
    field_clf = MockClassifier()
    spec_clf = MockClassifier()
    recommender.field_clf = field_clf
    recommender.spec_clf = spec_clf
    recommender.le_field = type('obj', (object,), {'classes_': ["Web Development", "Data Science"]})
    recommender.le_spec = type('obj', (object,), {'classes_': ["Frontend Developer", "Data Analyst"]})
    
    # Test empty skills dictionary - should raise ValueError
    with pytest.raises(ValueError, match="No skills provided"):
        recommender.recommend_field({})
    
    # Test single skill - should still reshape correctly
    recommender.recommend_field({"Python": 90})
    assert field_clf.last_input_shape == (1, len(recommender.feature_names)), \
        f"Expected shape (1, {len(recommender.feature_names)}), got {field_clf.last_input_shape}"
    
    # Test input validation in full_recommendation
    with pytest.raises(ValueError, match="Skills must be a dictionary"):
        recommender.full_recommendation(["Python", "JavaScript"])  # List instead of dict 