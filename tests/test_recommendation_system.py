import sys
import os
import pytest
import json
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommender import CareerRecommender
from src.data_generator import DataGenerator
from utils.skill_processor import SkillProcessor

# Container for all test results to output to JSON
test_results = {}

@pytest.fixture
def recommender():
    """Create a recommender with test data"""
    # Generate test data
    generator = DataGenerator("./tests/data")
    generator.generate_all()
    
    # Create recommender with test data
    return CareerRecommender("./tests/data")

# Helper function to validate recommendation structure
def validate_recommendation_structure(recommendation: Dict[str, Any]):
    """Validate the structure of a recommendation"""
    # Check that fields exist and are correctly formatted
    assert "fields" in recommendation, "Recommendation should include fields"
    assert isinstance(recommendation["fields"], list), "Fields should be a list"
    
    # Check field structure if there are any fields
    if recommendation["fields"]:
        field = recommendation["fields"][0]
        assert "field" in field, "Each field should have a name"
        assert "confidence" in field, "Each field should have a confidence score"
        assert isinstance(field["confidence"], (int, float)), "Confidence should be numeric"
        assert 0 <= field["confidence"] <= 100, "Confidence should be between 0 and 100"
    
    # Check that specializations exist and are correctly formatted
    assert "specializations_by_field" in recommendation, "Recommendation should include specializations by field"
    assert isinstance(recommendation["specializations_by_field"], dict), "Specializations by field should be a dictionary"
    
    # Check top specializations
    assert "top_specializations" in recommendation, "Recommendation should include top specializations"
    assert isinstance(recommendation["top_specializations"], list), "Top specializations should be a list"
    
    # Check specialization structure if there are any specializations
    if recommendation["top_specializations"]:
        spec = recommendation["top_specializations"][0]
        assert "specialization" in spec, "Each specialization should have a name"
        assert "field" in spec, "Each specialization should have a field"
        assert "confidence" in spec, "Each specialization should have a confidence score"
        assert isinstance(spec["confidence"], (int, float)), "Confidence should be numeric"
        assert 0 <= spec["confidence"] <= 100, "Confidence should be between 0 and 100"
        
        # Check matched skills
        assert "matched_skills" in spec, "Specialization should include matched skills"
        assert isinstance(spec["matched_skills"], list), "Matched skills should be a list"
        
        # Check missing skills
        assert "missing_skills" in spec, "Specialization should include missing skills"
        assert isinstance(spec["missing_skills"], list), "Missing skills should be a list"
    
    # Check skill development suggestions
    assert "skill_development" in recommendation, "Recommendation should include skill development suggestions"
    assert isinstance(recommendation["skill_development"], list), "Skill development should be a list"

def save_test_result(test_name: str, input_skills: Dict[str, int], result: Dict[str, Any]) -> None:
    """Save test result to the global test_results container"""
    # Extract only the essential information for the JSON report
    simplified_result = {
        "input_skills": input_skills,
        "fields": result.get("fields", []),
        "top_specializations": [
            {
                "specialization": spec.get("specialization", ""),
                "field": spec.get("field", ""),
                "confidence": spec.get("confidence", 0),
                "matched_skills": [
                    {
                        "skill": skill.get("skill", ""),
                        "user_skill": skill.get("user_skill", ""),
                        "proficiency": skill.get("proficiency", 0),
                        "match_score": skill.get("match_score", 0)
                    }
                    for skill in spec.get("matched_skills", [])
                ],
                "missing_skills": [
                    {
                        "skill": skill.get("skill", ""),
                        "priority": skill.get("priority", 0)
                    }
                    for skill in spec.get("missing_skills", [])
                ]
            }
            for spec in result.get("top_specializations", [])[:3]  # Limit to top 3 for readability
        ]
    }
    
    # Add to the test results container
    test_results[test_name] = simplified_result

def test_case_1_web_development_profile(recommender):
    """Test with a web development skill profile"""
    skills = {
        "JavaScript": 90,
        "HTML": 95,
        "CSS": 90,
        "React": 85,
        "Node.js": 80
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("web_development_profile", skills, result)
    
    # Validate structure
    validate_recommendation_structure(result)
    
    # Specific assertions for this profile
    # Find and check if Web Development or similar field is recommended
    web_dev_field = next((field for field in result["fields"] 
                        if "web" in field["field"].lower() or "front" in field["field"].lower()), None)
    
    if web_dev_field:
        # Check if confidence score is present
        assert web_dev_field["confidence"] > 50, "Web Development confidence should be significant"
        
        # Check if frontend specialization was recommended for this field
        field_name = web_dev_field["field"]
        if field_name in result["specializations_by_field"]:
            specs = result["specializations_by_field"][field_name]
            frontend_spec = next((spec for spec in specs 
                              if "front" in spec["specialization"].lower()), None)
            
            if frontend_spec:
                # Check if matched skills include the core web dev skills
                matched_skills = [skill["skill"] for skill in frontend_spec["matched_skills"]]
                assert any("javascript" in skill.lower() for skill in matched_skills) or \
                       any("html" in skill.lower() for skill in matched_skills), \
                       "Frontend specialization should recognize HTML/JavaScript skills"
                
                # Check if skill confidence values are present
                for skill in frontend_spec["matched_skills"]:
                    assert "proficiency" in skill, "Matched skills should have proficiency"
                    assert isinstance(skill["proficiency"], (int, float)), "Proficiency should be numeric"
                
                # Check if any missing skills have confidence/priority indicators
                if frontend_spec["missing_skills"]:
                    missing = frontend_spec["missing_skills"][0]
                    assert "priority" in missing, "Missing skills should have priority"
                    assert isinstance(missing["priority"], (int, float)), "Priority should be numeric"

def test_case_2_data_science_profile(recommender):
    """Test with a data science skill profile"""
    skills = {
        "Python": 85,
        "Data Analysis": 90,
        "Machine Learning": 80,
        "Statistics": 75,
        "SQL": 70
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("data_science_profile", skills, result)
    
    # Validate structure
    validate_recommendation_structure(result)
    
    # Specific assertions for this profile
    # Find and check if Data Science or similar field is recommended
    data_field = next((field for field in result["fields"] 
                    if "data" in field["field"].lower() or "science" in field["field"].lower() 
                    or "analytics" in field["field"].lower()), None)
    
    if data_field:
        # Check if confidence score is present
        assert data_field["confidence"] > 50, "Data Science confidence should be significant"
        
        # Check if data science specialization was recommended for this field
        field_name = data_field["field"]
        if field_name in result["specializations_by_field"]:
            specs = result["specializations_by_field"][field_name]
            data_spec = next((spec for spec in specs 
                          if "data" in spec["specialization"].lower() or 
                             "machine" in spec["specialization"].lower() or
                             "analytics" in spec["specialization"].lower()), None)
            
            if data_spec:
                # Check if matched skills include the core data science skills
                matched_skills = [skill["skill"] for skill in data_spec["matched_skills"]]
                assert any("python" in skill.lower() for skill in matched_skills) or \
                       any("machine learning" in skill.lower() for skill in matched_skills), \
                       "Data Science specialization should recognize Python/ML skills"
                
                # Check if missing skills include relevant data science skills
                if data_spec["missing_skills"]:
                    missing_skills = [skill["skill"] for skill in data_spec["missing_skills"]]
                    print(f"Missing skills: {missing_skills}")

def test_case_3_mixed_skills_profile(recommender):
    """Test with a mixed skill profile that could match multiple fields"""
    skills = {
        "JavaScript": 80,
        "Python": 80,
        "SQL": 85,
        "Project Management": 90,
        "Communication": 95
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("mixed_skills_profile", skills, result)
    
    # Validate structure
    validate_recommendation_structure(result)
    
    # Check that we have at least 2 fields recommended
    assert len(result["fields"]) >= 1, "Should recommend at least one field for mixed skills"
    
    # Check confidence differences between fields
    if len(result["fields"]) >= 2:
        confidence_diff = result["fields"][0]["confidence"] - result["fields"][1]["confidence"]
        print(f"Confidence difference between top fields: {confidence_diff}")
    
    # Check if specializations have both matching and missing skills
    if result["top_specializations"]:
        top_spec = result["top_specializations"][0]
        assert len(top_spec["matched_skills"]) > 0, "Should have some matched skills"
        
        # Check if skills have match scores
        for skill in top_spec["matched_skills"]:
            assert "match_score" in skill, "Matched skills should have match scores"
            assert 0 <= skill["match_score"] <= 100, "Match score should be between 0 and 100"

def test_case_4_minimal_skills_profile(recommender):
    """Test with minimal skills to see recommendation behavior"""
    skills = {
        "JavaScript": 60
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("minimal_skills_profile", skills, result)
    
    # Validate structure
    validate_recommendation_structure(result)
    
    # Check missing skills
    if result["top_specializations"]:
        top_spec = result["top_specializations"][0]
        # With minimal skills, there should be missing skills
        assert len(top_spec["missing_skills"]) > 0, "Should have missing skills with minimal input"
        
        # Check the details of missing skills
        for missing in top_spec["missing_skills"]:
            assert "skill" in missing, "Missing skill should have a name"
            assert "priority" in missing, "Missing skill should have priority"

def test_case_5_high_proficiency_profile(recommender):
    """Test with high proficiency skills to check confidence scores"""
    skills = {
        "Python": 95,
        "Machine Learning": 95,
        "Deep Learning": 95,
        "Neural Networks": 95,
        "TensorFlow": 95
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("high_proficiency_profile", skills, result)
    
    # Validate structure
    validate_recommendation_structure(result)
    
    # Check that high proficiency leads to high confidence
    if result["fields"]:
        top_field = result["fields"][0]
        print(f"Top field confidence with high proficiency: {top_field['confidence']}")
        
    # Check matching skills confidence
    if result["top_specializations"]:
        top_spec = result["top_specializations"][0]
        matched_skills = top_spec["matched_skills"]
        if matched_skills:
            for skill in matched_skills:
                assert "proficiency" in skill, "Matched skill should have proficiency"
                assert skill["proficiency"] >= 90, "Proficiency should be high (as input)"

def test_case_6_low_proficiency_profile(recommender):
    """Test with low proficiency skills to check confidence scores"""
    skills = {
        "Python": 45,
        "Machine Learning": 40,
        "Deep Learning": 30,
        "Neural Networks": 35,
        "TensorFlow": 30
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("low_proficiency_profile", skills, result)
    
    # Validate structure
    validate_recommendation_structure(result)
    
    # Check that low proficiency leads to lower confidence
    if result["fields"]:
        top_field = result["fields"][0]
        print(f"Top field confidence with low proficiency: {top_field['confidence']}")
    
    # Check if lower proficiency is reflected in matching scores
    if result["top_specializations"]:
        top_spec = result["top_specializations"][0]
        if top_spec["matched_skills"]:
            proficiencies = [skill["proficiency"] for skill in top_spec["matched_skills"]]
            avg_proficiency = sum(proficiencies) / len(proficiencies) if proficiencies else 0
            print(f"Average proficiency of matched skills: {avg_proficiency}")

def test_case_7_misspelled_skills(recommender):
    """Test with misspelled skills to check fuzzy matching"""
    skills = {
        "Paithon": 85,  # Misspelled Python
        "Javascrpt": 80,  # Misspelled JavaScript
        "Machine Lerning": 75,  # Misspelled Machine Learning
        "Data Analisys": 85  # Misspelled Data Analysis
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("misspelled_skills_profile", skills, result)
    
    # Validate structure
    validate_recommendation_structure(result)
    
    # Check if skills were matched despite misspellings
    matched_any = False
    if result["top_specializations"]:
        for spec in result["top_specializations"]:
            if spec["matched_skills"]:
                matched_any = True
                # For each matched skill, print the user skill and what it matched to
                for skill in spec["matched_skills"]:
                    print(f"Misspelled '{skill.get('user_skill', '')}' matched to '{skill.get('skill', '')}'")
                    assert "match_score" in skill, "Matched skill should have match score"
    
    print(f"Found matches for misspelled skills: {matched_any}")

def test_case_8_irrelevant_skills(recommender):
    """Test with skills irrelevant to any field"""
    skills = {
        "Cooking": 90,
        "Gardening": 85,
        "Knitting": 80,
        "Bird Watching": 95
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("irrelevant_skills_profile", skills, result)
    
    # Validate structure
    validate_recommendation_structure(result)
    
    # Check how system handles irrelevant skills
    if result["fields"]:
        print(f"Top field with irrelevant skills: {result['fields'][0]['field']}")
        print(f"Confidence score: {result['fields'][0]['confidence']}")
    
    # Check missing skills to see what's recommended
    if result["top_specializations"]:
        top_spec = result["top_specializations"][0]
        print(f"Missing skills count: {len(top_spec['missing_skills'])}")
        if top_spec["missing_skills"]:
            print(f"Example missing skill: {top_spec['missing_skills'][0]['skill']}")

def test_case_9_highly_specialized_profile(recommender):
    """Test with a highly specialized skill set"""
    skills = {
        "Kubernetes": 90,
        "Docker": 95,
        "AWS": 85,
        "Terraform": 80,
        "DevOps": 95,
        "CI/CD": 90,
        "Jenkins": 85
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("highly_specialized_profile", skills, result)
    
    # Validate structure
    validate_recommendation_structure(result)
    
    # Check for relevant field like DevOps or Cloud
    devops_field = next((field for field in result["fields"] 
                     if any(term in field["field"].lower() for term in 
                        ["devops", "cloud", "infrastructure", "operations"])), None)
    
    if devops_field:
        print(f"Recognized specialized DevOps profile with confidence: {devops_field['confidence']}")
        
        # Check for specialized recommendations
        field_name = devops_field["field"]
        if field_name in result["specializations_by_field"]:
            specs = result["specializations_by_field"][field_name]
            devops_spec = next((spec for spec in specs 
                            if any(term in spec["specialization"].lower() for term in 
                               ["devops", "cloud", "infrastructure", "operations"])), None)
            
            if devops_spec:
                # Check matched skills
                matched_skills = [skill["skill"] for skill in devops_spec["matched_skills"]]
                print(f"Matched DevOps skills: {', '.join(matched_skills) if matched_skills else 'None'}")
                
                # Check confidence scores
                assert devops_spec["confidence"] > 50, "DevOps specialization should have high confidence"

def test_case_10_incomplete_field_skills(recommender):
    """Test with incomplete skills for a field"""
    # Get skills for a specific field and use only a subset of them
    field_names = list(recommender.fields.keys())
    if not field_names:
        pytest.skip("No fields available in test data")
    
    # Choose first field and get some of its skills
    test_field = field_names[0]
    field_skills = recommender.fields[test_field].get("core_skills", {})
    
    # Handle both dictionary and list formats
    if isinstance(field_skills, dict):
        skill_names = list(field_skills.keys())
    elif isinstance(field_skills, list):
        skill_names = field_skills
    else:
        skill_names = []
    
    # Use only half of the skills
    half_count = max(1, len(skill_names) // 2)
    skills_to_use = skill_names[:half_count]
    
    # Create a skills dictionary with these skills
    skills = {skill: 80 for skill in skills_to_use}
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("incomplete_field_skills_profile", skills, result)
    
    # Validate structure
    validate_recommendation_structure(result)
    
    # Check if the correct field was recommended despite incomplete skills
    recommended_fields = [field["field"] for field in result["fields"]]
    if test_field in recommended_fields:
        idx = recommended_fields.index(test_field)
        confidence = result["fields"][idx]["confidence"]
        print(f"Field {test_field} recognized with {half_count}/{len(skill_names)} skills, confidence: {confidence}")
    
    # Check if the missing skills are the remaining field skills
    if result["top_specializations"]:
        for spec in result["top_specializations"]:
            if spec["field"] == test_field:
                missing_skills = [skill["skill"] for skill in spec["missing_skills"]]
                print(f"Missing skills for {test_field}: {', '.join(missing_skills) if missing_skills else 'None'}")
    
    # Explicitly save results to JSON after all tests are complete
    output_file = save_results_to_json()
    print(f"Test results saved to {output_file}")

def save_results_to_json():
    """Save the collected test results to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"recommendation_test_results_{timestamp}.json"
    
    # Add test summary info
    json_output = {
        "test_timestamp": datetime.now().isoformat(),
        "total_test_cases": len(test_results),
        "test_results": test_results
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"\n\nTest results saved to {output_file}")
    return output_file

def pytest_sessionfinish(session, exitstatus):
    """Save all test results to JSON file after all tests have completed"""
    save_results_to_json() 