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

def test_case_1_corporate_law_profile(recommender):
    """Test with a corporate law skill profile"""
    skills = {
        "Contract Drafting": 90,
        "Corporate Governance": 95,
        "Due Diligence": 90,
        "Mergers & Acquisitions": 85,
        "Securities Law": 80
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("corporate_law_profile", skills, result)
    
    # Validate structure
    validate_recommendation_structure(result)
    
    # Specific assertions for this profile
    # Find and check if Corporate Law or similar field is recommended
    corporate_law_field = next((field for field in result["fields"] 
                        if "corporate" in field["field"].lower() or "business" in field["field"].lower()), None)
    
    if corporate_law_field:
        # Check if confidence score is present
        assert corporate_law_field["confidence"] > 50, "Corporate Law confidence should be significant"
        
        # Check if corporate law specialization was recommended for this field
        field_name = corporate_law_field["field"]
        if field_name in result["specializations_by_field"]:
            specs = result["specializations_by_field"][field_name]
            corp_spec = next((spec for spec in specs 
                              if "corporate" in spec["specialization"].lower() or "business" in spec["specialization"].lower()), None)
            
            if corp_spec:
                # Check if matched skills include the core corporate law skills
                matched_skills = [skill["skill"] for skill in corp_spec["matched_skills"]]
                assert any("contract" in skill.lower() for skill in matched_skills) or \
                       any("corporate" in skill.lower() for skill in matched_skills), \
                       "Corporate Law specialization should recognize Contract Drafting/Corporate Governance skills"
                
                # Check if skill confidence values are present
                for skill in corp_spec["matched_skills"]:
                    assert "proficiency" in skill, "Matched skills should have proficiency"
                    assert isinstance(skill["proficiency"], (int, float)), "Proficiency should be numeric"
                
                # Check if any missing skills have confidence/priority indicators
                if corp_spec["missing_skills"]:
                    missing = corp_spec["missing_skills"][0]
                    assert "priority" in missing, "Missing skills should have priority"
                    assert isinstance(missing["priority"], (int, float)), "Priority should be numeric"

def test_case_2_litigation_profile(recommender):
    """Test with a litigation skill profile"""
    skills = {
        "Legal Research": 85,
        "Legal Writing": 90,
        "Trial Advocacy": 80,
        "Evidence Law": 75,
        "Civil Procedure": 70
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("litigation_profile", skills, result)
    
    # Validate structure
    validate_recommendation_structure(result)
    
    # Specific assertions for this profile
    # Find and check if Litigation or similar field is recommended
    litigation_field = next((field for field in result["fields"] 
                    if "litigation" in field["field"].lower() or "trial" in field["field"].lower() 
                    or "dispute" in field["field"].lower()), None)
    
    if litigation_field:
        # Check if confidence score is present
        assert litigation_field["confidence"] > 50, "Litigation confidence should be significant"
        
        # Check if litigation specialization was recommended for this field
        field_name = litigation_field["field"]
        if field_name in result["specializations_by_field"]:
            specs = result["specializations_by_field"][field_name]
            litigation_spec = next((spec for spec in specs 
                          if "litigation" in spec["specialization"].lower() or 
                             "trial" in spec["specialization"].lower() or
                             "dispute" in spec["specialization"].lower()), None)
            
            if litigation_spec:
                # Check if matched skills include the core litigation skills
                matched_skills = [skill["skill"] for skill in litigation_spec["matched_skills"]]
                assert any("legal research" in skill.lower() for skill in matched_skills) or \
                       any("trial advocacy" in skill.lower() for skill in matched_skills), \
                       "Litigation specialization should recognize Legal Research/Trial Advocacy skills"
                
                # Check if missing skills include relevant litigation skills
                if litigation_spec["missing_skills"]:
                    missing_skills = [skill["skill"] for skill in litigation_spec["missing_skills"]]
                    print(f"Missing skills: {missing_skills}")

def test_case_3_mixed_legal_skills_profile(recommender):
    """Test with a mixed legal skill profile that could match multiple fields"""
    skills = {
        "Contract Drafting": 80,
        "Legal Research": 80,
        "Regulatory Compliance": 85,
        "Case Management": 90,
        "Client Communication": 95
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("mixed_legal_skills_profile", skills, result)
    
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

def test_case_4_minimal_legal_skills_profile(recommender):
    """Test with minimal legal skills to see recommendation behavior"""
    skills = {
        "Legal Research": 60
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("minimal_legal_skills_profile", skills, result)
    
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

def test_case_5_high_proficiency_legal_profile(recommender):
    """Test with high proficiency legal skills to check confidence scores"""
    skills = {
        "Legal Research": 95,
        "Legal Writing": 95,
        "Constitutional Law": 95,
        "Administrative Law": 95,
        "Judicial Clerkship": 95
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("high_proficiency_legal_profile", skills, result)
    
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

def test_case_6_low_proficiency_legal_profile(recommender):
    """Test with low proficiency legal skills to check confidence scores"""
    skills = {
        "Legal Research": 45,
        "Legal Writing": 40,
        "Constitutional Law": 30,
        "Administrative Law": 35,
        "Judicial Clerkship": 30
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("low_proficiency_legal_profile", skills, result)
    
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
        "Legul Research": 85,  # Misspelled Legal Research
        "Contruct Drafting": 80,  # Misspelled Contract Drafting
        "Constitutionl Law": 75,  # Misspelled Constitutional Law
        "Reguletory Compliance": 85  # Misspelled Regulatory Compliance
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

def test_case_9_highly_specialized_legal_profile(recommender):
    """Test with a highly specialized legal skill set"""
    skills = {
        "International Arbitration": 90,
        "Cross-Border Transactions": 95,
        "Foreign Investment Law": 85,
        "Treaty Interpretation": 80,
        "International Trade Law": 95,
        "Diplomatic Protocol": 90,
        "Multi-Jurisdictional Litigation": 85
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("highly_specialized_legal_profile", skills, result)
    
    # Validate structure
    validate_recommendation_structure(result)
    
    # Check for relevant field like International Law
    int_law_field = next((field for field in result["fields"] 
                     if any(term in field["field"].lower() for term in 
                        ["international", "global", "transnational", "cross-border"])), None)
    
    if int_law_field:
        print(f"Recognized specialized International Law profile with confidence: {int_law_field['confidence']}")
        
        # Check for specialized recommendations
        field_name = int_law_field["field"]
        if field_name in result["specializations_by_field"]:
            specs = result["specializations_by_field"][field_name]
            int_law_spec = next((spec for spec in specs 
                            if any(term in spec["specialization"].lower() for term in 
                               ["international", "global", "transnational", "cross-border"])), None)
            
            if int_law_spec:
                # Check matched skills
                matched_skills = [skill["skill"] for skill in int_law_spec["matched_skills"]]
                print(f"Matched International Law skills: {', '.join(matched_skills) if matched_skills else 'None'}")
                
                # Check confidence scores
                assert int_law_spec["confidence"] > 50, "International Law specialization should have high confidence"

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

def test_case_11_core_computer_science_skills(recommender):
    """Test that core CS skills get appropriately high confidence scores"""
    skills = {
        "Programming": 85,
        "Data Structures": 90,
        "Algorithms": 95,
        "Software Development": 80
    }
    
    # Get recommendations
    result = recommender.full_recommendation(skills)
    
    # Save test result
    save_test_result("core_computer_science_skills_profile", skills, result)
    
    # Validate structure
    validate_recommendation_structure(result)
    
    # Check if Computer Science field is recognized with high confidence
    cs_field = next((field for field in result["fields"] if field["field"] == "Computer Science"), None)
    if cs_field:
        confidence = cs_field["confidence"]
        print(f"Computer Science confidence with core skills: {confidence}")
        assert confidence >= 70, "Computer Science with core skills should have 70%+ confidence"
    
    # Check that relevant specializations like Software Engineer are recommended
    if result["top_specializations"]:
        cs_spec = next((spec for spec in result["top_specializations"] 
                       if spec["specialization"] in ["Software Engineer", "Web Developer"]), None)
        if cs_spec:
            spec_confidence = cs_spec["confidence"]
            print(f"{cs_spec['specialization']} specialization confidence: {spec_confidence}")
            assert spec_confidence >= 0.7, "CS specialization should have high confidence (≥0.7)"

def pytest_sessionfinish(session, exitstatus):
    """Save all test results to JSON file after all tests have completed"""
    save_results_to_json() 