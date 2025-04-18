#!/usr/bin/env python3
# Detailed recommendation system test script for Python
import json
import requests
from pprint import pprint

# API endpoint for simplified recommendations
API_ENDPOINT = "http://localhost:8000/api/recommend"

# The skills provided by the user
test_skills = {
    "Dancing": 65,
    "Painting": 70,
    "Advertising": 50,
    "Marketing": 70
}

def test_recommendation_system():
    try:
        print("Testing Career Recommendation System with skills:")
        print(json.dumps(test_skills, indent=2))
        print("\nSending request to API...")
        
        response = requests.post(
            API_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json={"skills": test_skills}
        )
        
        data = response.json()
        
        print("\n=== RECOMMENDATION RESULTS ===\n")
        
        if "recommendations" not in data:
            print("ERROR: No recommendations received from API")
            print("Raw response:", json.dumps(data, indent=2))
            return
        
        recommendations = data["recommendations"]
        
        # Display fields information
        print("CAREER FIELDS:")
        if recommendations.get("fields") and len(recommendations["fields"]) > 0:
            for index, field in enumerate(recommendations["fields"]):
                print(f"\n{index + 1}. Field: {field.get('field')}")
                print(f"   Confidence: {field.get('confidence')}%")
                
                print("   Matching Skills:")
                if field.get("matching_skills") and len(field["matching_skills"]) > 0:
                    for skill in field["matching_skills"]:
                        print(f"     - {skill}")
                else:
                    print("     None")
                
                print("   Missing Skills:")
                if field.get("missing_skills") and len(field["missing_skills"]) > 0:
                    for skill in field["missing_skills"]:
                        if isinstance(skill, str):
                            skill_name = skill
                            priority = "N/A"
                        else:
                            skill_name = skill.get("skill")
                            priority = skill.get("priority", "N/A")
                        print(f"     - {skill_name} (Priority: {priority})")
                else:
                    print("     None")
        else:
            print("No career fields found in recommendations")
        
        # Display specializations information
        print("\nSPECIALIZATIONS:")
        if recommendations.get("specializations") and len(recommendations["specializations"]) > 0:
            for index, spec in enumerate(recommendations["specializations"]):
                print(f"\n{index + 1}. Specialization: {spec.get('specialization')}")
                print(f"   Field: {spec.get('field', 'N/A')}")
                print(f"   Confidence: {spec.get('confidence')}%")
                
                print("   Matched Skills:")
                if spec.get("matched_skills") and len(spec["matched_skills"]) > 0:
                    for skill in spec["matched_skills"]:
                        if isinstance(skill, str):
                            print(f"     - {skill}")
                        else:
                            skill_name = skill.get("skill") or skill.get("user_skill") or "Unknown"
                            user_skill = skill.get("user_skill") or skill_name
                            proficiency = skill.get("proficiency", "N/A")
                            match_score = skill.get("match_score", "N/A")
                            print(f"     - {skill_name} (User Skill: {user_skill}, Proficiency: {proficiency}, Match Score: {match_score})")
                else:
                    print("     None")
                
                print("   Missing Skills:")
                if spec.get("missing_skills") and len(spec["missing_skills"]) > 0:
                    for skill in spec["missing_skills"]:
                        if isinstance(skill, str):
                            skill_name = skill
                            priority = "N/A"
                        else:
                            skill_name = skill.get("skill")
                            priority = skill.get("priority", "N/A")
                        print(f"     - {skill_name} (Priority: {priority})")
                else:
                    print("     None")
        else:
            print("No specializations found in recommendations")
        
        # Analyze the issue with empty fields array
        if (not recommendations.get("fields") or len(recommendations["fields"]) == 0) and \
           recommendations.get("specializations") and len(recommendations["specializations"]) > 0:
            print("\n=== ISSUE ANALYSIS ===")
            print("Your recommendation system is returning specializations but no fields.")
            print('This explains your error: "fields: Array(0), specializations: Array(3)"')
            print("\nPOSSIBLE SOLUTIONS:")
            print("1. Implement a processing function to derive fields from specializations")
            print("2. Modify your frontend component to display specializations when fields are empty")
            print("3. Check the API implementation to ensure fields are being properly calculated and returned")
            
            # Suggest a Python fix
            print("\nPYTHON FIX FOR API:")
            print("You could modify the format_frontend_response function in src/api.py to derive fields from specializations.")
            print("Here's a code snippet to add:")
            print("""
def derive_fields_from_specializations(specializations):
    field_map = {}
    
    for spec in specializations:
        field_name = spec.get('field', 'General')
        
        if field_name not in field_map:
            field_map[field_name] = {
                'field': field_name,
                'confidence': 0,
                'matching_skills': [],
                'missing_skills': []
            }
        
        # Update confidence
        field_map[field_name]['confidence'] = max(
            field_map[field_name]['confidence'],
            spec.get('confidence', 0)
        )
        
        # Collect unique matching skills
        if spec.get('matched_skills'):
            for skill in spec['matched_skills']:
                skill_name = skill.get('skill', '') if isinstance(skill, dict) else skill
                if skill_name and skill_name not in field_map[field_name]['matching_skills']:
                    field_map[field_name]['matching_skills'].append(skill_name)
        
        # Collect unique missing skills
        if spec.get('missing_skills'):
            for skill in spec['missing_skills']:
                skill_obj = {'skill': skill, 'priority': 50} if isinstance(skill, str) else skill
                if not any(s.get('skill') == skill_obj.get('skill') for s in field_map[field_name]['missing_skills']):
                    field_map[field_name]['missing_skills'].append(skill_obj)
    
    # Convert field map to list and sort by confidence
    return sorted(list(field_map.values()), key=lambda x: x.get('confidence', 0), reverse=True)

# Then in format_frontend_response function, add:
if not formatted_fields and formatted_specs:
    formatted_fields = derive_fields_from_specializations(formatted_specs)
            """)
    except Exception as e:
        print(f"Error testing recommendation system: {e}")

if __name__ == "__main__":
    test_recommendation_system() 