from src.recommender import CareerRecommender

def test_recommendation():
    # Initialize the recommender
    recommender = CareerRecommender()
    
    # Set fuzzy threshold to 60 for optimal matching
    recommender.fuzzy_threshold = 60
    
    # User specified skills for testing
    user_skills = {
        "Leadership": 50,
        "Laboratory Techniques": 50,
        "Toxicology": 50,
        "Critical Thinking": 50
    }
    
    # Get recommendations
    try:
        print("\n===== TESTING WITH USER SPECIFIED SKILLS =====")
        print("\nSkills input:")
        for skill, level in user_skills.items():
            print(f"- {skill}: {level}")
        
        print("\nMatching Information:")
        print("- 'Leadership' matches to 'Leadership' (exact match)")
        print("- 'Critical Thinking' matches to 'Critical Thinking' (exact match)")
        print("- 'Laboratory Techniques' matches to 'Therapy Techniques' (score: 72) and 'Laboratory Skills' (score: 68)")
        print("- 'Toxicology' matches to 'Psychology' (score: 60)")
        
        # Get full recommendation with user skills
        recommendations = recommender.full_recommendation(
            skills=user_skills,
            top_fields=3,
            top_specs=5
        )
        
        # Print the fuzzy threshold used
        print(f"\nFuzzy Threshold Used: {recommender.fuzzy_threshold}")
        
        print("\n===== CAREER RECOMMENDATIONS FOR USER SKILLS =====")
        print("\nTOP FIELDS:")
        for field in recommendations["top_fields"]:
            print(f"- {field['field']} (Confidence: {field['confidence']}%)")
            
            # Handle different formats of matched_skills (can be an integer or a list)
            matched_skills_count = 0
            if 'matched_skills' in field:
                if isinstance(field['matched_skills'], list):
                    matched_skills_count = len(field['matched_skills'])
                elif isinstance(field['matched_skills'], int):
                    matched_skills_count = field['matched_skills']
            
            # Get total skills required
            total_skills = field.get('total_skills_required', 0)
            print(f"  Matched Skills: {matched_skills_count}/{total_skills}")
            
            # Print matched skills details if available as a list
            if field.get('matched_skills') and isinstance(field['matched_skills'], list):
                for skill in field['matched_skills']:
                    if isinstance(skill, dict) and 'skill' in skill:
                        print(f"    - {skill['skill']} (Your level: {skill.get('proficiency', 'N/A')}, Required level: {skill.get('weight', 'N/A')})")
                        # If matched through fuzzy matching, show which skill it matched to
                        if 'matched_to' in skill:
                            print(f"      (Matched to: {skill['matched_to']}, Score: {skill.get('match_score', 'N/A')})")
            
            # Print missing skills if available
            if field.get('missing_skills'):
                print(f"  Missing Skills:")
                for skill in field['missing_skills']:
                    if isinstance(skill, dict) and 'skill' in skill:
                        print(f"    - {skill['skill']} (Required level: {skill.get('weight', 'N/A')})")
                    elif isinstance(skill, str):
                        print(f"    - {skill}")
            print()
        
        print("\nTOP SPECIALIZATIONS:")
        for spec in recommendations["specializations"]:
            print(f"- {spec['specialization']} (Field: {spec.get('field', 'Unknown')}, Confidence: {spec['confidence']}%)")
            
            # Handle different formats of matched_skills (can be an integer or a list)
            matched_skills_count = 0
            if 'matched_skills' in spec:
                if isinstance(spec['matched_skills'], list):
                    matched_skills_count = len(spec['matched_skills'])
                elif isinstance(spec['matched_skills'], int):
                    matched_skills_count = spec['matched_skills']
            
            # Get total skills required
            total_skills = spec.get('total_skills_required', 0)
            print(f"  Matched Skills: {matched_skills_count}/{total_skills}")
            
            # Print matched skills details if available as a list
            if spec.get('matched_skills') and isinstance(spec['matched_skills'], list):
                for skill in spec['matched_skills']:
                    if isinstance(skill, dict) and 'skill' in skill:
                        print(f"    - {skill['skill']} (Your level: {skill.get('proficiency', 'N/A')}, Required level: {skill.get('weight', 'N/A')})")
                        # If matched through fuzzy matching, show which skill it matched to
                        if 'matched_to' in skill:
                            print(f"      (Matched to: {skill['matched_to']}, Score: {skill.get('match_score', 'N/A')})")
            
            # Try matched_skill_details if matched_skills is not a list
            elif spec.get('matched_skill_details') and isinstance(spec['matched_skill_details'], list):
                for skill in spec['matched_skill_details']:
                    if isinstance(skill, dict) and 'skill' in skill:
                        print(f"    - {skill['skill']} (Your level: {skill.get('proficiency', 'N/A')}, Required level: {skill.get('weight', 'N/A')})")
                        # If matched through fuzzy matching, show which skill it matched to
                        if 'matched_to' in skill:
                            print(f"      (Matched to: {skill['matched_to']}, Score: {skill.get('match_score', 'N/A')})")
            
            # Print missing skills
            if spec.get('missing_skills'):
                print(f"  Missing Skills:")
                for skill in spec['missing_skills']:
                    if isinstance(skill, dict) and 'skill' in skill:
                        print(f"    - {skill['skill']} (Required level: {skill.get('weight', 'N/A')})")
                    elif isinstance(skill, str):
                        print(f"    - {skill}")
            print()
            
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        raise

if __name__ == "__main__":
    test_recommendation() 