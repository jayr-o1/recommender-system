from src.recommender import CareerRecommender

def test_recommendation():
    # Initialize the recommender
    recommender = CareerRecommender()
    
    # Sample user skills for testing with non-existent skills
    # Many of these are variations or misspellings of existing skills
    user_skills = {
        "Projectt Managementt": 90,     # Misspelling of "Project Management"
        "Team Leadership": 85,          # Similar to "Leadership"
        "Verbal Communications": 88,     # Variation of "Communication"
        "Risk Assessment": 82,          # Similar to "Risk Management"
        "Client Relations": 85,         # New skill, not in the system
        "Project Planning": 90,         # Similar to "Planning"
        "Python Coding": 75,            # Variation of "Python"
        "Javascript Programming": 70,    # Variation of "JavaScript"
        "Agile Method": 85,             # Similar to "Agile Methodologies"
        "Scrum Master": 80,             # Similar to "Scrum"
        "Automated Testing": 60         # Similar to "Testing"
    }
    
    # Get recommendations
    try:
        print("\n===== TESTING WITH NON-EXISTENT SKILLS =====")
        print("\nOriginal skills input:")
        for skill, level in user_skills.items():
            print(f"- {skill}: {level}")
            
        # Using the skill processor directly with a lower matching threshold
        from src.cli import RecommenderCLI
        cli = RecommenderCLI()
        
        # Get all known skills from skill weights
        reference_skills = list(recommender.skill_weights.keys())
        
        # Try different thresholds for fuzzy matching
        thresholds = [0.8, 0.7, 0.6, 0.5]
        for threshold in thresholds:
            print(f"\n--- Fuzzy matching with threshold: {threshold} ---")
            # Create a dictionary to hold matched skills for this threshold
            matched_at_threshold = {}
            unmatched_at_threshold = []
            
            # Process each skill individually to see the matches
            for skill, level in user_skills.items():
                # Convert to tuple for caching
                reference_skills_tuple = tuple(reference_skills)
                # Try to match the skill
                matched = cli.skill_processor.match_skill(skill, reference_skills_tuple, threshold)
                if matched:
                    matched_at_threshold[matched] = level
                    print(f"'{skill}' matched to '{matched}'")
                else:
                    unmatched_at_threshold.append(skill)
            
            print(f"\nMatched {len(matched_at_threshold)} out of {len(user_skills)} skills")
            if unmatched_at_threshold:
                print(f"Unmatched skills: {', '.join(unmatched_at_threshold)}")
        
        # Use threshold of 0.6 for our actual recommendation test
        print("\n\n===== TESTING RECOMMENDATIONS WITH THRESHOLD 0.6 =====")
        # Process skills with a threshold of 0.6
        matched_skills = {}
        for skill, level in user_skills.items():
            # Try to match the skill
            matched = cli.skill_processor.match_skill(skill, tuple(reference_skills), 0.6)
            if matched:
                matched_skills[matched] = level
        
        # Get full recommendation with matched skills
        recommendations = recommender.full_recommendation(
            skills=matched_skills,
            top_fields=3,
            top_specs=3
        )
        
        print("\n===== CAREER RECOMMENDATIONS WITH FUZZY-MATCHED SKILLS =====")
        print("\nTOP FIELDS:")
        for field in recommendations["top_fields"]:
            print(f"- {field['field']} (Confidence: {field['confidence']}%)")
        
        print("\nTOP SPECIALIZATIONS:")
        for spec in recommendations["specializations"]:
            print(f"- {spec['specialization']} (Field: {spec['field']}, Confidence: {spec['confidence']}%)")
            print(f"  Matched Skills: {len(spec['matched_skills'])}/{spec['total_skills_required']}")
            for skill in spec['matched_skills']:
                print(f"    - {skill['skill']} (Your level: {skill['proficiency']}, Required level: {skill['weight']})")
            if spec['missing_skills']:
                print(f"  Missing Skills:")
                for skill in spec['missing_skills']:
                    print(f"    - {skill['skill']} (Required level: {skill['weight']})")
            print()
            
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        raise

if __name__ == "__main__":
    test_recommendation() 