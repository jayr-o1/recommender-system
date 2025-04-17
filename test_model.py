from src.recommender import CareerRecommender

def test_recommendation():
    # Initialize the recommender
    recommender = CareerRecommender()
    
    # Sample user skills for testing - Technical Project Manager profile
    user_skills = {
        "Project Management": 90,
        "Leadership": 85,
        "Communication": 88,
        "Risk Management": 80,
        "Stakeholder Management": 85,
        "Planning": 90,
        "Problem Solving": 92,
        "Budgeting": 75,
        "Time Management": 85,
        "Python": 75,
        "JavaScript": 70,
        "Software Design": 65,
        "Agile Methodologies": 85,
        "Scrum": 80,
        "Testing": 60
    }
    
    # Get recommendations
    try:
        recommendations = recommender.full_recommendation(
            skills=user_skills,
            top_fields=3,
            top_specs=5
        )
        
        print("\n===== CAREER RECOMMENDATIONS FOR TECHNICAL PROJECT MANAGER PROFILE =====")
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
        return None

if __name__ == "__main__":
    test_recommendation() 