from src.recommender import CareerRecommender

def test_recommendation():
    # Initialize the recommender
    recommender = CareerRecommender()
    
    # Sample user skills for testing
    user_skills = {
        "Python": 90,
        "JavaScript": 85,
        "Data Analysis": 80,
        "Machine Learning": 75,
        "Problem Solving": 95
    }
    
    # Get recommendations
    try:
        recommendations = recommender.full_recommendation(
            skills=user_skills,
            top_fields=2,
            top_specs=3
        )
        
        print("\n===== CAREER RECOMMENDATIONS =====")
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