from src.recommender import CareerRecommender
import json
import sys

def get_recommendations(user_skills):
    """Get career recommendations for a given skill set"""
    recommender = CareerRecommender()
    try:
        return recommender.full_recommendation(
            skills=user_skills,
            top_fields=2,
            top_specs=3
        )
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return None

def print_recommendations(recommendations, title):
    """Print recommendations in a readable format"""
    print(f"\n===== {title} =====")
    print("\nTOP FIELDS:")
    for field in recommendations["top_fields"]:
        print(f"- {field['field']} (Confidence: {field['confidence']}%)")
    
    print("\nTOP SPECIALIZATIONS:")
    for spec in recommendations["specializations"]:
        print(f"- {spec['specialization']} (Field: {spec['field']}, Confidence: {spec['confidence']}%)")
        print(f"  Matched Skills: {len(spec['matched_skills'])}/{spec['total_skills_required']}")
        print()

def compare_profiles():
    """Compare recommendations for different skill profiles"""
    profiles = {
        "Software Developer": {
            "Python": 90,
            "JavaScript": 85,
            "Java": 80,
            "Data Structures": 85,
            "Algorithms": 90,
            "Software Design": 80,
            "Testing": 75,
            "Version Control": 85
        },
        "Data Scientist": {
            "Python": 95,
            "R": 85,
            "Machine Learning": 90,
            "Statistics": 90,
            "Data Analysis": 95,
            "Big Data": 80,
            "Data Visualization": 85
        },
        "Web Developer": {
            "HTML": 95,
            "CSS": 95,
            "JavaScript": 95,
            "React": 85,
            "Node.js": 80,
            "Responsive Design": 90,
            "Web Security": 75
        },
        "Business Analyst": {
            "Data Analysis": 85,
            "Excel": 90,
            "SQL": 80,
            "Communication": 90,
            "Critical Thinking": 85,
            "Financial Analysis": 80
        },
        "Healthcare Professional": {
            "Patient Care": 95,
            "Medical Knowledge": 90,
            "Communication": 85,
            "Empathy": 90,
            "Critical Thinking": 85
        }
    }
    
    # Get recommendations for each profile
    results = {}
    for name, skills in profiles.items():
        recommendations = get_recommendations(skills)
        if recommendations:
            print_recommendations(recommendations, f"RECOMMENDATIONS FOR {name.upper()}")
            results[name] = recommendations
    
    return results

if __name__ == "__main__":
    print("Comparing career recommendations for different skill profiles...")
    compare_profiles() 