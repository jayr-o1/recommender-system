from src.recommender import CareerRecommender
import argparse
import time
import json

def test_recommendation(skills=None, use_semantic=True, fuzzy_threshold=65, top_fields=3, top_specs=5):
    """
    Test the recommender with given skills and parameters
    
    Args:
        skills: Dictionary of skills and proficiency (or None to use default test set)
        use_semantic: Whether to use semantic matching
        fuzzy_threshold: Fuzzy matching threshold
        top_fields: Number of top fields to return
        top_specs: Number of top specializations to return
    """
    # Initialize the recommender
    recommender = CareerRecommender(fuzzy_threshold=fuzzy_threshold, use_semantic=use_semantic)
    
    # Default test skills if none provided
    if skills is None:
        skills = {
            "Leadership": 80,
            "Laboratory Techniques": 75,
            "Toxicology": 70,
            "Critical Thinking": 85,
            "Python Coding Experience": 90,
            "Data Visualization": 80,
            "Chemical Compound Analysis": 65,
            "Deep Neural Network Design": 75,
            "Project Team Management": 80
        }
    
    # Get recommendations
    try:
        start_time = time.time()
        
        print("\n===== TESTING CAREER RECOMMENDER =====")
        print(f"Semantic Matching: {'Enabled' if use_semantic else 'Disabled'}")
        print(f"Fuzzy Threshold: {fuzzy_threshold}")
        
        print("\nSkills input:")
        for skill, level in skills.items():
            print(f"- {skill}: {level}")
        
        # Get full recommendation with user skills
        recommendations = recommender.full_recommendation(
            skills=skills,
            top_fields=top_fields,
            top_specs=top_specs
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"\nProcessing time: {processing_time:.2f} seconds")
        
        print("\n===== CAREER RECOMMENDATIONS =====")
        print("\nTOP FIELDS:")
        for i, field in enumerate(recommendations["fields"], 1):
            print(f"{i}. {field['field']} (Confidence: {field['confidence']:.1f}%)")
            if field.get('description'):
                print(f"   Description: {field['description']}")
                
        print("\nTOP SPECIALIZATIONS:")
        for i, spec in enumerate(recommendations["specializations"], 1):
            print(f"{i}. {spec['specialization']} (Field: {spec.get('field', 'Unknown')}, Confidence: {spec['confidence'] * 100:.1f}%)")
            
            if spec.get('description'):
                print(f"   Description: {spec['description']}")
            
            # Print matched skills details with semantic matching information
            if spec.get('matched_skill_details') and isinstance(spec['matched_skill_details'], list):
                print(f"   Matched Skills ({len(spec['matched_skill_details'])} skills):")
                for skill in spec['matched_skill_details']:
                    if isinstance(skill, dict) and 'skill' in skill:
                        print(f"   - {skill['skill']} (Required level: {skill.get('weight', 'N/A')})")
                        print(f"     → Matched to: {skill.get('matched_to', 'N/A')} (Your level: {skill.get('proficiency', 'N/A')})")
                        print(f"     → Match score: {skill.get('match_score', 'N/A')}, Importance: {skill.get('importance', 1.0)}")
            
            # Print missing skills
            if spec.get('missing_skills'):
                print(f"   Missing Skills ({len(spec['missing_skills'])} skills):")
                for skill in spec['missing_skills']:
                    if isinstance(skill, dict) and 'skill' in skill:
                        print(f"   - {skill['skill']} (Required level: {skill.get('weight', 'N/A')})")
                    elif isinstance(skill, str):
                        print(f"   - {skill}")
            print()
            
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        raise

def compare_matching_methods(test_cases=None, fuzzy_threshold=65):
    """
    Compare fuzzy matching vs semantic matching on challenging examples
    
    Args:
        test_cases: List of (user_skill, standard_skill, difficulty) tuples
        fuzzy_threshold: Threshold for fuzzy matching
    """
    print("\n===== COMPARING MATCHING METHODS =====")
    
    # Create two recommenders - one with semantic matching, one without
    fuzzy_recommender = CareerRecommender(fuzzy_threshold=fuzzy_threshold, use_semantic=False)
    semantic_recommender = CareerRecommender(fuzzy_threshold=fuzzy_threshold, use_semantic=True)
    
    # Default test cases if none provided
    if test_cases is None:
        test_cases = [
            ("Python Coding", "Python", "Simple"),
            ("Javascript Development", "JavaScript", "Simple"),
            ("Statistical Analysis", "Statistics", "Medium"),
            ("UI Development", "User Interface Design", "Medium"),
            ("Deep Learning Models", "Neural Networks", "Hard"),
            ("HPLC Analysis Methods", "Chromatography", "Hard"),
            ("Synthesizing Organic Compounds", "Organic Chemistry", "Specialized"),
            ("Managing Cross-Functional Teams", "Team Management", "Specialized"),
        ]
    
    print(f"{'User Skill':<35} {'Standard Skill':<25} {'Fuzzy Match':<15} {'Semantic Match':<15}")
    print("-" * 90)
    
    semantic_wins = 0
    fuzzy_wins = 0
    ties = 0
    
    for user_skill, standard_skill, difficulty in test_cases:
        # Test fuzzy matching
        is_fuzzy_match, fuzzy_score = fuzzy_recommender._match_skill_improved(user_skill, standard_skill)
        
        # Test semantic matching
        is_semantic_match, semantic_score = semantic_recommender._match_skill_improved(user_skill, standard_skill)
        
        # Determine winner
        if is_semantic_match and not is_fuzzy_match:
            semantic_wins += 1
            winner = "Semantic"
        elif is_fuzzy_match and not is_semantic_match:
            fuzzy_wins += 1
            winner = "Fuzzy"
        elif is_semantic_match and is_fuzzy_match:
            if semantic_score > fuzzy_score:
                semantic_wins += 1
                winner = "Semantic"
            elif fuzzy_score > semantic_score:
                fuzzy_wins += 1
                winner = "Fuzzy"
            else:
                ties += 1
                winner = "Tie"
        else:
            ties += 1
            winner = "None"
        
        # Print results
        print(f"{user_skill:<35} {standard_skill:<25} "
              f"{'✓' if is_fuzzy_match else '✗'} ({fuzzy_score}%)      "
              f"{'✓' if is_semantic_match else '✗'} ({semantic_score}%)  [{winner}]")
    
    # Print summary
    print("\nSummary:")
    print(f"- Semantic wins: {semantic_wins}/{len(test_cases)} ({semantic_wins/len(test_cases)*100:.1f}%)")
    print(f"- Fuzzy wins: {fuzzy_wins}/{len(test_cases)} ({fuzzy_wins/len(test_cases)*100:.1f}%)")
    print(f"- Ties: {ties}/{len(test_cases)} ({ties/len(test_cases)*100:.1f}%)")
    print("\nNote: ✓ means the skill was matched successfully, ✗ means it failed to match")

def parse_custom_skills(skills_string):
    """Parse a string of skills into a dictionary"""
    if not skills_string:
        return None
        
    skills = {}
    parts = skills_string.split(',')
    
    for part in parts:
        part = part.strip()
        if ':' in part:
            skill, prof = part.split(':', 1)
            try:
                skills[skill.strip()] = int(prof.strip())
            except ValueError:
                skills[skill.strip()] = 70  # Default if not a valid number
        else:
            skills[part] = 70  # Default proficiency
            
    return skills

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the career recommender model')
    parser.add_argument('--skills', type=str, help='Comma-separated list of skills with optional proficiency (e.g., "Python:90,Data Analysis:85")')
    parser.add_argument('--no-semantic', action='store_true', help='Disable semantic matching')
    parser.add_argument('--threshold', type=int, default=65, help='Fuzzy matching threshold (0-100)')
    parser.add_argument('--top-fields', type=int, default=3, help='Number of top fields to return')
    parser.add_argument('--top-specs', type=int, default=5, help='Number of top specializations to return')
    parser.add_argument('--compare-only', action='store_true', help='Only run the matching methods comparison')
    
    args = parser.parse_args()
    
    # Parse custom skills if provided
    custom_skills = parse_custom_skills(args.skills)
    
    # Run tests based on arguments
    if not args.compare_only:
        test_recommendation(
            skills=custom_skills,
            use_semantic=not args.no_semantic,
            fuzzy_threshold=args.threshold,
            top_fields=args.top_fields,
            top_specs=args.top_specs
        )
    
    compare_matching_methods(fuzzy_threshold=args.threshold) 