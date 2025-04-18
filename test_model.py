from src.recommender import CareerRecommender
import argparse
import time
import json
import sys
import os
from utils.semantic_matcher import SemanticMatcher

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

# Test scenarios for the improved version of the career recommender
# This version focuses on testing:
# 1. Skill matching precision across domains
# 2. Missing skills prioritization
# 3. Confidence distribution accuracy
# 4. Cross-domain matching improvements

def print_section(title):
    """Print a section title with formatting"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")

def test_ui_ux_matching():
    """Test improved matching for UI/UX skills to ensure they match to design instead of engineering"""
    print_section("Testing UI/UX Skill Matching Precision")
    
    # Initialize the recommender with semantic matching
    recommender = CareerRecommender(use_semantic=True)
    
    # Test scenario: UI/UX skills should match to design-related fields, not engineering
    user_skills = {
        "User Interface Design": 85,
        "Visual Design": 90,
        "Wireframing": 80,
        "User Experience Research": 75,
        "Prototyping": 85,
        "JavaScript": 70,
        "HTML": 75,
        "CSS": 80,
    }
    
    # Get recommendations
    results = recommender.full_recommendation(user_skills, top_fields=3, top_specs=5)
    
    # Print field results
    print("Field recommendations:")
    for field in results["fields"]:
        print(f"- {field['field']}: {field['confidence']}%")
    
    # Print top specialization
    top_spec = results["specializations"][0]
    print(f"\nTop specialization: {top_spec['specialization']} ({top_spec['confidence']}%)")
    
    # Check all matched skills for the top specialization
    print("\nSkill matches for top specialization:")
    for match in top_spec["matched_skills"]:
        print(f"- {match['required_skill']} matched to {match['user_skill']} ({match['match_score']}%)")
    
    # Verify if UI/UX Design is matched correctly (should not match to Technical Design)
    has_ui_ux_match = False
    has_technical_design_match = False
    
    for spec in results["specializations"]:
        for match in spec["matched_skills"]:
            if "UI" in match["required_skill"] or "UX" in match["required_skill"]:
                has_ui_ux_match = True
                print(f"\nFound UI/UX match: {match['required_skill']} matched to {match['user_skill']}")
            if match["required_skill"] == "Technical Design" and ("UI" in match["user_skill"] or "UX" in match["user_skill"]):
                has_technical_design_match = True
                print(f"\nWARNING: Incorrect match: Technical Design matched to {match['user_skill']}")
    
    if not has_technical_design_match:
        print("\nSUCCESS: No incorrect matches between UI/UX skills and Technical Design")
    else:
        print("\nFAILURE: UI/UX skills were incorrectly matched to Technical Design")

def test_missing_skills_prioritization():
    """Test the prioritization of missing skills based on importance and relevance"""
    print_section("Testing Missing Skills Prioritization")
    
    # Initialize the recommender with semantic matching
    recommender = CareerRecommender(use_semantic=True)
    
    # Test scenario: Software Developer with some skills but missing others
    user_skills = {
        "Python": 90,
        "JavaScript": 85,
        "HTML": 80,
        "CSS": 75,
        "Problem Solving": 85,
        "Git": 70,
    }
    
    # Get recommendations
    results = recommender.full_recommendation(user_skills, top_fields=1, top_specs=3)
    
    # Print top specialization
    top_spec = results["specializations"][0]
    print(f"Top specialization: {top_spec['specialization']} ({top_spec['confidence']}%)")
    
    # Check prioritized missing skills
    print("\nPrioritized missing skills (sorted by priority):")
    for i, missing in enumerate(top_spec["missing_skills"]):
        print(f"{i+1}. {missing['skill']} (Importance: {missing['importance']}, Priority: {missing['priority_score']})")
        if "closest_user_skill" in missing and missing["closest_user_skill"]:
            print(f"   Closest user skill: {missing['closest_user_skill']} (Similarity: {missing['similarity']}%)")
    
    # Test direct prioritization method from SemanticMatcher
    print("\nTesting SemanticMatcher prioritize_missing_skills method directly:")
    
    # Create target skills dictionary (from Software Engineer specialization)
    target_skills = {
        "Java": 80,
        "Data Structures": 90,
        "Algorithms": 85,
        "Software Design": 85,
        "Testing": 80,
        "Version Control": 75
    }
    
    # Get prioritized missing skills
    prioritized = SemanticMatcher.prioritize_missing_skills(user_skills, target_skills, threshold=0.65)
    
    print("\nPrioritized missing skills (from SemanticMatcher):")
    for i, missing in enumerate(prioritized):
        print(f"{i+1}. {missing['skill']} (Priority: {missing['priority_score']:.2f}, Domain: {missing['domain']})")
        if missing['closest_user_skill']:
            print(f"   Closest match: {missing['closest_user_skill']} (Similarity: {missing['similarity']}%)")

def test_confidence_distribution():
    """Test improved confidence distribution across recommendations"""
    print_section("Testing Confidence Distribution")
    
    # Initialize the recommender with semantic matching
    recommender = CareerRecommender(use_semantic=True)
    
    # Test scenario: Game Development skills
    game_dev_skills = {
        "C++": 85,
        "Unity": 90,
        "3D Modeling": 75,
        "Game Design": 80,
        "JavaScript": 70,
        "Problem Solving": 85,
        "Animation": 65,
    }
    
    # Test scenario: Front-end development skills
    frontend_skills = {
        "HTML": 90,
        "CSS": 95,
        "JavaScript": 85,
        "React": 80,
        "UI/UX Design": 75,
        "Responsive Design": 80,
        "Version Control": 70,
    }
    
    # Get recommendations for both scenarios
    game_results = recommender.full_recommendation(game_dev_skills, top_fields=1, top_specs=5)
    frontend_results = recommender.full_recommendation(frontend_skills, top_fields=1, top_specs=5)
    
    # Print confidence distributions
    print("Game Development Confidences:")
    for i, spec in enumerate(game_results["specializations"]):
        print(f"{i+1}. {spec['specialization']}: {spec['confidence']}%")
    
    print("\nFront-end Development Confidences:")
    for i, spec in enumerate(frontend_results["specializations"]):
        print(f"{i+1}. {spec['specialization']}: {spec['confidence']}%")
    
    # Check if the top match for game dev has at least 70% confidence
    top_game_confidence = game_results["specializations"][0]["confidence"]
    if top_game_confidence >= 70:
        print(f"\nSUCCESS: Top game development specialization has good confidence ({top_game_confidence}%)")
    else:
        print(f"\nWARNING: Top game development specialization has low confidence ({top_game_confidence}%)")
    
    # Check the confidence gap between ranks to ensure proper distribution
    game_confidences = [spec["confidence"] for spec in game_results["specializations"]]
    frontend_confidences = [spec["confidence"] for spec in frontend_results["specializations"]]
    
    # Calculate confidence gaps (difference between consecutive rankings)
    if len(game_confidences) >= 2:
        game_gaps = [game_confidences[i] - game_confidences[i+1] for i in range(len(game_confidences)-1)]
        print(f"\nGame development confidence gaps between ranks: {game_gaps}")
    
    if len(frontend_confidences) >= 2:
        frontend_gaps = [frontend_confidences[i] - frontend_confidences[i+1] for i in range(len(frontend_confidences)-1)]
        print(f"Frontend development confidence gaps between ranks: {frontend_gaps}")

def test_cross_domain_matching():
    """Test improved cross-domain matching for skills that span multiple domains"""
    print_section("Testing Cross-Domain Matching")
    
    # Initialize the recommender with semantic matching
    recommender = CareerRecommender(use_semantic=True)
    
    # Test scenario: Mixed skills across design and engineering
    mixed_skills = {
        "Graphic Design": 90,
        "UI/UX Design": 85,
        "CAD": 75,
        "3D Modeling": 80,
        "Product Design": 85,
        "Sketching": 80,
        "Typography": 70,
    }
    
    # Get recommendations
    results = recommender.full_recommendation(mixed_skills, top_fields=3, top_specs=5)
    
    # Print field results
    print("Field recommendations:")
    for field in results["fields"]:
        print(f"- {field['field']}: {field['confidence']}%")
    
    # Print specializations
    print("\nSpecialization recommendations:")
    for i, spec in enumerate(results["specializations"]):
        print(f"{i+1}. {spec['specialization']} ({spec['field']}): {spec['confidence']}%")
    
    # Check matches for graphic design
    print("\nChecking matches for 'Graphic Design':")
    for spec in results["specializations"]:
        for match in spec["matched_skills"]:
            if match["user_skill"] == "Graphic Design":
                print(f"- '{match['required_skill']}' in '{spec['specialization']}' matched to 'Graphic Design' ({match['match_score']}%)")
    
    # Check direct skill matching with SemanticMatcher
    print("\nTesting direct skill matching with SemanticMatcher:")
    design_skills = ["Visual Design", "UI/UX Design", "Graphic Design", "Illustration", "Creative Design"]
    engineering_skills = ["Technical Design", "CAD Design", "Engineering Design", "Industrial Design", "Product Design"]
    
    # Test matching graphic design to both domains
    print("\nMatching 'Graphic Design' to design skills:")
    for skill in design_skills:
        matched, score = SemanticMatcher.match_skill("Graphic Design", [skill], threshold=0.6)
        print(f"- To '{skill}': {'✓' if matched else '✗'} ({score:.2f})")
    
    print("\nMatching 'Graphic Design' to engineering skills:")
    for skill in engineering_skills:
        matched, score = SemanticMatcher.match_skill("Graphic Design", [skill], threshold=0.6)
        print(f"- To '{skill}': {'✓' if matched else '✗'} ({score:.2f})")

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
    
    # Run all tests
    test_ui_ux_matching()
    test_missing_skills_prioritization()
    test_confidence_distribution()
    test_cross_domain_matching()
    
    print("\nAll tests completed.") 