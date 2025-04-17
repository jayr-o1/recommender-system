#!/usr/bin/env python3
"""
User-friendly command-line utility to get career recommendations based on skills and proficiency levels.
"""

import os
import sys
import argparse
from consolidated_workflow import get_recommendations_from_consolidated, consolidate_all_data

def prompt_for_skills():
    """
    Interactive prompt for user to enter skills and proficiency levels.
    
    Returns:
        str: Formatted skill-proficiency string
    """
    print("\n" + "="*60)
    print("     CAREER RECOMMENDATION SYSTEM - SKILL ASSESSMENT")
    print("="*60)
    print("\nWelcome to the UnitechHR Career Recommendation System!")
    print("This tool will help you discover career paths that match your skills.")
    print("\nPlease enter your skills and rate your proficiency for each one.")
    print("For each skill, you'll be asked to rate your proficiency from 0-100%:")
    print("  0-30%:  Beginner level")
    print("  31-60%: Intermediate level")
    print("  61-85%: Advanced level")
    print("  86-100%: Expert level")
    print("\nEnter 'done' when you're finished adding skills.")
    
    skills = []
    
    while True:
        skill = input("\nEnter a skill (or 'done' to finish): ").strip()
        
        if skill.lower() == 'done':
            break
            
        if not skill:
            continue
            
        while True:
            try:
                proficiency = input(f"Rate your proficiency in {skill} (0-100): ").strip()
                proficiency = int(proficiency)
                
                if 0 <= proficiency <= 100:
                    # Add description of level
                    level = "Beginner"
                    if proficiency > 30:
                        level = "Intermediate"
                    if proficiency > 60:
                        level = "Advanced"
                    if proficiency > 85:
                        level = "Expert"
                        
                    print(f"Added: {skill} - {level} level ({proficiency}%)")
                    break
                else:
                    print("Proficiency must be between 0 and 100. Please try again.")
            except ValueError:
                print("Please enter a valid number between 0 and 100.")
        
        skills.append(f"{skill} {proficiency}")
    
    if not skills:
        print("\nYou didn't enter any skills. Using example skills...")
        skills = [
            "Python 85",
            "Machine Learning 75", 
            "Data Analysis 80",
            "SQL 90"
        ]
        print("Using example skills:")
        for skill in skills:
            parts = skill.split()
            s = " ".join(parts[:-1])
            p = int(parts[-1])
            print(f"  - {s} ({p}%)")
    
    return "\n".join(skills)

def save_skills_to_file(skills_input):
    """
    Save skills input to a file for future reference.
    
    Args:
        skills_input (str): Formatted skill-proficiency string
        
    Returns:
        str: Path to the saved file
    """
    # Create a save directory if it doesn't exist
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data")
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"skills_{timestamp}.txt"
    file_path = os.path.join(save_dir, filename)
    
    # Save to file
    with open(file_path, 'w') as f:
        f.write(skills_input)
    
    return file_path

def main():
    """Main function to run the interactive career recommendation utility."""
    
    print("\n" + "="*60)
    print("            UNITECH HR CAREER RECOMMENDATION SYSTEM")
    print("="*60)
    
    parser = argparse.ArgumentParser(description='Interactive Career Recommendation Tool')
    parser.add_argument('--file', type=str, default=None,
                        help='Load skills from a file instead of interactive prompt')
    parser.add_argument('--save', action='store_true',
                        help='Save entered skills to a file for future reference')
    
    args = parser.parse_args()
    
    # Get skills input either from file or interactive prompt
    if args.file:
        try:
            with open(args.file, 'r') as f:
                skills_input = f.read()
            print(f"Loaded skills from {args.file}")
            
            # Show the loaded skills
            print("\nSkills loaded:")
            for line in skills_input.strip().split('\n'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        skill = ' '.join(parts[:-1])
                        prof = int(parts[-1])
                        print(f"  - {skill} ({prof}%)")
                    except ValueError:
                        print(f"  - {line}")
        except Exception as e:
            print(f"Error loading skills file: {str(e)}")
            print("Falling back to interactive prompt.")
            skills_input = prompt_for_skills()
    else:
        skills_input = prompt_for_skills()
    
    # Save skills if requested
    if args.save:
        saved_file = save_skills_to_file(skills_input)
        print(f"\nYour skills profile has been saved to: {saved_file}")
        print(f"You can reuse it later with: --file {saved_file}")
    
    # Consolidate data and get recommendations
    print("\nAnalyzing your skills and generating recommendations...")
    consolidated_data = consolidate_all_data()
    get_recommendations_from_consolidated(skills_input, consolidated_data)
    
    # Closing message
    print("\nThank you for using the Career Recommendation System!")
    print("For more details or to re-run with different skills, use:")
    print("  python get_career_recommendation.py")
    print("  python get_career_recommendation.py --file YOUR_SKILLS_FILE")
    print("="*60)

if __name__ == "__main__":
    main() 