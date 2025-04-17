"""
Specialization Skills Management Script

This script provides a command-line interface for managing the specialization-specific skills
used by the career recommendation system to identify missing skills for specific career paths.
"""

import sys
import os
import json
from utils.data_manager import (
    load_specialization_skills,
    save_specialization_skills,
    add_specialization_skills,
    get_specialization_skills,
    remove_specialization,
    add_bulk_specializations,
    get_all_specializations
)

def main():
    """
    Main function for the specialization skills management script.
    """
    # Print banner
    print("\n===== Specialization Skills Manager =====\n")
    
    # Handle command line arguments
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        # List all current specializations
        specializations = get_all_specializations()
        if specializations:
            print("Current specializations in the data file:")
            for spec in sorted(specializations):
                print(f"- {spec}")
            print(f"\nTotal: {len(specializations)} specializations")
        else:
            print("No specializations found in the data file.")
    
    elif command == "view":
        if len(sys.argv) < 3:
            print("Error: Missing specialization name")
            print_usage()
            return
        
        specialization = sys.argv[2]
        skills = get_specialization_skills(specialization)
        
        if skills:
            print(f"\nSkills for {specialization}:")
            for skill in sorted(skills):
                print(f"- {skill}")
            print(f"\nTotal: {len(skills)} skills")
        else:
            print(f"No skills found for specialization: {specialization}")
    
    elif command == "add":
        if len(sys.argv) < 3:
            print("Error: Missing specialization name")
            print_usage()
            return
        
        specialization = sys.argv[2]
        
        # Interactive mode to enter skills
        print(f"Adding skills for: {specialization}")
        print("Enter skills one per line. Enter an empty line when done.")
        
        skills = []
        while True:
            skill = input("> ")
            if not skill:
                break
            skills.append(skill)
        
        if not skills:
            print("No skills entered. Operation cancelled.")
            return
        
        success = add_specialization_skills(specialization, skills)
        if success:
            print(f"Successfully added {len(skills)} skills for {specialization}")
        else:
            print("Failed to add skills. See error log for details.")
    
    elif command == "bulk":
        if len(sys.argv) < 3:
            print("Error: Missing JSON file path")
            print_usage()
            return
        
        json_file = sys.argv[2]
        if not os.path.exists(json_file):
            print(f"Error: File not found: {json_file}")
            return
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                print("Error: JSON file must contain a dictionary mapping specializations to skill lists")
                return
            
            success = add_bulk_specializations(data)
            if success:
                print(f"Successfully added/updated {len(data)} specializations")
            else:
                print("Failed to add specializations. See error log for details.")
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in file")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    elif command == "remove":
        if len(sys.argv) < 3:
            print("Error: Missing specialization name")
            print_usage()
            return
        
        specialization = sys.argv[2]
        
        # Confirm deletion
        confirm = input(f"Are you sure you want to remove '{specialization}'? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
        
        success = remove_specialization(specialization)
        if success:
            print(f"Successfully removed specialization: {specialization}")
        else:
            print("Failed to remove specialization. See error log for details.")
    
    elif command == "export":
        if len(sys.argv) < 3:
            print("Error: Missing output file path")
            print_usage()
            return
        
        output_file = sys.argv[2]
        
        # Get all skills data
        skills_data = load_specialization_skills()
        
        try:
            with open(output_file, 'w') as f:
                json.dump(skills_data, f, indent=2)
            print(f"Successfully exported {len(skills_data)} specializations to {output_file}")
        except Exception as e:
            print(f"Error exporting skills data: {str(e)}")
    
    else:
        print(f"Unknown command: {command}")
        print_usage()

def print_usage():
    """Print usage information for the script."""
    print("\nUsage:")
    print("  python manage_skills.py list")
    print("      - List all specializations in the data file")
    print("  python manage_skills.py view SPECIALIZATION")
    print("      - View skills for a specific specialization")
    print("  python manage_skills.py add SPECIALIZATION")
    print("      - Add skills for a specialization interactively")
    print("  python manage_skills.py bulk FILE.json")
    print("      - Add multiple specializations from a JSON file")
    print("  python manage_skills.py remove SPECIALIZATION")
    print("      - Remove a specialization from the data file")
    print("  python manage_skills.py export FILE.json")
    print("      - Export all specialization skills to a JSON file")
    print("\nExamples:")
    print("  python manage_skills.py list")
    print("  python manage_skills.py view \"Data Scientist\"")
    print("  python manage_skills.py add \"Web Developer\"")
    print("  python manage_skills.py bulk new_specializations.json")
    print("  python manage_skills.py remove \"Outdated Role\"")
    print("  python manage_skills.py export backup_skills.json")

if __name__ == "__main__":
    main() 