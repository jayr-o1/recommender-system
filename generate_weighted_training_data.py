#!/usr/bin/env python3
"""
Generate weighted training data for the career recommender model.
This script creates synthetic training data that includes skill proficiency levels
and applies weights based on skill importance for different specializations.
"""

import os
import sys
import json
import random
import pandas as pd
import numpy as np
from collections import defaultdict

# Add parent directory to path to handle imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility functions
from utils.model_trainer import load_specialization_skills, load_skill_weights

class WeightedTrainingDataGenerator:
    """Generator for weighted training data with proficiency levels."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the generator.
        
        Args:
            output_dir: Directory to save output files
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = output_dir or os.path.join(self.base_dir, "data")
        
        # Load specialization skills and weights
        self.specialization_skills = load_specialization_skills()
        self.skill_weights = load_skill_weights()
        
        # Load sample proficiency data
        sample_data_path = os.path.join(self.base_dir, "data", "sample_proficiency_data.json")
        try:
            with open(sample_data_path, 'r') as f:
                self.sample_proficiency_data = json.load(f)
        except Exception as e:
            print(f"Error loading sample proficiency data: {str(e)}")
            self.sample_proficiency_data = {}
            
    def generate_career_path_data(self, num_samples=1000, output_file=None):
        """
        Generate synthetic career path data with weighted skill requirements.
        
        Args:
            num_samples: Number of synthetic career paths to generate
            output_file: Path to save the data, or None to use default
            
        Returns:
            DataFrame: Generated career path data
        """
        # Prepare output file
        if output_file is None:
            output_file = os.path.join(self.output_dir, "synthetic_career_path_weighted_data.json")
            
        # Create a list of all fields and their specializations
        fields = {}
        for specialization in self.specialization_skills.keys():
            # Determine field based on specialization
            if "Engineer" in specialization or "Developer" in specialization or "Scientist" in specialization:
                field = "Computer Science"
            elif "Manager" in specialization or "Management" in specialization:
                field = "Business"
            elif "Analyst" in specialization and not "Data" in specialization:
                field = "Business"
            elif "Financial" in specialization or "Finance" in specialization or "Investment" in specialization:
                field = "Finance"
            elif "Marketing" in specialization or "Brand" in specialization:
                field = "Marketing"
            elif "HR" in specialization or "Human Resources" in specialization:
                field = "Human Resources"
            else:
                field = "Other"
                
            if field not in fields:
                fields[field] = []
            fields[field].append(specialization)
            
        # Generate synthetic data
        data = []
        
        for _ in range(num_samples):
            # Pick a random field
            field = random.choice(list(fields.keys()))
            
            # Pick a specialization from that field
            specialization = random.choice(fields[field])
            
            # Get required skills for this specialization
            required_skills = self.specialization_skills.get(specialization, [])
            
            # Get weights for these skills
            skill_weights = self.skill_weights.get(specialization, {})
            
            # Define a minimum proficiency level for each skill based on its importance
            min_proficiencies = {}
            for skill in required_skills:
                weight = skill_weights.get(skill, 0.5)
                
                # Higher weight means higher minimum proficiency
                if weight >= 0.9:  # Core skills
                    min_proficiency = random.randint(70, 85)
                elif weight >= 0.7:  # Important skills
                    min_proficiency = random.randint(60, 75)
                elif weight >= 0.4:  # Nice-to-have skills
                    min_proficiency = random.randint(50, 65)
                else:  # Supplementary skills
                    min_proficiency = random.randint(30, 50)
                    
                min_proficiencies[skill] = min_proficiency
                
            # Create a record
            record = {
                'Field': field,
                'Specialization': specialization,
                'Required Skills': ", ".join(required_skills),
                'Experience Level': random.choice(['Entry', 'Mid', 'Senior']),
                'Minimum Proficiency Requirements': json.dumps(min_proficiencies)
            }
            
            data.append(record)
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save to JSON
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Generated career path data saved to {output_file}")
        
        return df
        
    def generate_employee_data(self, num_samples=5000, output_file=None):
        """
        Generate synthetic employee data with skill proficiency levels.
        
        Args:
            num_samples: Number of synthetic employees to generate
            output_file: Path to save the data, or None to use default
            
        Returns:
            DataFrame: Generated employee data
        """
        # Prepare output file
        if output_file is None:
            output_file = os.path.join(self.output_dir, "synthetic_employee_weighted_data.json")
            
        # Prepare fields and specializations
        fields = defaultdict(list)
        
        for specialization in self.specialization_skills.keys():
            # Determine field based on specialization (same logic as above)
            if "Engineer" in specialization or "Developer" in specialization or "Scientist" in specialization:
                field = "Computer Science"
            elif "Manager" in specialization or "Management" in specialization:
                field = "Business"
            elif "Analyst" in specialization and not "Data" in specialization:
                field = "Business"
            elif "Financial" in specialization or "Finance" in specialization or "Investment" in specialization:
                field = "Finance"
            elif "Marketing" in specialization or "Brand" in specialization:
                field = "Marketing"
            elif "HR" in specialization or "Human Resources" in specialization:
                field = "Human Resources"
            else:
                field = "Other"
                
            fields[field].append(specialization)
            
        # Generate synthetic data
        data = []
        
        for i in range(num_samples):
            # Pick a random field and specialization
            field = random.choice(list(fields.keys()))
            specialization = random.choice(fields[field])
            
            # Get specialization skills and weights
            spec_skills = self.specialization_skills.get(specialization, [])
            spec_weights = self.skill_weights.get(specialization, {})
            
            # Determine how many skills this employee has (more experience = more skills)
            experience_years = random.randint(1, 15)
            num_skills = min(max(5, int(experience_years * 1.5)), len(spec_skills))
            
            # For senior employees, bias towards higher-weighted skills
            if experience_years >= 10:
                # Sort skills by weight
                weighted_skills = [(skill, spec_weights.get(skill, 0.5)) for skill in spec_skills]
                weighted_skills.sort(key=lambda x: x[1], reverse=True)
                
                # Pick more core skills
                selected_skills = [s[0] for s in weighted_skills[:num_skills]]
            else:
                # Randomly select skills
                selected_skills = random.sample(spec_skills, num_skills)
                
            # Assign proficiency levels
            skill_proficiencies = {}
            for skill in selected_skills:
                # Higher experience and skill weight leads to higher proficiency
                weight = spec_weights.get(skill, 0.5)
                base_proficiency = 30 + experience_years * 4  # 30-90 based on experience
                
                # Weight adjustment (+0 to +20 based on weight)
                weight_factor = int(weight * 20)
                
                # Randomize a bit
                random_factor = random.randint(-10, 10)
                
                # Calculate final proficiency (capped at 100)
                proficiency = min(base_proficiency + weight_factor + random_factor, 100)
                
                skill_proficiencies[skill] = proficiency
                
            # Create employee record
            record = {
                'Employee ID': f'EMP{i+1:05d}',
                'Age': random.randint(22, 60),
                'Years Experience': experience_years,
                'Skills': ", ".join(selected_skills),
                'Skill Proficiencies': json.dumps(skill_proficiencies),
                'Field': field,
                'Specialization': specialization,
                'Career Goal': random.choice(fields[field])  # Random goal within same field
            }
            
            data.append(record)
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save to JSON
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Generated employee data saved to {output_file}")
        
        return df
        
    def generate_proficiency_test_data(self, num_samples=10, output_file=None):
        """
        Generate test data specifically for proficiency-based matching testing.
        
        Args:
            num_samples: Number of test profiles per specialization
            output_file: Path to save the data, or None to use default
            
        Returns:
            dict: Generated test data
        """
        # Prepare output file
        if output_file is None:
            output_file = os.path.join(self.output_dir, "proficiency_test_data.json")
            
        # Select a subset of specializations for testing
        test_specializations = [
            "Software Engineer",
            "Data Scientist",
            "Machine Learning Engineer",
            "Project Manager",
            "Financial Analyst",
            "Marketing Specialist"
        ]
        
        # Filter to make sure all specializations exist in our data
        test_specializations = [s for s in test_specializations if s in self.specialization_skills]
        
        test_data = {}
        
        for specialization in test_specializations:
            test_data[specialization] = {}
            spec_skills = self.specialization_skills.get(specialization, [])
            spec_weights = self.skill_weights.get(specialization, {})
            
            # Generate test profiles for each specialization
            for skill_level in ["beginner", "intermediate", "advanced", "expert"]:
                # Create examples for each skill level
                profiles = []
                
                for i in range(num_samples):
                    profile = {}
                    
                    # Determine how many skills to include and at what proficiency
                    if skill_level == "beginner":
                        skill_count = int(len(spec_skills) * 0.3)  # 30% of skills
                        base_proficiency = 35  # Average proficiency
                        variance = 15  # Variance in proficiency
                    elif skill_level == "intermediate":
                        skill_count = int(len(spec_skills) * 0.5)  # 50% of skills
                        base_proficiency = 55
                        variance = 15
                    elif skill_level == "advanced":
                        skill_count = int(len(spec_skills) * 0.7)  # 70% of skills
                        base_proficiency = 75
                        variance = 15
                    else:  # expert
                        skill_count = int(len(spec_skills) * 0.9)  # 90% of skills
                        base_proficiency = 90
                        variance = 10
                        
                    # Ensure at least 3 skills
                    skill_count = max(skill_count, 3)
                    
                    # Prioritize higher-weighted skills for selection
                    weighted_skills = [(skill, spec_weights.get(skill, 0.5)) for skill in spec_skills]
                    weighted_skills.sort(key=lambda x: x[1] + random.random() * 0.3, reverse=True)
                    
                    # Select top skills based on weight plus some randomness
                    selected_skills = [s[0] for s in weighted_skills[:skill_count]]
                    
                    # Assign proficiency levels
                    for skill in selected_skills:
                        # Weight affects proficiency variance
                        weight = spec_weights.get(skill, 0.5)
                        
                        # Higher weights get better proficiency for that skill
                        weight_bonus = int(weight * 15)  # 0-15 bonus based on weight
                        
                        # Add randomness
                        proficiency = base_proficiency + weight_bonus + random.randint(-variance, variance)
                        
                        # Cap between 10 and 100
                        proficiency = max(min(proficiency, 100), 10)
                        
                        profile[skill] = proficiency
                        
                    # Add some unrelated skills
                    other_skills = 0
                    if skill_level == "beginner":
                        other_skills = 1
                    elif skill_level == "intermediate":
                        other_skills = 2
                    elif skill_level in ["advanced", "expert"]:
                        other_skills = 3
                        
                    # Find other skills from different specializations
                    all_skills = set()
                    for s, skills in self.specialization_skills.items():
                        if s != specialization:
                            all_skills.update(skills)
                    
                    all_skills = all_skills - set(selected_skills)
                    
                    if other_skills > 0 and all_skills:
                        unrelated_skills = random.sample(list(all_skills), min(other_skills, len(all_skills)))
                        for skill in unrelated_skills:
                            # Unrelated skills typically have lower proficiency
                            proficiency = random.randint(30, 70)
                            profile[skill] = proficiency
                    
                    profiles.append({
                        "profile_id": f"{specialization.replace(' ', '_')}_{skill_level}_{i+1}",
                        "skills": profile
                    })
                    
                test_data[specialization][skill_level] = profiles
                
        # Save to JSON
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"Generated proficiency test data saved to {output_file}")
        
        return test_data
        
def main():
    """Run the data generation process."""
    print("=== Weighted Training Data Generator ===")
    
    # Create generator
    generator = WeightedTrainingDataGenerator()
    
    # Generate career path data
    print("\nGenerating career path data...")
    career_path_data = generator.generate_career_path_data(num_samples=200)
    
    # Generate employee data
    print("\nGenerating employee data...")
    employee_data = generator.generate_employee_data(num_samples=1000)
    
    # Generate test data
    print("\nGenerating proficiency test data...")
    test_data = generator.generate_proficiency_test_data()
    
    print("\nData generation complete!")
    print(f"Career paths: {len(career_path_data)} records")
    print(f"Employees: {len(employee_data)} records")
    
if __name__ == "__main__":
    main() 