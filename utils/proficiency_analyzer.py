#!/usr/bin/env python3
"""
Utility for analyzing skill proficiency levels and providing enhanced matching.
This module provides functions to process skill proficiency data and generate
insights for career recommendations.
"""

import os
import sys
import json
import difflib
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent directory to path to handle imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
from utils.model_trainer import calculate_skill_match_percentage, load_specialization_skills, load_skill_weights

class ProficiencyAnalyzer:
    """Analyzer for skill proficiency and matching."""
    
    def __init__(self, sample_data_path=None, skill_weights_path=None, specialization_skills_path=None):
        """
        Initialize the proficiency analyzer with data paths.
        
        Args:
            sample_data_path: Path to sample proficiency data
            skill_weights_path: Path to skill weights data
            specialization_skills_path: Path to specialization skills data
        """
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set default paths if not provided
        if not sample_data_path:
            sample_data_path = os.path.join(self.base_dir, "data", "sample_proficiency_data.json")
        if not skill_weights_path:
            skill_weights_path = os.path.join(self.base_dir, "data", "skill_weights.json")
        if not specialization_skills_path:
            specialization_skills_path = os.path.join(self.base_dir, "data", "specialization_skills.json")
            
        # Load data
        self.sample_data = self._load_json(sample_data_path)
        self.skill_weights = load_skill_weights()
        self.specialization_skills = load_specialization_skills()
        
        # Load metadata
        metadata_path = os.path.join(self.base_dir, "data", "skill_weights_metadata.json")
        self.metadata = self._load_json(metadata_path).get('metadata', {})
        
    def _load_json(self, path):
        """Load JSON data from a file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
            return {}
            
    def analyze_user_skills(self, user_id, specializations=None, top_n=5):
        """
        Analyze a user's skills against all or specific specializations.
        
        Args:
            user_id: User ID from sample data
            specializations: List of specializations to check, or None for all
            top_n: Number of top specializations to return
            
        Returns:
            dict: Analysis results including top matches and detailed metrics
        """
        if user_id not in self.sample_data:
            return {"error": f"User {user_id} not found in sample data"}
            
        user_skills = self.sample_data[user_id]
        
        # Determine specializations to check
        specs_to_check = specializations or list(self.specialization_skills.keys())
        
        # Calculate match for each specialization
        matches = []
        for spec in specs_to_check:
            match_info = calculate_skill_match_percentage(
                user_skills,
                spec,
                self.specialization_skills,
                self.skill_weights
            )
            
            # Add specialization name to match info
            match_info['specialization'] = spec
            matches.append(match_info)
            
        # Sort matches by match percentage (descending)
        matches.sort(key=lambda x: x['match_percentage'], reverse=True)
        
        # Prepare detailed analysis for top matches
        top_matches = matches[:top_n]
        detailed_analysis = []
        
        for match in top_matches:
            spec = match['specialization']
            
            # Get interpretation of match score
            match_interpretation = self._interpret_match_score(match['match_percentage'])
            
            # Get top missing skills sorted by importance
            missing_skills = []
            if spec in self.skill_weights and 'missing_skills' in match:
                for skill in match['missing_skills']:
                    skill_weight = self.skill_weights[spec].get(skill, 0.5)
                    missing_skills.append({
                        'skill': skill,
                        'weight': skill_weight,
                        'importance': self._get_importance_level(skill_weight)
                    })
                    
                # Sort by weight (descending)
                missing_skills.sort(key=lambda x: x['weight'], reverse=True)
            
            # Create detailed analysis
            analysis = {
                'specialization': spec,
                'match_percentage': match['match_percentage'],
                'skill_coverage': match['skill_coverage'],
                'proficiency_score': match['proficiency_score'],
                'interpretation': match_interpretation,
                'matched_skills': match['matched_skills'],
                'partially_matched_skills': match['partially_matched_skills'],
                'missing_skills': missing_skills[:10]  # Limit to top 10
            }
            
            detailed_analysis.append(analysis)
            
        return {
            'user_id': user_id,
            'user_skills': user_skills,
            'top_matches': top_matches,
            'detailed_analysis': detailed_analysis
        }
    
    def _interpret_match_score(self, score):
        """Interpret a match score based on metadata."""
        for range_str, description in self.metadata.get('matchScoreInterpretation', {}).items():
            if '-' in range_str:
                lower, upper = map(int, range_str.split('-'))
                if lower <= score <= upper:
                    return description
        
        return "Unknown match quality"
    
    def _get_importance_level(self, weight):
        """Get importance level description for a weight."""
        # Convert weight to string for dictionary lookup
        weight_str = str(float(weight))
        
        # Find closest weight in metadata
        weights = list(self.metadata.get('weightDescriptions', {}).keys())
        if not weights:
            return "Unknown importance"
            
        closest_weight = min(weights, key=lambda x: abs(float(x) - weight))
        return self.metadata.get('weightDescriptions', {}).get(closest_weight, "Unknown importance")
    
    def generate_skill_gap_report(self, user_id, specialization):
        """
        Generate a detailed skill gap report for a user and specialization.
        
        Args:
            user_id: User ID from sample data
            specialization: Target specialization
            
        Returns:
            dict: Detailed skill gap report
        """
        if user_id not in self.sample_data:
            return {"error": f"User {user_id} not found in sample data"}
        
        if specialization not in self.specialization_skills:
            return {"error": f"Specialization {specialization} not found"}
            
        user_skills = self.sample_data[user_id]
        
        # Get match information
        match_info = calculate_skill_match_percentage(
            user_skills,
            specialization,
            self.specialization_skills,
            self.skill_weights
        )
        
        # Get all required skills for this specialization
        all_required_skills = self.specialization_skills.get(specialization, [])
        spec_weights = self.skill_weights.get(specialization, {})
        
        # Categorize skills
        matched = match_info['matched_skills']
        partially_matched = match_info['partially_matched_skills']
        missing = match_info['missing_skills']
        
        # Create skill details
        skill_details = []
        
        # Process matched skills
        for skill in matched:
            proficiency = user_skills.get(skill, 70)
            weight = spec_weights.get(skill, 0.5)
            
            skill_details.append({
                'skill': skill,
                'status': 'matched',
                'proficiency': proficiency,
                'weight': weight,
                'importance': self._get_importance_level(weight),
                'proficiency_level': self._get_proficiency_level(proficiency)
            })
            
        # Process partially matched skills
        for skill in partially_matched:
            # Find the user skill that matched partially
            matching_user_skill = None
            for user_skill in user_skills:
                if self._are_skills_related(skill, user_skill):
                    matching_user_skill = user_skill
                    break
                    
            proficiency = user_skills.get(matching_user_skill, 50) if matching_user_skill else 50
            weight = spec_weights.get(skill, 0.5)
            
            skill_details.append({
                'skill': skill,
                'status': 'partially_matched',
                'matched_with': matching_user_skill,
                'proficiency': proficiency * 0.5,  # Half credit for partial matches
                'weight': weight,
                'importance': self._get_importance_level(weight),
                'proficiency_level': self._get_proficiency_level(proficiency * 0.5)
            })
            
        # Process missing skills
        for skill in missing:
            weight = spec_weights.get(skill, 0.5)
            
            skill_details.append({
                'skill': skill,
                'status': 'missing',
                'proficiency': 0,
                'weight': weight,
                'importance': self._get_importance_level(weight),
                'proficiency_level': 'None'
            })
            
        # Sort by importance (weight) and then by status
        skill_details.sort(key=lambda x: (-x['weight'], x['status']))
        
        return {
            'user_id': user_id,
            'specialization': specialization,
            'match_percentage': match_info['match_percentage'],
            'skill_coverage': match_info['skill_coverage'],
            'proficiency_score': match_info['proficiency_score'],
            'interpretation': self._interpret_match_score(match_info['match_percentage']),
            'skill_details': skill_details,
            'development_priorities': [s for s in skill_details if s['status'] == 'missing'][:5]
        }
        
    def _get_proficiency_level(self, proficiency):
        """Get textual proficiency level based on proficiency value."""
        for range_str, description in self.metadata.get('proficiencyLevels', {}).items():
            if '-' in range_str:
                lower, upper = map(int, range_str.split('-'))
                if lower <= proficiency <= upper:
                    return description.split('-')[0].strip()  # Return just the level name
        
        return "Unknown"
        
    def _are_skills_related(self, skill1, skill2):
        """Check if two skills are related."""
        # Convert to lowercase for comparison
        s1 = skill1.lower()
        s2 = skill2.lower()
        
        # Direct substring match
        if s1 in s2 or s2 in s1:
            return True
            
        # Check similarity ratio
        similarity = difflib.SequenceMatcher(None, s1, s2).ratio()
        if similarity > 0.8:
            return True
            
        return False
        
    def plot_skill_match_heatmap(self, user_ids=None, specializations=None, output_path=None):
        """
        Generate a heatmap visualization of skill matches.
        
        Args:
            user_ids: List of user IDs to include, or None for all
            specializations: List of specializations to include, or None for all
            output_path: Path to save the plot, or None to display
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Determine users and specializations to include
            users = user_ids or list(self.sample_data.keys())
            specs = specializations or list(self.specialization_skills.keys())[:10]  # Limit to first 10 by default
            
            # Calculate match percentages
            match_data = []
            for user_id in users:
                user_matches = []
                user_skills = self.sample_data.get(user_id, {})
                
                for spec in specs:
                    match_info = calculate_skill_match_percentage(
                        user_skills,
                        spec,
                        self.specialization_skills,
                        self.skill_weights
                    )
                    user_matches.append(match_info['match_percentage'])
                    
                match_data.append(user_matches)
                
            # Convert to numpy array
            match_array = np.array(match_data)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            plt.imshow(match_array, cmap='YlGnBu', aspect='auto')
            plt.colorbar(label='Match Percentage')
            
            # Add labels
            plt.xticks(np.arange(len(specs)), specs, rotation=45, ha='right')
            plt.yticks(np.arange(len(users)), users)
            
            plt.title('Skill Match Percentages by User and Specialization')
            plt.tight_layout()
            
            # Save or display
            if output_path:
                plt.savefig(output_path)
                plt.close()
                print(f"Heatmap saved to {output_path}")
            else:
                plt.show()
                
            return True
        except Exception as e:
            print(f"Error generating heatmap: {str(e)}")
            return False
            
def main():
    """Run analyzer with sample data."""
    analyzer = ProficiencyAnalyzer()
    
    print("=== Testing Proficiency Analyzer ===")
    
    # Analyze a technical user
    tech_analysis = analyzer.analyze_user_skills('user1')
    print(f"\nTop matches for technical user:")
    for match in tech_analysis['detailed_analysis']:
        print(f"  {match['specialization']}: {match['match_percentage']:.2f}% ({match['interpretation']})")
        
    # Analyze a business user
    business_analysis = analyzer.analyze_user_skills('user3')
    print(f"\nTop matches for business user:")
    for match in business_analysis['detailed_analysis']:
        print(f"  {match['specialization']}: {match['match_percentage']:.2f}% ({match['interpretation']})")
        
    # Generate skill gap report
    gap_report = analyzer.generate_skill_gap_report('user1', 'Data Scientist')
    print(f"\nSkill gap report for user1 -> Data Scientist:")
    print(f"  Match: {gap_report['match_percentage']:.2f}% ({gap_report['interpretation']})")
    print(f"  Development priorities:")
    for skill in gap_report['development_priorities']:
        print(f"    - {skill['skill']} ({skill['importance']})")
        
    # Generate heatmap visualization
    analyzer.plot_skill_match_heatmap(
        user_ids=['user1', 'user2', 'user3', 'user4'], 
        specializations=['Data Scientist', 'Software Engineer', 'Machine Learning Engineer', 'Project Manager'],
        output_path=os.path.join(analyzer.base_dir, "data", "skill_match_heatmap.png")
    )
    
if __name__ == "__main__":
    main() 