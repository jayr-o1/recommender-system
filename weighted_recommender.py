#!/usr/bin/env python3
"""
Weighted career recommender that accounts for skill proficiency levels.
This module provides functions to recommend career paths based on weighted skill
matching that prioritizes skills by importance and accounts for proficiency levels.
"""

import os
import sys
import json
import difflib
import numpy as np
from collections import defaultdict

# Add parent directory to path to handle imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility functions
from utils.model_trainer import load_specialization_skills, load_skill_weights

class WeightedSkillRecommender:
    """Career recommender using weighted skill matching with proficiency levels."""
    
    def __init__(self, similarity_threshold=0.8, partial_match_factor=0.5, skill_weights_path=None, specialization_skills_path=None):
        """
        Initialize the weighted skill recommender.
        
        Args:
            similarity_threshold: Threshold for considering skills as similar (0.0-1.0)
            partial_match_factor: Credit factor for partial skill matches (0.0-1.0)
            skill_weights_path: Path to skill weights data
            specialization_skills_path: Path to specialization skills data
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.similarity_threshold = similarity_threshold
        self.partial_match_factor = partial_match_factor
        
        # Load specialization skills and weights
        self.specialization_skills = load_specialization_skills()
        self.skill_weights = load_skill_weights()
        
        # Load metadata
        metadata_path = os.path.join(self.base_dir, "data", "skill_weights_metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f).get('metadata', {})
        except Exception as e:
            print(f"Warning: Could not load metadata: {str(e)}")
            self.metadata = {}
            
    def recommend(self, user_skills, current_field=None, current_specialization=None, top_n=5):
        """
        Recommend career paths based on user skills with proficiency levels.
        
        Args:
            user_skills: Dict of user skills with proficiency levels {skill_name: proficiency}
            current_field: Current field of the user (optional)
            current_specialization: Current specialization of the user (optional)
            top_n: Number of top recommendations to return
            
        Returns:
            dict: Recommendations with detailed analysis
        """
        # Validate input
        if not user_skills:
            return {
                "success": False,
                "message": "No skills provided",
                "recommendations": None
            }
            
        # Convert string skills to dict if needed
        if isinstance(user_skills, list):
            # Convert list of skill names to dict with default proficiency
            user_skills = {skill: 70 for skill in user_skills}
            
        # Calculate match for each specialization
        all_matches = []
        
        for specialization in self.specialization_skills:
            match_info = self._calculate_weighted_match(
                user_skills, 
                specialization
            )
            
            # Add specialization name and related metadata
            match_info['specialization'] = specialization
            
            # Determine if this is in the same field as current
            if current_field:
                match_info['same_field'] = self._is_same_field(specialization, current_field)
            else:
                match_info['same_field'] = False
                
            # Determine if this is the current specialization
            match_info['is_current'] = specialization == current_specialization
            
            all_matches.append(match_info)
            
        # Sort matches by match percentage (descending)
        all_matches.sort(key=lambda x: x['match_percentage'], reverse=True)
        
        # Get top matches
        top_matches = all_matches[:top_n]
        
        # Prepare recommendations response
        top_fields = self._aggregate_fields(all_matches[:top_n*2])
        top_specializations = self._format_top_specializations(top_matches)
        
        # Generate explanation
        explanation = self._generate_explanation(user_skills, top_matches, current_field, current_specialization)
        
        recommendations = {
            "top_fields": top_fields,
            "top_specializations": top_specializations,
            "explanation": explanation
        }
        
        return {
            "success": True,
            "recommendations": recommendations
        }
            
    def _calculate_weighted_match(self, user_skills, specialization):
        """
        Calculate weighted match between user skills and a specialization.
        
        Args:
            user_skills: Dict of user skills with proficiency levels
            specialization: Target specialization
            
        Returns:
            dict: Match information
        """
        # Get required skills and their weights for this specialization
        spec_skills = self.specialization_skills.get(specialization, [])
        spec_weights = self.skill_weights.get(specialization, {})
        
        if not spec_skills:
            return {
                'match_percentage': 0,
                'skill_coverage': 0,
                'proficiency_score': 0,
                'matched_skills': [],
                'partially_matched_skills': [],
                'missing_skills': []
            }
            
        # Track matched, partially matched, and missing skills
        matched_skills = []
        partially_matched_skills = []
        missing_skills = []
        
        # Track user skills used in matches to avoid double counting
        used_user_skills = set()
        
        # Calculate match score
        total_weight = 0
        weighted_match_score = 0
        
        for skill in spec_skills:
            # Get weight for this skill (default 0.5 if not specified)
            weight = spec_weights.get(skill, 0.5)
            total_weight += weight
            
            # Check for direct match
            if skill in user_skills:
                proficiency = user_skills[skill]
                matched_skills.append(skill)
                used_user_skills.add(skill)
                
                # Weighted proficiency score (normalized to 0-1)
                norm_proficiency = proficiency / 100.0
                weighted_match_score += weight * norm_proficiency
                
            else:
                # Check for partial match (similar skill names)
                best_match = self._find_similar_skill(skill, user_skills, used_user_skills)
                
                if best_match:
                    similar_skill, similarity = best_match
                    proficiency = user_skills[similar_skill]
                    partially_matched_skills.append(skill)
                    used_user_skills.add(similar_skill)
                    
                    # Partial match gets reduced credit based on similarity and partial match factor
                    norm_proficiency = proficiency / 100.0
                    weighted_match_score += weight * norm_proficiency * similarity * self.partial_match_factor
                    
                else:
                    # Skill is missing
                    missing_skills.append(skill)
                    
        # Calculate overall match percentage
        match_percentage = (weighted_match_score / total_weight) * 100 if total_weight > 0 else 0
        
        # Calculate skill coverage (percentage of required skills matched)
        skill_coverage = ((len(matched_skills) + len(partially_matched_skills)) / len(spec_skills)) * 100 if spec_skills else 0
        
        # Calculate proficiency score (average proficiency of matched skills)
        proficiency_score = 0
        if matched_skills:
            proficiency_sum = sum(user_skills[skill] for skill in matched_skills)
            proficiency_score = proficiency_sum / len(matched_skills)
            
        # Apply specialization-specific adjustments based on skill patterns
        # Detect data science skills
        data_science_skills = ["machine learning", "deep learning", "tensorflow", "pytorch", "data science", 
                             "nlp", "natural language processing", "neural networks", "data analysis", 
                             "statistics", "pandas", "numpy", "scikit-learn"]
        
        # Detect finance skills
        finance_skills = ["financial modeling", "financial analysis", "algorithmic trading", "derivatives", 
                         "risk management", "quantitative analysis", "financial markets", "investment", 
                         "portfolio", "hedge fund", "trading", "pricing models"]
        
        # Count the presence of domain-specific skills
        ds_skill_count = sum(1 for skill in user_skills if 
                            any(ds_skill in skill.lower() for ds_skill in data_science_skills))
        finance_skill_count = sum(1 for skill in user_skills if 
                                any(fin_skill in skill.lower() for fin_skill in finance_skills))
        
        # Apply adjustments for "Quantitative Analyst" specialization
        if specialization == "Quantitative Analyst":
            # If user has many data science skills but few finance skills, reduce match
            if ds_skill_count >= 4 and finance_skill_count <= 1:
                match_percentage *= 0.7  # Reduce match for data science-heavy profiles
            # If user has finance skills, boost match
            elif finance_skill_count >= 3:
                match_percentage = min(100, match_percentage * 1.2)  # Boost for finance-heavy profiles
        
        # Apply adjustments for Data Science specializations
        if specialization in ["Data Scientist", "Machine Learning Engineer", "AI Research Scientist"]:
            # If user has many data science skills, boost match
            if ds_skill_count >= 4:
                match_percentage = min(100, match_percentage * 1.3)  # Significant boost for data science profiles
            
        return {
            'match_percentage': round(match_percentage, 2),
            'skill_coverage': round(skill_coverage, 2),
            'proficiency_score': round(proficiency_score, 2),
            'matched_skills': matched_skills,
            'partially_matched_skills': partially_matched_skills,
            'missing_skills': missing_skills
        }
            
    def _find_similar_skill(self, skill, user_skills, used_skills):
        """
        Find the most similar skill in user skills that hasn't been used yet.
        
        Args:
            skill: Target skill to match
            user_skills: Dict of user skills
            used_skills: Set of already used skills
            
        Returns:
            tuple: (similar_skill, similarity) or None if no match
        """
        best_match = None
        best_similarity = 0
        
        for user_skill in user_skills:
            # Skip already used skills
            if user_skill in used_skills:
                continue
                
            # Calculate similarity between skills
            similarity = self._calculate_skill_similarity(skill, user_skill)
            
            if similarity >= self.similarity_threshold and similarity > best_similarity:
                best_match = user_skill
                best_similarity = similarity
                
        return (best_match, best_similarity) if best_match else None
        
    def _calculate_skill_similarity(self, skill1, skill2):
        """
        Calculate similarity between two skills using multiple methods.
        
        Args:
            skill1: First skill name
            skill2: Second skill name
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Normalize skills for comparison (lowercase, remove special chars)
        s1 = skill1.lower().strip()
        s2 = skill2.lower().strip()
        
        # Direct match
        if s1 == s2:
            return 1.0
            
        # Check for one being a substring of the other
        if s1 in s2 or s2 in s1:
            # Calculate relative length of overlap
            overlap_ratio = min(len(s1), len(s2)) / max(len(s1), len(s2))
            # Higher similarity for more complete overlap
            return 0.8 + (0.2 * overlap_ratio)
            
        # Handle common variations
        # Remove common prefixes/suffixes for comparison
        common_prefixes = ['advanced ', 'basic ', 'intermediate ', 'proficient in ', 'knowledge of ']
        common_suffixes = [' programming', ' development', ' language', ' framework', ' design', ' analysis', ' management']
        
        s1_normalized = s1
        s2_normalized = s2
        
        for prefix in common_prefixes:
            s1_normalized = s1_normalized[len(prefix):] if s1_normalized.startswith(prefix) else s1_normalized
            s2_normalized = s2_normalized[len(prefix):] if s2_normalized.startswith(prefix) else s2_normalized
            
        for suffix in common_suffixes:
            s1_normalized = s1_normalized[:-len(suffix)] if s1_normalized.endswith(suffix) else s1_normalized
            s2_normalized = s2_normalized[:-len(suffix)] if s2_normalized.endswith(suffix) else s2_normalized
        
        # Check if normalized versions match
        if s1_normalized == s2_normalized:
            return 0.9
        
        # Handle common acronyms and full forms
        common_acronyms = {
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'ui': 'user interface',
            'ux': 'user experience',
            'js': 'javascript',
            'py': 'python',
            'ts': 'typescript',
            'db': 'database',
            'api': 'application programming interface',
            'oop': 'object oriented programming',
            'ci': 'continuous integration',
            'cd': 'continuous deployment',
            'devops': 'development operations'
        }
        
        # Expand acronyms if possible
        s1_expanded = common_acronyms.get(s1_normalized, s1_normalized)
        s2_expanded = common_acronyms.get(s2_normalized, s2_normalized)
        
        # Check if expanded versions match
        if s1_expanded == s2_expanded:
            return 0.85
        
        # Handle version numbers (e.g., "Python 3" ≈ "Python")
        import re
        version_pattern = r'(.*?)\s*\d+(\.\d+)*$'
        s1_base = re.match(version_pattern, s1_normalized)
        s2_base = re.match(version_pattern, s2_normalized)
        
        if s1_base and s2_base and s1_base.group(1).strip() == s2_base.group(1).strip():
            return 0.8
        
        # Use difflib sequence matcher for more general similarity
        from difflib import SequenceMatcher
        sequence_similarity = SequenceMatcher(None, s1_normalized, s2_normalized).ratio()
        
        # Use word-level similarity for longer phrases
        words1 = set(s1_normalized.split())
        words2 = set(s2_normalized.split())
        
        if words1 and words2:
            jaccard_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
            # Combine both similarities with more weight on sequence for short terms
            # and more weight on word-level for longer phrases
            if len(words1) <= 2 and len(words2) <= 2:
                combined_similarity = 0.8 * sequence_similarity + 0.2 * jaccard_similarity
            else:
                combined_similarity = 0.4 * sequence_similarity + 0.6 * jaccard_similarity
            
            return combined_similarity
            
        return sequence_similarity
        
    def _is_same_field(self, specialization, field):
        """
        Determine if a specialization belongs to a specific field.
        
        Args:
            specialization: Specialization to check
            field: Field to compare against
            
        Returns:
            bool: True if specialization is in the field
        """
        # This is a simple heuristic based on keywords
        field_keywords = {
            "Computer Science": ["engineer", "developer", "programmer", "scientist", "data", "software", "web"],
            "Business": ["manager", "management", "business", "analyst"],
            "Finance": ["finance", "financial", "accounting", "investment"],
            "Marketing": ["marketing", "brand", "content", "social media"],
            "Design": ["design", "ui", "ux", "user", "graphic"],
            "Human Resources": ["hr", "human resources", "recruiting", "talent"]
        }
        
        # Check if specialization contains any of the field's keywords
        if field in field_keywords:
            keywords = field_keywords[field]
            specialization_lower = specialization.lower()
            
            for keyword in keywords:
                if keyword in specialization_lower:
                    return True
                    
        return False
        
    def _aggregate_fields(self, top_matches):
        """
        Aggregate top matches by field.
        
        Args:
            top_matches: List of top matching specializations
            
        Returns:
            list: Aggregated field recommendations
        """
        fields = defaultdict(lambda: {
            'field': '',
            'match_percentage': 0,
            'matching_skills': set(),
            'missing_skills': set(),
            'count': 0
        })
        
        for match in top_matches:
            spec = match['specialization']
            
            # Determine field from specialization
            # Special case handling for specific specializations
            if spec == "Quantitative Analyst":
                field = "Finance"
            elif "Data Scientist" in spec or "Machine Learning" in spec or "AI" in spec or "Data Science" in spec or "NLP" in spec or "Natural Language Processing" in spec:
                field = "Data Science"
            elif "Engineer" in spec or "Developer" in spec or "Programmer" in spec:
                field = "Computer Science"
            elif "Manager" in spec or "Management" in spec:
                field = "Business"
            elif "Analyst" in spec and "Data" not in spec and "Quantitative" not in spec:
                field = "Business"
            elif "Financial" in spec or "Finance" in spec or "Quantitative" in spec:
                field = "Finance"
            elif "Marketing" in spec:
                field = "Marketing"
            elif "Design" in spec or "UI" in spec or "UX" in spec:
                field = "Design"
            elif "HR" in spec or "Human Resources" in spec:
                field = "Human Resources"
            else:
                field = "Other"
                
            # Update field data
            fields[field]['field'] = field
            fields[field]['match_percentage'] += match['match_percentage']
            fields[field]['matching_skills'].update(match['matched_skills'])
            fields[field]['matching_skills'].update(match['partially_matched_skills'])
            fields[field]['missing_skills'].update(match['missing_skills'])
            fields[field]['count'] += 1
            
        # Calculate average match percentage and format output
        field_recommendations = []
        
        for field_name, data in fields.items():
            if data['count'] > 0:
                avg_match = data['match_percentage'] / data['count']
                
                field_rec = {
                    'field': field_name,
                    'match_percentage': round(avg_match, 2),
                    'matching_skills': list(data['matching_skills'])[:10],  # Limit to top 10
                    'missing_skills': list(data['missing_skills'])[:5]      # Limit to top 5
                }
                
                field_recommendations.append(field_rec)
                
        # Sort by match percentage
        field_recommendations.sort(key=lambda x: x['match_percentage'], reverse=True)
        
        return field_recommendations
        
    def _format_top_specializations(self, matches):
        """
        Format specialization matches for recommendation output.
        
        Args:
            matches: List of specialization matches
            
        Returns:
            list: Formatted specialization recommendations
        """
        formatted_matches = []
        
        for match in matches:
            formatted = {
                'specialization': match['specialization'],
                'match_percentage': match['match_percentage'],
                'skill_coverage': match['skill_coverage'],
                'proficiency_score': match['proficiency_score'],
                'matching_skills': match['matched_skills'],
                'missing_skills': match['missing_skills']
            }
            
            # Add same field and current specialization flags if available
            if 'same_field' in match:
                formatted['same_field'] = match['same_field']
                
            if 'is_current' in match:
                formatted['is_current'] = match['is_current']
                
            formatted_matches.append(formatted)
            
        return formatted_matches
            
    def _generate_explanation(self, user_skills, top_matches, current_field, current_specialization):
        """
        Generate explanation for recommendations.
        
        Args:
            user_skills: Dict of user skills with proficiency levels
            top_matches: List of top matching specializations
            current_field: Current field of the user (optional)
            current_specialization: Current specialization of the user (optional)
            
        Returns:
            dict: Explanation object
        """
        # Identify key strengths (skills with high proficiency)
        key_strengths = []
        for skill, proficiency in user_skills.items():
            if proficiency >= 80:
                key_strengths.append({
                    'skill': skill,
                    'proficiency': proficiency,
                    'relevance': 'high' if any(skill in match['matched_skills'] for match in top_matches) else 'medium'
                })
                
        # Sort strengths by proficiency and relevance
        key_strengths.sort(key=lambda x: (x['relevance'] == 'high', x['proficiency']), reverse=True)
        
        # Identify development areas (missing skills that appear frequently in top matches)
        missing_skill_count = defaultdict(int)
        for match in top_matches:
            for skill in match['missing_skills']:
                missing_skill_count[skill] += 1
                
        # Get most common missing skills
        development_areas = []
        for skill, count in sorted(missing_skill_count.items(), key=lambda x: x[1], reverse=True):
            # Find a match that has this missing skill to get its weight
            for match in top_matches:
                if skill in match['missing_skills']:
                    spec = match['specialization']
                    weight = self.skill_weights.get(spec, {}).get(skill, 0.5)
                    
                    development_areas.append({
                        'skill': skill,
                        'importance': weight,
                        'frequency': count
                    })
                    break
                    
        # Generate summary text
        summary = "Based on your skill profile with proficiency levels, we've identified potential career paths."
        
        if current_specialization:
            summary += f" Your current role as a {current_specialization}"
            if current_field:
                summary += f" in {current_field}"
            summary += " has been considered in these recommendations."
            
        # Generate detailed text with insights about the recommendations
        details = ""
        if top_matches:
            best_match = top_matches[0]
            
            # Interpret match quality
            if best_match['match_percentage'] >= 80:
                details += f"You have an excellent match ({best_match['match_percentage']}%) with {best_match['specialization']}. "
                details += "Your skill proficiency levels indicate you're well-prepared for this career path."
            elif best_match['match_percentage'] >= 60:
                details += f"You have a good match ({best_match['match_percentage']}%) with {best_match['specialization']}. "
                details += "With some targeted skill development, you could excel in this area."
            elif best_match['match_percentage'] >= 40:
                details += f"You have a moderate match ({best_match['match_percentage']}%) with {best_match['specialization']}. "
                details += "This could be a viable path with focused upskilling in key areas."
            else:
                details += "Your current skills show limited direct alignment with the analyzed specializations. "
                details += "Consider developing skills in your areas of interest to improve your match."
                
            # Add transition insights if current specialization provided
            if current_specialization and not best_match['is_current']:
                if best_match['same_field']:
                    details += f"\n\nA transition from {current_specialization} to {best_match['specialization']} "
                    details += "would be relatively straightforward as they are in the same field."
                else:
                    details += f"\n\nA transition from {current_specialization} to {best_match['specialization']} "
                    details += "would be a more significant career change across different fields."
        
        return {
            'summary': summary,
            'details': details,
            'skill_analysis': {
                'key_strengths': key_strengths[:5],  # Top 5 strengths
                'development_areas': development_areas[:5]  # Top 5 areas to develop
            }
        }
            
def main():
    """Test the weighted recommender with some example data."""
    print("=== Testing Weighted Skill Recommender ===")
    
    # Create recommender
    recommender = WeightedSkillRecommender()
    
    # Test with a software developer's skills
    software_dev_skills = {
        "Python": 85,
        "JavaScript": 70,
        "React.js": 75,
        "Node.js": 65,
        "SQL": 80,
        "MongoDB": 60,
        "Git": 90,
        "Docker": 50,
        "Web Development": 75,
        "API Design": 70
    }
    
    print("\nRecommendations for Software Developer:")
    results = recommender.recommend(
        software_dev_skills,
        current_field="Computer Science",
        current_specialization="Software Developer"
    )
    
    if results['success']:
        recs = results['recommendations']
        print(f"\nTop Fields:")
        for field in recs['top_fields'][:3]:
            print(f"  - {field['field']}: {field['match_percentage']}%")
            
        print(f"\nTop Specializations:")
        for spec in recs['top_specializations'][:3]:
            print(f"  - {spec['specialization']}: {spec['match_percentage']}%")
            print(f"    Skill coverage: {spec['skill_coverage']}%, Proficiency: {spec['proficiency_score']}%")
            print(f"    Missing skills: {', '.join(spec['missing_skills'][:3])}")
            
        print(f"\nExplanation Summary:")
        print(f"  {recs['explanation']['summary']}")
    else:
        print(f"Error: {results['message']}")
        
if __name__ == "__main__":
    main() 