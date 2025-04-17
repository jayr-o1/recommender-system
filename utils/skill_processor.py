import re
from typing import Dict, List, Tuple, Any, Optional, Set
import difflib
from functools import lru_cache

class SkillProcessor:
    """
    Utility class for processing skills input and matching skills 
    with fuzzy matching to handle variations in skill names
    """
    
    # Cache for skill matching to avoid redundant computation
    _match_cache = {}
    
    @staticmethod
    def parse_skills_input(skills_text: str) -> Dict[str, int]:
        """
        Parse skills input from text format (e.g. "Python 90, SQL 80, Data Analysis 85")
        
        Args:
            skills_text: Comma-separated string of skills and proficiency
            
        Returns:
            Dictionary mapping skill names to proficiency scores
            
        Raises:
            ValueError: If input text is not properly formatted
        """
        if not skills_text or not isinstance(skills_text, str):
            raise ValueError("Skills text must be a non-empty string")
            
        skills_dict = {}
        
        # Split by commas
        parts = [p.strip() for p in skills_text.split(',')]
        
        for part in parts:
            if not part:  # Skip empty parts
                continue
                
            # Try to match "Skill Name Proficiency" pattern
            match = re.match(r'(.*?)(\d+)$', part.strip())
            
            if match:
                skill_name = match.group(1).strip()
                if not skill_name:
                    continue  # Skip if skill name is empty
                    
                try:
                    proficiency = int(match.group(2))
                    # Ensure proficiency is within range
                    proficiency = max(1, min(100, proficiency))
                    skills_dict[skill_name] = proficiency
                except ValueError:
                    # If proficiency can't be parsed as int, use default value
                    skills_dict[skill_name] = 70  # Default proficiency
            else:
                # If no proficiency specified, use default value
                if part.strip():  # Only add if not empty
                    skills_dict[part.strip()] = 70  # Default proficiency
        
        return skills_dict
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def match_skill(skill: str, reference_skills_tuple: Tuple[str, ...], 
                   threshold: float = 0.8) -> Optional[str]:
        """
        Match a skill name to the closest reference skill using fuzzy matching
        
        Uses caching for performance improvement
        
        Args:
            skill: Skill name to match
            reference_skills_tuple: Tuple of reference skill names (must be tuple for caching)
            threshold: Minimum similarity score (0-1) to consider a match
            
        Returns:
            Matched reference skill name or None if no match found
        """
        # Convert tuple back to list
        reference_skills = list(reference_skills_tuple)
        
        # Check for cached results
        cache_key = (skill.lower(), reference_skills_tuple, threshold)
        if cache_key in SkillProcessor._match_cache:
            return SkillProcessor._match_cache[cache_key]
            
        # Direct match first
        if skill in reference_skills:
            SkillProcessor._match_cache[cache_key] = skill
            return skill
            
        # Case-insensitive match
        skill_lower = skill.lower()
        for ref_skill in reference_skills:
            if skill_lower == ref_skill.lower():
                SkillProcessor._match_cache[cache_key] = ref_skill
                return ref_skill
        
        # Fuzzy matching
        matches = difflib.get_close_matches(skill, reference_skills, n=1, cutoff=threshold)
        result = matches[0] if matches else None
        
        # Cache the result
        SkillProcessor._match_cache[cache_key] = result
        
        return result
    
    @staticmethod
    def standardize_skills(skills: Dict[str, int], 
                          reference_skills: List[str]) -> Tuple[Dict[str, int], List[str]]:
        """
        Standardize skill names by matching them to reference skills
        
        Args:
            skills: Dictionary of user skills and proficiency
            reference_skills: List of standard skill names
            
        Returns:
            Tuple of standardized skills dictionary and list of unmatched skills
            
        Raises:
            ValueError: If reference_skills is empty
        """
        if not reference_skills:
            raise ValueError("Reference skills list cannot be empty")
            
        standardized = {}
        unmatched = []
        
        # Convert to tuple for caching
        reference_skills_tuple = tuple(reference_skills)
        
        # Process skills in batches to optimize
        for skill, proficiency in skills.items():
            matched = SkillProcessor.match_skill(skill, reference_skills_tuple)
            if matched:
                # If the skill is already in standardized with a different proficiency,
                # keep the higher proficiency
                if matched in standardized:
                    standardized[matched] = max(standardized[matched], proficiency)
                else:
                    standardized[matched] = proficiency
            else:
                unmatched.append(skill)
        
        return standardized, unmatched
    
    @staticmethod
    def get_missing_skills(user_skills: Dict[str, int], 
                         required_skills: Dict[str, int],
                         threshold: int = 70) -> List[Dict[str, Any]]:
        """
        Identify missing skills based on required skills for a specialization
        
        Args:
            user_skills: Dictionary of user skills and proficiency
            required_skills: Dictionary of required skills and their importance
            threshold: Minimum importance threshold to include in missing skills
            
        Returns:
            List of missing skills with their importance
            
        Raises:
            ValueError: If required_skills is empty
        """
        if not required_skills:
            return []
            
        missing = []
        
        # Create a set of lowercase user skills for faster lookups
        user_skills_lower = {skill.lower(): proficiency for skill, proficiency in user_skills.items()}
        
        for skill, importance in required_skills.items():
            if importance >= threshold:
                # Check if skill exists in user skills (case insensitive)
                if skill.lower() not in user_skills_lower:
                    missing.append({
                        "skill": skill,
                        "weight": importance
                    })
        
        # Sort by importance
        missing.sort(key=lambda x: x["weight"], reverse=True)
        return missing
    
    @staticmethod
    def clear_cache() -> None:
        """Clear the internal caches to free memory"""
        SkillProcessor._match_cache.clear()
        # Also clear the lru_cache for match_skill
        SkillProcessor.match_skill.cache_clear()
    
    @staticmethod
    def get_overlapping_skills(user_skills: Dict[str, int], 
                             target_skills: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
        """
        Find overlapping skills between user skills and target skills
        
        Args:
            user_skills: Dictionary of user skills and proficiency
            target_skills: Dictionary of target skills and their importance
            
        Returns:
            Dictionary of overlapping skills with proficiency and importance
        """
        overlapping = {}
        
        # Create lowercase mappings for case-insensitive comparison
        user_skills_lower = {skill.lower(): (skill, prof) for skill, prof in user_skills.items()}
        
        for skill, importance in target_skills.items():
            skill_lower = skill.lower()
            if skill_lower in user_skills_lower:
                original_skill, proficiency = user_skills_lower[skill_lower]
                overlapping[skill] = {
                    "proficiency": proficiency,
                    "importance": importance,
                    "match_score": (proficiency / 100) * (importance / 100)
                }
                
        return overlapping 