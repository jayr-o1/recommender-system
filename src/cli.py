#!/usr/bin/env python3
import argparse
import json
import sys
import os
import textwrap
import logging
import traceback
from typing import Dict, List, Any, Optional

# Add parent directory to path so we can import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommender import CareerRecommender
from utils.skill_processor import SkillProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecommenderCLI:
    """Command-line interface for the career recommender system"""
    
    def __init__(self, data_path: str = "data", model_path: str = "model"):
        """
        Initialize the CLI with recommender and skill processor
        
        Args:
            data_path: Path to data directory
            model_path: Path to model directory
        """
        try:
            self.recommender = CareerRecommender(data_path, model_path)
            self.skill_processor = SkillProcessor()
            logger.info("Recommender CLI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize recommender: {e}")
            raise
    
    def _format_specialization(self, spec: Dict[str, Any]) -> str:
        """Format a specialization for display"""
        try:
            result = []
            result.append(f"\n{spec['specialization']} ({spec['field']})")
            result.append(f"  Confidence: {spec['confidence']}%")
            result.append(f"  Skills matched: {spec['skills_matched']}/{spec['total_skills_required']}")
            
            if spec.get('matched_skills'):
                result.append("\n  Matched Skills:")
                for skill in spec['matched_skills']:
                    result.append(f"    - {skill['skill']} (Your proficiency: {skill['proficiency']}%, Weight: {skill['weight']}%)")
            
            if spec.get('missing_skills'):
                result.append("\n  Missing Skills:")
                for skill in spec['missing_skills']:
                    result.append(f"    - {skill['skill']} (Weight: {skill['weight']}%)")
                    
            return "\n".join(result)
        except KeyError as e:
            logger.error(f"Missing key in specialization data: {e}")
            return f"Error formatting specialization: {spec.get('specialization', 'Unknown')}"
    
    def process_skills(self, skills_text: str) -> Dict[str, int]:
        """
        Process skills input from text
        
        Args:
            skills_text: Comma-separated string of skills and proficiency
            
        Returns:
            Dictionary of skills and proficiency levels
            
        Raises:
            ValueError: If skills_text is invalid or empty
        """
        if not skills_text or not skills_text.strip():
            raise ValueError("Skills input cannot be empty")
            
        try:
            skills = self.skill_processor.parse_skills_input(skills_text)
            if not skills:
                raise ValueError("No valid skills found in input")
            return skills
        except Exception as e:
            logger.error(f"Error processing skills input: {e}")
            raise ValueError(f"Failed to process skills: {e}")
    
    def validate_skills(self, skills: Dict[str, int]) -> Dict[str, int]:
        """
        Validate and standardize skills against known skills
        
        Args:
            skills: Dictionary of skills and proficiency levels
            
        Returns:
            Standardized skills dictionary
            
        Raises:
            ValueError: If no skills can be matched
        """
        try:
            # Get all known skills from skill weights
            reference_skills = list(self.recommender.skill_weights.keys())
            
            # Standardize skills
            standardized_skills, unmatched = self.skill_processor.standardize_skills(skills, reference_skills)
            
            if not standardized_skills:
                raise ValueError("None of the provided skills could be matched to known skills")
                
            if unmatched:
                logger.warning(f"Some skills could not be matched: {', '.join(unmatched)}")
                
            return standardized_skills
        except Exception as e:
            logger.error(f"Error validating skills: {e}")
            raise
    
    def get_field_recommendations(self, skills: Dict[str, int], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Get field recommendations based on skills
        
        Args:
            skills: Dictionary of skills and proficiency levels
            top_n: Number of top recommendations to return
            
        Returns:
            List of field recommendations
            
        Raises:
            ValueError: If skills is empty or invalid
        """
        if not skills:
            raise ValueError("Skills dictionary cannot be empty")
            
        try:
            standardized_skills = self.validate_skills(skills)
            return self.recommender.recommend_field(standardized_skills, top_n)
        except Exception as e:
            logger.error(f"Error getting field recommendations: {e}")
            raise
    
    def get_specialization_recommendations(self, skills: Dict[str, int], 
                                         field: Optional[str] = None, 
                                         top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Get specialization recommendations based on skills
        
        Args:
            skills: Dictionary of skills and proficiency levels
            field: Optional field to filter by
            top_n: Number of top recommendations to return
            
        Returns:
            List of specialization recommendations
            
        Raises:
            ValueError: If skills is empty or invalid
        """
        if not skills:
            raise ValueError("Skills dictionary cannot be empty")
            
        try:
            standardized_skills = self.validate_skills(skills)
            return self.recommender.recommend_specializations(standardized_skills, field, top_n)
        except Exception as e:
            logger.error(f"Error getting specialization recommendations: {e}")
            raise
    
    def run(self):
        """Run the CLI application"""
        parser = argparse.ArgumentParser(description="Career Path Recommender CLI")
        subparsers = parser.add_subparsers(dest="command", help="Command to run")
        
        # Field recommendation command
        field_parser = subparsers.add_parser("field", help="Recommend career fields")
        field_parser.add_argument("--skills", "-s", required=True, help="Comma-separated list of skills with proficiency (e.g. 'Python 90, SQL 80')")
        field_parser.add_argument("--top", "-t", type=int, default=3, help="Number of top recommendations to show")
        
        # Specialization recommendation command
        spec_parser = subparsers.add_parser("spec", help="Recommend specializations")
        spec_parser.add_argument("--skills", "-s", required=True, help="Comma-separated list of skills with proficiency (e.g. 'Python 90, SQL 80')")
        spec_parser.add_argument("--field", "-f", help="Filter by field")
        spec_parser.add_argument("--top", "-t", type=int, default=3, help="Number of top recommendations to show")
        
        # Parse arguments
        args = parser.parse_args()
        
        try:
            if not args.command:
                parser.print_help()
                return
                
            # Process skills
            skills = self.process_skills(args.skills)
            
            if args.command == "field":
                # Get field recommendations
                recommendations = self.get_field_recommendations(skills, args.top)
                
                print(f"\nTop {len(recommendations)} Field Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n{i}. {rec['field']} (Confidence: {rec['confidence']}%)")
                    print(f"   Matched Skills: {rec['matched_skills']}")
                    
            elif args.command == "spec":
                # Get specialization recommendations
                recommendations = self.get_specialization_recommendations(skills, args.field, args.top)
                
                print(f"\nTop {len(recommendations)} Specialization Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {self._format_specialization(rec)}")
                    
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            logger.error(f"Exception in CLI: {traceback.format_exc()}")
            return 1
            
        return 0


if __name__ == "__main__":
    try:
        cli = RecommenderCLI()
        sys.exit(cli.run())
    except Exception as e:
        print(f"Failed to start recommender: {e}")
        logger.error(f"Fatal error: {traceback.format_exc()}")
        sys.exit(1) 