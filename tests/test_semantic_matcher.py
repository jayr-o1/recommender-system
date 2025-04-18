import unittest
import sys
import os
import json
from typing import Dict, List, Any

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.semantic_matcher import SemanticMatcher, ModelConfig

class TestSemanticMatcher(unittest.TestCase):
    """Test cases for the SemanticMatcher class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test cases."""
        # Configure model with test settings
        SemanticMatcher.configure_model(ModelConfig(
            model_name="small",
            warmup_on_init=True,
            enable_progress_bars=False
        ))
        
    def test_model_init(self):
        """Test that the model initializes correctly."""
        model = SemanticMatcher.get_model()
        self.assertIsNotNone(model, "Model should be loaded")
        
    def test_semantic_similarity(self):
        """Test semantic similarity calculation."""
        # Test similar terms
        similarity = SemanticMatcher.semantic_similarity(
            "python programming", 
            "python development"
        )
        self.assertGreater(similarity, 0.7, "Similar terms should have high similarity")
        
        # Test dissimilar terms
        similarity = SemanticMatcher.semantic_similarity(
            "python programming", 
            "gardening"
        )
        self.assertLess(similarity, 0.5, "Dissimilar terms should have low similarity")
        
    def test_match_skill(self):
        """Test skill matching."""
        reference_skills = ["python development", "javascript", "data analysis"]
        
        # Test exact match
        matched, score = SemanticMatcher.match_skill("python development", reference_skills)
        self.assertEqual(matched, "python development", "Should match exactly")
        self.assertEqual(score, 1.0, "Exact match should have score 1.0")
        
        # Test similar match
        matched, score = SemanticMatcher.match_skill("python programming", reference_skills)
        self.assertEqual(matched, "python development", "Should match similar skill")
        self.assertGreater(score, 0.7, "Similar match should have high score")
        
        # Test no match
        matched, score = SemanticMatcher.match_skill("gardening", reference_skills, threshold=0.7)
        self.assertIsNone(matched, "Should not match dissimilar skill")
        self.assertLess(score, 0.7, "No match should have low score")
        
    def test_weighted_skill_match(self):
        """Test weighted skill matching."""
        user_skills = {
            "python": 80,
            "javascript": 60,
            "sql": 70
        }
        
        target_skills = {
            "python development": 90,
            "javascript programming": 80,
            "database management": 70,
            "machine learning": 60
        }
        
        matches = SemanticMatcher.weighted_skill_match(user_skills, target_skills)
        
        # Check if we got matches
        self.assertGreaterEqual(len(matches), 3, "Should match at least 3 skills")
        
        # Check if matches are sorted by weighted score
        for i in range(len(matches) - 1):
            self.assertGreaterEqual(
                matches[i]["weighted_score"],
                matches[i+1]["weighted_score"],
                "Matches should be sorted by weighted score"
            )
            
    def test_prioritize_missing_skills(self):
        """Test prioritizing missing skills."""
        user_skills = {
            "python": 80,
            "javascript": 60,
            "sql": 70
        }
        
        target_skills = {
            "python development": 90,
            "javascript programming": 80,
            "database management": 70,
            "machine learning": 60,
            "data visualization": 50
        }
        
        missing = SemanticMatcher.prioritize_missing_skills(user_skills, target_skills)
        
        # Check if we identified missing skills
        self.assertGreaterEqual(len(missing), 1, "Should identify at least one missing skill")
        
        # Check if missing skills are sorted by priority
        for i in range(len(missing) - 1):
            self.assertGreaterEqual(
                missing[i]["priority_score"],
                missing[i+1]["priority_score"],
                "Missing skills should be sorted by priority"
            )
            
    def test_cluster_skills(self):
        """Test skill clustering."""
        skills = [
            "python", "javascript", "java", "c++",
            "data analysis", "machine learning", "deep learning",
            "project management", "team leadership", "agile methodologies",
            "graphic design", "ui design", "ux design"
        ]
        
        clusters = SemanticMatcher.cluster_skills(skills)
        
        # Check if we got clusters
        self.assertGreaterEqual(len(clusters), 3, "Should create at least 3 clusters")
        
        # Check if all skills are included
        all_clustered = []
        for cluster_skills in clusters.values():
            all_clustered.extend(cluster_skills)
            
        self.assertEqual(
            sorted(all_clustered), 
            sorted(skills),
            "All skills should be included in clusters"
        )
        
    def test_analyze_skill_trends(self):
        """Test skill trend analysis."""
        skill_snapshots = {
            "2021-01": {
                "python": 60,
                "javascript": 50,
                "sql": 40
            },
            "2022-01": {
                "python": 70,
                "javascript": 60,
                "sql": 50,
                "machine learning": 30
            },
            "2023-01": {
                "python": 80,
                "javascript": 70,
                "machine learning": 50,
                "data visualization": 40
            }
        }
        
        trends = SemanticMatcher.analyze_skill_trends(skill_snapshots)
        
        # Check if trend analysis works
        self.assertIn("timestamps", trends, "Should include timestamps")
        self.assertIn("skills", trends, "Should include skills")
        self.assertIn("emerging_skills", trends, "Should include emerging skills")
        self.assertIn("new_skills", trends, "Should include new skills")
        self.assertIn("abandoned_skills", trends, "Should include abandoned skills")
        
        # Check specific skills
        self.assertIn("machine learning", trends["new_skills"], 
                    "Machine learning should be identified as a new skill")
        self.assertIn("sql", trends["abandoned_skills"],
                    "SQL should be identified as an abandoned skill")
        
    def test_suggest_skill_development_path(self):
        """Test skill development path suggestions."""
        user_skills = {
            "python": 80,
            "javascript": 60,
            "sql": 70
        }
        
        target_skills = {
            "python": 90,
            "javascript": 80,
            "sql": 70,
            "machine learning": 90,
            "data visualization": 80,
            "deep learning": 70,
            "cloud computing": 60
        }
        
        suggestions = SemanticMatcher.suggest_skill_development_path(
            user_skills, target_skills, max_suggestions=3
        )
        
        # Check if we got suggestions
        self.assertEqual(len(suggestions), 3, "Should provide 3 suggestions")
        
        # Check if suggestions have required fields
        for suggestion in suggestions:
            self.assertIn("skill", suggestion)
            self.assertIn("importance", suggestion)
            self.assertIn("learnability", suggestion)
            self.assertIn("priority", suggestion)
        
    def test_configuration(self):
        """Test configuration methods."""
        # Test matching configuration
        original_config = SemanticMatcher.get_matching_config()
        
        # Update configuration
        new_config = {
            "similarity_threshold": 0.7,
            "partial_match_threshold": 0.5
        }
        SemanticMatcher.configure_matching(new_config)
        
        # Check if configuration was updated
        updated_config = SemanticMatcher.get_matching_config()
        self.assertEqual(
            updated_config["similarity_threshold"], 
            0.7, 
            "Similarity threshold should be updated"
        )
        self.assertEqual(
            updated_config["partial_match_threshold"], 
            0.5, 
            "Partial match threshold should be updated"
        )
        
        # Reset configuration
        SemanticMatcher.reset_matching_config()
        reset_config = SemanticMatcher.get_matching_config()
        self.assertEqual(
            reset_config,
            SemanticMatcher.DEFAULT_MATCHING_CONFIG,
            "Configuration should be reset to defaults"
        )
        
    def test_model_config_serialization(self):
        """Test model configuration serialization."""
        config = ModelConfig(
            model_name="large",
            warmup_on_init=True,
            enable_progress_bars=False
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        self.assertEqual(config_dict["model_version"], "large")
        self.assertTrue(config_dict["warmup_on_init"])
        self.assertFalse(config_dict["enable_progress_bars"])
        
        # Test from_dict
        new_config = ModelConfig.from_dict(config_dict)
        self.assertEqual(new_config.model_version, "large")
        self.assertTrue(new_config.warmup_on_init)
        self.assertFalse(new_config.enable_progress_bars)
        
if __name__ == "__main__":
    unittest.main() 