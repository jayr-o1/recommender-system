import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from functools import lru_cache
import logging
from fuzzywuzzy import fuzz
# Import sentence_transformers only when needed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticMatcher:
    """
    Advanced skill matching using semantic similarity with transformers.
    Provides more accurate matching for specialized terminology by considering
    semantic meaning rather than just string similarity.
    """
    
    # Singleton model instance for efficiency
    _model = None
    
    # Cache for embeddings to avoid redundant computation
    _embedding_cache = {}
    
    # Cache for skill matching results
    _match_cache = {}
    
    # Specialized domain terms by category
    DOMAIN_TERMS = {
        "chemistry": {
            "organic", "inorganic", "analytical", "biochemistry", "spectroscopy", 
            "chromatography", "synthesis", "catalysis", "polymer", "pharmaceutical"
        },
        "programming": {
            "python", "javascript", "typescript", "java", "c++", "ruby", "php", 
            "react", "angular", "vue", "node", "django", "flask", "spring"
        },
        "data_science": {
            "machine learning", "deep learning", "neural networks", "statistics", 
            "data analysis", "pandas", "tensorflow", "pytorch", "nlp", "computer vision"
        },
        "design": {
            "ui", "ux", "photoshop", "illustrator", "indesign", "figma", "sketch",
            "typography", "wireframing", "prototyping", "responsive"
        },
        "business": {
            "management", "leadership", "strategy", "marketing", "analytics", 
            "project management", "agile", "scrum", "kanban", "waterfall"
        }
    }
    
    @classmethod
    def get_model(cls):
        """Get or initialize the sentence transformer model"""
        if cls._model is None:
            try:
                # Import here to avoid circular imports
                from sentence_transformers import SentenceTransformer
                # Use a smaller, faster model for production
                cls._model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer model")
            except Exception as e:
                logger.error(f"Error loading sentence transformer model: {e}")
                # Fallback to None - will use fuzzy matching instead
                cls._model = None
        return cls._model
    
    @classmethod
    def clear_cache(cls):
        """Clear all caches to free memory"""
        cls._embedding_cache.clear()
        cls._match_cache.clear()
        # Also clear any lru_cache decorated methods
        cls.get_embedding.cache_clear()
        cls.semantic_similarity.cache_clear()
    
    @classmethod
    @lru_cache(maxsize=1024)
    def get_embedding(cls, text: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a text string
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if model not available
        """
        if text in cls._embedding_cache:
            return cls._embedding_cache[text]
            
        model = cls.get_model()
        if model is None:
            return None
            
        try:
            embedding = model.encode(text, convert_to_tensor=False)
            cls._embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for '{text}': {e}")
            return None
    
    @classmethod
    @lru_cache(maxsize=4096)
    def semantic_similarity(cls, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Check if we have the model available
        if cls.get_model() is None:
            # Fallback to fuzzy string matching
            return fuzz.token_sort_ratio(text1.lower(), text2.lower()) / 100
            
        # Get embeddings
        emb1 = cls.get_embedding(text1)
        emb2 = cls.get_embedding(text2)
        
        # If either embedding failed, fall back to fuzzy matching
        if emb1 is None or emb2 is None:
            return fuzz.token_sort_ratio(text1.lower(), text2.lower()) / 100
            
        try:
            # Import here to avoid circular imports
            from sentence_transformers import util
            # Calculate cosine similarity
            return float(util.cos_sim(emb1, emb2)[0][0])
        except Exception:
            # Fallback to fuzzy string matching if util import fails
            return fuzz.token_sort_ratio(text1.lower(), text2.lower()) / 100
    
    @classmethod
    def get_domain_bonus(cls, skill: str) -> Dict[str, float]:
        """
        Calculate bonus scores for domain-specific terms in the skill
        
        Args:
            skill: Skill name to analyze
            
        Returns:
            Dictionary of domain categories and their bonus scores
        """
        skill_lower = skill.lower()
        bonuses = {}
        
        for domain, terms in cls.DOMAIN_TERMS.items():
            # Check for exact matches in domain terms
            domain_score = 0
            for term in terms:
                if term in skill_lower:
                    # Give higher bonus for full word matches
                    term_parts = term.split()
                    if len(term_parts) > 1:
                        # Multi-word term, check if all words appear
                        if all(part in skill_lower.split() for part in term_parts):
                            domain_score += 0.2
                        else:
                            domain_score += 0.1
                    else:
                        # Single word term, check if it's a full word
                        skill_parts = skill_lower.split()
                        if term in skill_parts:
                            domain_score += 0.2
                        else:
                            domain_score += 0.1
            
            if domain_score > 0:
                bonuses[domain] = min(domain_score, 0.5)  # Cap at 0.5
                
        return bonuses
    
    @classmethod
    def match_skill(cls, skill: str, reference_skills: List[str], 
                   threshold: float = 0.6, use_semantic: bool = True) -> Tuple[Optional[str], float]:
        """
        Match a skill name to the closest reference skill using semantic similarity
        
        Args:
            skill: Skill name to match
            reference_skills: List of reference skill names
            threshold: Minimum similarity score (0-1) to consider a match
            use_semantic: Whether to use semantic matching or fall back to fuzzy
            
        Returns:
            Tuple of (matched reference skill name or None, similarity score)
        """
        if not reference_skills:
            return None, 0.0
            
        # Create tuple for cache key
        reference_tuple = tuple(sorted(reference_skills))
        
        # Check for cached results
        cache_key = (skill.lower(), reference_tuple, threshold, use_semantic)
        if cache_key in cls._match_cache:
            return cls._match_cache[cache_key]
            
        # Try direct match first (case-insensitive)
        skill_lower = skill.lower()
        for ref_skill in reference_skills:
            if skill_lower == ref_skill.lower():
                result = (ref_skill, 1.0)
                cls._match_cache[cache_key] = result
                return result
        
        # Apply semantic matching
        best_match = None
        best_score = 0.0
        
        # Get domain bonuses for the input skill
        domain_bonuses = cls.get_domain_bonus(skill)
        
        for ref_skill in reference_skills:
            # Calculate base similarity score
            if use_semantic:
                similarity = cls.semantic_similarity(skill, ref_skill)
            else:
                # Fallback to fuzzy matching
                similarity = fuzz.token_sort_ratio(skill_lower, ref_skill.lower()) / 100
            
            # Apply domain-specific bonuses
            ref_skill_bonuses = cls.get_domain_bonus(ref_skill)
            bonus = 0.0
            
            # If both skills have the same domain, apply bonus
            for domain, score in domain_bonuses.items():
                if domain in ref_skill_bonuses:
                    bonus += min(score, ref_skill_bonuses[domain])
            
            # Apply the bonus (capped to avoid exceeding 1.0)
            final_score = min(1.0, similarity + bonus)
            
            if final_score > best_score:
                best_match = ref_skill
                best_score = final_score
        
        # Only return match if it exceeds threshold
        if best_score >= threshold:
            result = (best_match, best_score)
        else:
            result = (None, best_score)
            
        # Cache the result
        cls._match_cache[cache_key] = result
        return result
    
    @classmethod
    def weighted_skill_match(cls, user_skills: Dict[str, int], 
                           target_skills: Dict[str, int],
                           threshold: float = 0.65) -> List[Dict[str, Any]]:
        """
        Find matches between user skills and target skills with weighted scoring
        
        Args:
            user_skills: Dictionary of user skills and proficiency (1-100)
            target_skills: Dictionary of target skills and their importance (1-100)
            threshold: Minimum similarity score (0-1) to consider a match
            
        Returns:
            List of matched skills with details and scores
        """
        matches = []
        
        # Get all user skills and target skills for reference
        user_skill_names = list(user_skills.keys())
        target_skill_names = list(target_skills.keys())
        
        for target_skill, importance in target_skills.items():
            # Find the best matching user skill
            best_match = None
            best_score = 0.0
            best_proficiency = 0
            
            for user_skill, proficiency in user_skills.items():
                matched_skill, score = cls.match_skill(user_skill, [target_skill], threshold)
                if matched_skill and score > best_score:
                    best_match = user_skill
                    best_score = score
                    best_proficiency = proficiency
            
            # If we found a match
            if best_match:
                # Calculate a weighted match score that considers:
                # 1. Semantic similarity (best_score)
                # 2. User proficiency (normalized to 0-1)
                # 3. Skill importance (normalized to 0-1)
                proficiency_factor = best_proficiency / 100
                importance_factor = importance / 100
                weighted_score = best_score * proficiency_factor * importance_factor
                
                matches.append({
                    "target_skill": target_skill,
                    "user_skill": best_match,
                    "similarity": best_score,
                    "proficiency": best_proficiency,
                    "importance": importance,
                    "weighted_score": weighted_score
                })
                
        # Sort by weighted score
        matches.sort(key=lambda x: x["weighted_score"], reverse=True)
        return matches 