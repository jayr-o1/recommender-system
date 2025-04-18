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
    
    # Flag to control progress bar display (should be disabled for API)
    _show_progress_bars = True
    
    # Specialized domain terms by category - ENHANCED with more comprehensive terms
    DOMAIN_TERMS = {
        "chemistry": {
            "organic", "inorganic", "analytical", "biochemistry", "spectroscopy", 
            "chromatography", "synthesis", "catalysis", "polymer", "pharmaceutical",
            "toxicology", "chemical", "laboratory", "techniques", "instrumentation",
            "compounds", "reactions", "titration", "molecular", "solvents",
            "bioassay", "formulation", "purity", "pharmacology", "crystallography"
        },
        "programming": {
            "python", "javascript", "typescript", "java", "c++", "ruby", "php", 
            "react", "angular", "vue", "node", "django", "flask", "spring",
            "kotlin", "swift", "rust", "go", "c#", "dotnet", "asp.net",
            "sql", "nosql", "database", "api", "rest", "graphql", "webdev",
            "frontend", "backend", "fullstack", "mobile", "web", "app"
        },
        "data_science": {
            "machine learning", "deep learning", "neural networks", "statistics", 
            "data analysis", "pandas", "tensorflow", "pytorch", "nlp", "computer vision",
            "big data", "data mining", "data visualization", "feature engineering",
            "clustering", "classification", "regression", "forecasting", "prediction",
            "etl", "business intelligence", "tableau", "power bi", "analytics"
        },
        "design": {
            "ui", "ux", "ui/ux", "user interface", "user experience", "web design",
            "photoshop", "illustrator", "indesign", "figma", "sketch",
            "typography", "wireframing", "prototyping", "responsive",
            "graphic design", "visual design", "interaction design", "information architecture",
            "usability", "accessibility", "creative", "brand identity", "layout"
        },
        "business": {
            "management", "leadership", "strategy", "marketing", "analytics", 
            "project management", "agile", "scrum", "kanban", "waterfall",
            "advertising", "branding", "market research", "public relations", "sales",
            "business development", "operations", "finance", "accounting", "consulting",
            "entrepreneurship", "e-commerce", "digital marketing", "content marketing", "seo"
        },
        "cybersecurity": {
            "pentesting", "penetration testing", "network security", "vulnerability", "ethical hacking",
            "security analysis", "malware", "encryption", "firewall", "intrusion detection",
            "ddos", "security audit", "cyber threat", "incident response", "forensics",
            "access control", "authentication", "cryptography", "security governance", "compliance",
            "security operations", "risk assessment", "threat hunting", "security architecture", "osint"
        },
        "arts": {
            "dancing", "painting", "drawing", "sketching", "sculpture", 
            "performance", "choreography", "visual arts", "fine arts", "crafts",
            "photography", "design", "illustration", "ceramics", "printmaking",
            "art direction", "fashion design", "digital art", "3d modeling", "animation",
            "comics", "concept art", "character design", "motion graphics", "art history"
        },
        "engineering": {
            "civil engineering", "mechanical engineering", "electrical engineering", "technical design",
            "cad", "structural analysis", "thermodynamics", "fluid dynamics", "materials science",
            "finite element analysis", "circuit design", "robotics", "control systems",
            "manufacturing", "industrial design", "system design", "prototyping",
            "mechanical design", "aerospace", "architecture", "3d modeling", "autocad"
        },
        "healthcare": {
            "patient care", "medical knowledge", "diagnosis", "treatment", "therapy",
            "nursing", "pharmacy", "clinical", "surgical", "radiology",
            "healthcare", "medical", "mental health", "telehealth", "rehabilitation",
            "anatomy", "physiology", "pathology", "emergency care", "primary care"
        },
        "education": {
            "teaching", "curriculum development", "instructional design", "assessment",
            "lesson planning", "training", "e-learning", "education technology", "pedagogy",
            "classroom management", "student engagement", "online teaching", "learning objectives",
            "educational psychology", "special education", "early childhood education", "adult learning"
        }
    }
    
    # Additional term mappings for cross-domain skills
    CROSS_DOMAIN_MAPPINGS = {
        # Design to Engineering mappings
        "graphic design": ["visual design", "illustration", "creative design"],
        "technical design": ["cad design", "engineering design", "technical drawing"],
        "ui/ux design": ["user interface design", "user experience design", "interaction design"],
        
        # Common abbreviations and alternative names
        "artificial intelligence": ["ai", "machine learning"],
        "user experience": ["ux", "ux design", "user research"],
        "user interface": ["ui", "ui design", "interface design"],
        "version control": ["git", "github", "version management"],
        "database management": ["sql", "database administration", "data management"],
        
        # Skill variations
        "web development": ["web design", "website development", "frontend development"],
        "mobile development": ["android development", "ios development", "app development"],
        "data analysis": ["data analytics", "statistical analysis", "data processing"]
    }
    
    @classmethod
    def set_progress_bars(cls, show_progress: bool):
        """
        Set whether to show progress bars during model encoding.
        
        Args:
            show_progress: Whether to show progress bars
        """
        cls._show_progress_bars = show_progress
        logger.info(f"Progress bars {'enabled' if show_progress else 'disabled'}")
    
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
            # Pass show_progress_bar flag to control whether progress bars are shown
            embedding = model.encode(
                text, 
                convert_to_tensor=False, 
                show_progress_bar=cls._show_progress_bars
            )
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
        
        # Check for exact domain term matches
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
                            domain_score += 0.25  # Increased from 0.2
                        else:
                            domain_score += 0.15  # Increased from 0.1
                    else:
                        # Single word term, check if it's a full word
                        skill_parts = skill_lower.split()
                        if term in skill_parts:
                            domain_score += 0.25  # Increased from 0.2
                        else:
                            domain_score += 0.15  # Increased from 0.1
            
            if domain_score > 0:
                bonuses[domain] = min(domain_score, 0.5)  # Cap at 0.5
        
        # Check for cross-domain mappings
        for primary_skill, related_skills in cls.CROSS_DOMAIN_MAPPINGS.items():
            if primary_skill.lower() in skill_lower:
                # If this is a primary term, add a bonus to help with cross-domain matching
                for domain in cls.DOMAIN_TERMS.keys():
                    if domain not in bonuses:
                        # Check if any domain terms contain elements of this skill
                        if any(term in primary_skill.lower() for term in cls.DOMAIN_TERMS[domain]):
                            bonuses[domain] = 0.3
            
            # Check if this skill matches any of the related skills
            for related_skill in related_skills:
                if related_skill.lower() in skill_lower:
                    # Find the domain of the primary skill
                    for domain in cls.DOMAIN_TERMS.keys():
                        if primary_skill.lower() in ' '.join(cls.DOMAIN_TERMS[domain]):
                            if domain in bonuses:
                                bonuses[domain] = max(bonuses[domain], 0.35)
                            else:
                                bonuses[domain] = 0.35
                
        return bonuses
    
    @classmethod
    def get_prioritized_domains(cls, skill: str) -> List[str]:
        """
        Get a prioritized list of domains that match the skill
        
        Args:
            skill: Skill name to analyze
            
        Returns:
            List of domain names sorted by match strength
        """
        bonuses = cls.get_domain_bonus(skill)
        
        # Sort domains by bonus score (descending)
        sorted_domains = sorted(bonuses.items(), key=lambda x: x[1], reverse=True)
        return [domain for domain, _ in sorted_domains]
    
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
        
        # Special case mappings for common alternative terms
        special_mappings = {
            "pentesting": "penetration testing",
            "pen testing": "penetration testing",
            "pentest": "penetration testing",
            "ui/ux": "ui/ux design",
            "ui design": "user interface design",
            "ux design": "user experience design",
            "graphic design": "visual design"
        }
        
        # Check for special mappings
        if skill_lower in special_mappings:
            mapped_skill = special_mappings[skill_lower]
            for ref_skill in reference_skills:
                if mapped_skill == ref_skill.lower():
                    result = (ref_skill, 1.0)  # Consider it a perfect match
                    cls._match_cache[cache_key] = result
                    return result
                    
        # Check cross-domain mappings for potential matches
        for primary_skill, related_skills in cls.CROSS_DOMAIN_MAPPINGS.items():
            if skill_lower in [s.lower() for s in related_skills] or skill_lower == primary_skill.lower():
                # Try to find matching reference skills that might be in a different domain
                all_related = related_skills + [primary_skill]
                for related in all_related:
                    for ref_skill in reference_skills:
                        if related.lower() == ref_skill.lower():
                            result = (ref_skill, 0.95)  # Strong but not perfect match
                            cls._match_cache[cache_key] = result
                            return result
        
        # Apply semantic matching
        best_match = None
        best_score = 0.0
        
        # Get domain bonuses for the input skill
        domain_bonuses = cls.get_domain_bonus(skill)
        
        # Prioritize matching within the same domain when possible
        prioritized_domains = cls.get_prioritized_domains(skill)
        
        # Group reference skills by their domains
        domain_grouped_refs = {}
        for ref_skill in reference_skills:
            ref_domains = cls.get_prioritized_domains(ref_skill)
            for domain in ref_domains:
                if domain not in domain_grouped_refs:
                    domain_grouped_refs[domain] = []
                domain_grouped_refs[domain].append(ref_skill)
        
        # Try to match within prioritized domains first
        for domain in prioritized_domains:
            if domain in domain_grouped_refs:
                for ref_skill in domain_grouped_refs[domain]:
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
        
        # If we didn't find a good match within prioritized domains, try all references
        if best_score < threshold:
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
    
    @classmethod
    def prioritize_missing_skills(cls, user_skills: Dict[str, int], 
                                target_skills: Dict[str, int],
                                threshold: float = 0.65) -> List[Dict[str, Any]]:
        """
        Analyze and prioritize missing skills based on importance and career relevance
        
        Args:
            user_skills: Dictionary of user skills and proficiency (1-100)
            target_skills: Dictionary of target skills and their importance (1-100)
            threshold: Minimum similarity score to consider a match
            
        Returns:
            List of missing skills sorted by priority with importance scores
        """
        # First find all matching skills
        matches = cls.weighted_skill_match(user_skills, target_skills, threshold)
        matched_target_skills = {match["target_skill"] for match in matches}
        
        # Identify missing skills
        missing = []
        for skill, importance in target_skills.items():
            if skill not in matched_target_skills:
                # Check if any user skill is somewhat similar but below threshold
                best_partial_match = None
                best_partial_score = 0
                
                for user_skill in user_skills:
                    _, score = cls.match_skill(user_skill, [skill], 0.4)  # Lower threshold for partial matches
                    if score > best_partial_score:
                        best_partial_match = user_skill
                        best_partial_score = score
                
                # Calculate priority based on importance and partial match score
                # Higher importance and lower partial match means higher priority
                priority_score = (importance / 100) * (1 - (best_partial_score * 0.5))
                
                # Get domain information
                domains = cls.get_prioritized_domains(skill)
                primary_domain = domains[0] if domains else "general"
                
                missing.append({
                    "skill": skill,
                    "importance": importance,
                    "closest_user_skill": best_partial_match,
                    "similarity": best_partial_score,
                    "priority_score": priority_score,
                    "domain": primary_domain
                })
        
        # Sort by priority score (higher is more important to learn)
        missing.sort(key=lambda x: x["priority_score"], reverse=True)
        return missing 