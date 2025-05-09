import numpy as np
import os
import time
import asyncio
from typing import Dict, List, Tuple, Any, Optional, Set, Union, Callable
from functools import lru_cache
import logging
from fuzzywuzzy import fuzz
import json
# Import sentence_transformers only when needed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConfig:
    """
    Configuration class for semantic model settings and versioning.
    """
    # Default model configurations
    DEFAULT_MODELS = {
        "small": "all-MiniLM-L6-v2",      # Fast, compact model (default)
        "medium": "all-mpnet-base-v2",     # Good balance of speed and accuracy
        "large": "all-distilroberta-v1",   # More accurate but slower
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2"  # For multi-language support
    }
    
    def __init__(
        self, 
        model_name: str = "small",
        custom_model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        warmup_on_init: bool = False,
        warmup_texts: Optional[List[str]] = None,
        enable_progress_bars: bool = False
    ):
        """
        Initialize model configuration.
        
        Args:
            model_name: Name or size of model to use (small, medium, large, multilingual, or custom)
            custom_model_path: Path to custom model if model_name is "custom"
            cache_dir: Directory to cache models
            warmup_on_init: Whether to warmup model on initialization
            warmup_texts: Sample texts for model warmup
            enable_progress_bars: Whether to show progress bars during encoding
        """
        self.model_version = model_name
        
        # Resolve model name from predefined models or use custom path
        if custom_model_path:
            self.model_path = custom_model_path
        elif model_name in self.DEFAULT_MODELS:
            self.model_path = self.DEFAULT_MODELS[model_name]
        else:
            # Assume model_name is a direct HuggingFace model identifier
            self.model_path = model_name
            
        self.cache_dir = cache_dir
        self.warmup_on_init = warmup_on_init
        self.warmup_texts = warmup_texts or [
            "programming skills python javascript", 
            "data analysis statistics machine learning",
            "project management leadership"
        ]
        self.enable_progress_bars = enable_progress_bars
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            "model_version": self.model_version,
            "model_path": self.model_path,
            "cache_dir": self.cache_dir,
            "warmup_on_init": self.warmup_on_init,
            "enable_progress_bars": self.enable_progress_bars
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary"""
        return cls(
            model_name=config_dict.get("model_version", "small"),
            custom_model_path=config_dict.get("model_path"),
            cache_dir=config_dict.get("cache_dir"),
            warmup_on_init=config_dict.get("warmup_on_init", False),
            enable_progress_bars=config_dict.get("enable_progress_bars", False)
        )
        
    @classmethod
    def from_json(cls, json_path: str) -> 'ModelConfig':
        """Load configuration from JSON file"""
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Failed to load config from {json_path}: {e}")
            return cls()  # Return default config on error
            
    def save_to_json(self, json_path: str) -> bool:
        """Save configuration to JSON file"""
        try:
            with open(json_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {json_path}: {e}")
            return False

class SemanticMatcher:
    """
    Advanced skill matching using semantic similarity with transformers.
    Provides more accurate matching for specialized terminology by considering
    semantic meaning rather than just string similarity.
    """
    
    # Singleton model instance for efficiency
    _model = None
    
    # Model configuration
    _model_config = ModelConfig()
    
    # Cache for embeddings to avoid redundant computation
    _embedding_cache = {}
    
    # Cache for skill matching results
    _match_cache = {}
    
    # Flag to control progress bar display (should be disabled for API)
    _show_progress_bars = False
    
    # Default matching thresholds and weights that can be configured
    DEFAULT_MATCHING_CONFIG = {
        "similarity_threshold": 0.87,           # Increased from 0.85
        "partial_match_threshold": 0.65,        # Increased from 0.60
        "domain_bonus_cap": 0.30,               # Reduced from 0.35
        "cross_domain_bonus": 0.10,             # Reduced from 0.15
        "cross_domain_penalty": 0.35,           # Increased from 0.20 to be more strict
        "exact_match_score": 1.0,
        "special_mapping_score": 0.95,
        "specialization_threshold": 0.75        # New parameter for specialization-specific matching
    }
    
    # Current matching configuration
    _matching_config = DEFAULT_MATCHING_CONFIG.copy()
    
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
        "legal": {
            "legal", "law", "litigation", "corporate law", "compliance", "regulatory", "regulations",
            "contracts", "intellectual property", "patent", "trademark", "copyright", "licensing",
            "legal research", "case law", "jurisprudence", "legal writing", "legal analysis",
            "legal counsel", "attorney", "lawyer", "solicitor", "barrister", "judicial",
            "court proceedings", "trial", "deposition", "corporate governance", "due diligence",
            "mergers", "acquisitions", "legal documentation", "arbitration", "mediation",
            "dispute resolution", "negotiation", "tort", "civil law", "criminal law", "family law",
            "real estate law", "commercial law", "business law", "legal advising", "paralegal",
            "legal ethics", "legal workflow", "legislative", "case management", "legal drafting"
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
        "data analysis": ["data analytics", "statistical analysis", "data processing"],
        
        # Legal domain specific mappings
        "legal research": ["case law research", "statutory research", "legal analysis"],
        "corporate law": ["business law", "commercial law", "corporate governance"],
        "litigation": ["trial practice", "court proceedings", "dispute resolution"],
        "regulatory compliance": ["compliance", "regulatory affairs", "governance"],
        "contract management": ["contract review", "contract drafting", "legal documentation"],
        "legal writing": ["legal drafting", "brief writing", "legal documentation"],
        
        # Prevent cross-domain confusion between legal and cybersecurity
        "case management": ["legal case management", "caseload management"],
        "compliance": ["regulatory compliance", "legal compliance"]
    }
    
    @classmethod
    def set_progress_bars(cls, show_progress: bool):
        """
        Set whether to show progress bars during model encoding.
        
        Args:
            show_progress: Whether to show progress bars
        """
        cls._show_progress_bars = show_progress
        cls._model_config.enable_progress_bars = show_progress
        logger.info(f"Progress bars {'enabled' if show_progress else 'disabled'}")
    
    @classmethod
    def configure_model(cls, config: Union[ModelConfig, Dict[str, Any]]) -> bool:
        """
        Configure the model with new settings. If the model is already loaded,
        this will reload it with the new configuration.
        
        Args:
            config: ModelConfig instance or dict with configuration parameters
            
        Returns:
            Boolean indicating if configuration was successful
        """
        try:
            # Convert dict to ModelConfig if needed
            if isinstance(config, dict):
                config = ModelConfig.from_dict(config)
                
            # Store the new configuration
            cls._model_config = config
            cls._show_progress_bars = config.enable_progress_bars
            
            # Clear existing model to force reload with new config
            cls._model = None
            
            # Clear caches
            cls.clear_cache()
            
            # Optionally warmup the model
            if config.warmup_on_init:
                cls.warmup_model()
                
            return True
        except Exception as e:
            logger.error(f"Failed to configure model: {e}")
            return False
    
    @classmethod
    def get_model(cls):
        """Get or initialize the sentence transformer model"""
        if cls._model is None:
            try:
                # Import here to avoid circular imports
                from sentence_transformers import SentenceTransformer
                
                # Use configuration to determine which model to load
                model_path = cls._model_config.model_path
                cache_dir = cls._model_config.cache_dir
                
                start_time = time.time()
                logger.info(f"Loading sentence transformer model: {model_path}")
                
                # Load the model with appropriate parameters
                kwargs = {"cache_folder": cache_dir} if cache_dir else {}
                cls._model = SentenceTransformer(model_path, **kwargs)
                
                load_time = time.time() - start_time
                logger.info(f"Model loaded in {load_time:.2f} seconds")
                logger.info(f"Progress bars are {'enabled' if cls._show_progress_bars else 'disabled'}")
                
                # Perform warmup if configured
                if cls._model_config.warmup_on_init:
                    cls.warmup_model()
                    
            except Exception as e:
                logger.error(f"Error loading sentence transformer model: {e}")
                # Fallback to None - will use fuzzy matching instead
                cls._model = None
        return cls._model
    
    @classmethod
    def warmup_model(cls, warmup_texts: Optional[List[str]] = None):
        """
        Warmup the model by encoding sample texts to initialize internal caches.
        This can improve performance for subsequent calls.
        
        Args:
            warmup_texts: Optional list of texts to encode (uses defaults if None)
        """
        model = cls.get_model()
        if model is None:
            logger.warning("Cannot warmup model - no model loaded")
            return
            
        # Use provided texts or fall back to default warmup texts
        texts = warmup_texts or cls._model_config.warmup_texts
        
        logger.info(f"Warming up model with {len(texts)} sample texts")
        start_time = time.time()
        
        try:
            # Encode the warmup texts
            model.encode(
                texts,
                convert_to_tensor=False,
                show_progress_bar=cls._show_progress_bars
            )
            
            warmup_time = time.time() - start_time
            logger.info(f"Model warmup completed in {warmup_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during model warmup: {e}")
    
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
    def get_embeddings_batch(cls, texts: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Get embeddings for a batch of texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Dictionary mapping texts to their embeddings
        """
        # Check which texts are not in cache
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text not in cls._embedding_cache:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # If all texts are cached, return from cache
        if not uncached_texts:
            return {text: cls._embedding_cache.get(text) for text in texts}
        
        # Get model
        model = cls.get_model()
        if model is None:
            # Fallback to individual processing using fuzzy matching later
            return {text: None for text in texts}
        
        # Process uncached texts in batch
        try:
            start_time = time.time()
            
            # Batch encode the texts
            batch_embeddings = model.encode(
                uncached_texts,
                convert_to_tensor=False,
                show_progress_bar=cls._show_progress_bars
            )
            
            # Update cache with new embeddings
            for i, text in enumerate(uncached_texts):
                cls._embedding_cache[text] = batch_embeddings[i]
                
            batch_time = time.time() - start_time
            logger.debug(f"Batch processed {len(uncached_texts)} texts in {batch_time:.2f}s ({batch_time/len(uncached_texts):.4f}s per text)")
            
        except Exception as e:
            logger.error(f"Error batch processing embeddings: {e}")
            # Return Nones for uncached texts
            for text in uncached_texts:
                cls._embedding_cache[text] = None
        
        # Return all embeddings
        return {text: cls._embedding_cache.get(text) for text in texts}
    
    @classmethod
    async def get_embedding_async(cls, text: str) -> Optional[np.ndarray]:
        """
        Asynchronous version of get_embedding for use in async APIs.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if model not available
        """
        # Check cache first
        if text in cls._embedding_cache:
            return cls._embedding_cache[text]
        
        # Run embedding generation in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, cls.get_embedding, text)
    
    @classmethod
    async def get_embeddings_batch_async(cls, texts: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Asynchronous version of batch embedding generation.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Dictionary mapping texts to their embeddings
        """
        # Run batch processing in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, cls.get_embeddings_batch, texts)
    
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
    async def semantic_similarity_async(cls, text1: str, text2: str) -> float:
        """
        Asynchronous version of semantic similarity calculation.
        
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
        
        # Get embeddings asynchronously
        emb1 = await cls.get_embedding_async(text1)
        emb2 = await cls.get_embedding_async(text2)
        
        # If either embedding failed, fall back to fuzzy matching
        if emb1 is None or emb2 is None:
            return fuzz.token_sort_ratio(text1.lower(), text2.lower()) / 100
        
        # Run the similarity calculation in a thread pool
        loop = asyncio.get_event_loop()
        try:
            from sentence_transformers import util
            result = await loop.run_in_executor(
                None,
                lambda: float(util.cos_sim(emb1, emb2)[0][0])
            )
            return result
        except Exception:
            # Fallback to fuzzy matching
            return fuzz.token_sort_ratio(text1.lower(), text2.lower()) / 100
    
    @classmethod
    def configure_matching(cls, config: Dict[str, float]) -> None:
        """
        Configure the matching parameters and thresholds.
        
        Args:
            config: Dictionary of configuration parameters
        """
        # Update the matching configuration with provided values
        for key, value in config.items():
            if key in cls._matching_config:
                cls._matching_config[key] = value
                
        # Clear the match cache to force recalculation with new parameters
        cls._match_cache.clear()
        
        logger.info(f"Updated matching configuration: {cls._matching_config}")
        
    @classmethod
    def reset_matching_config(cls) -> None:
        """Reset matching configuration to defaults"""
        cls._matching_config = cls.DEFAULT_MATCHING_CONFIG.copy()
        cls._match_cache.clear()
        logger.info("Reset matching configuration to defaults")
    
    @classmethod
    def get_matching_config(cls) -> Dict[str, float]:
        """Get current matching configuration"""
        return cls._matching_config.copy()
        
    @classmethod
    def configure_domains(cls, domain_config: Dict[str, Set[str]], 
                        add_to_existing: bool = True) -> None:
        """
        Configure or extend the domain terms.
        
        Args:
            domain_config: Dictionary of domain names to sets of terms
            add_to_existing: Whether to add to existing domains or replace them
        """
        if add_to_existing:
            # Merge with existing domains
            for domain, terms in domain_config.items():
                if domain in cls.DOMAIN_TERMS:
                    cls.DOMAIN_TERMS[domain].update(terms)
                else:
                    cls.DOMAIN_TERMS[domain] = terms
        else:
            # Replace domains
            cls.DOMAIN_TERMS = domain_config
            
        # Clear caches
        cls._match_cache.clear()
        
        logger.info(f"Updated domain configuration with {len(domain_config)} domains")
        
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
        
        # Special handling for misspelled legal terms
        common_legal_misspellings = {
            "reguletory": "regulatory",
            "leagal": "legal",
            "litagation": "litigation",
            "complyance": "compliance",
            "corporat": "corporate",
            "atturney": "attorney",
            "attorny": "attorney",
            "lawer": "lawyer",
            "lawyr": "lawyer",
            "legl": "legal"
        }
        
        # Check for misspellings and correct them for domain detection
        corrected_skill = skill_lower
        for misspelled, correct in common_legal_misspellings.items():
            if misspelled in skill_lower:
                corrected_skill = skill_lower.replace(misspelled, correct)
                # Add legal domain bonus immediately if legal misspelling found
                bonuses["legal"] = 0.35
                
        # Check for exact domain term matches
        for domain, terms in cls.DOMAIN_TERMS.items():
            # Check for exact matches in domain terms
            domain_score = 0
            for term in terms:
                # Check in both original and corrected skill
                for check_skill in [skill_lower, corrected_skill]:
                    if term in check_skill:
                        # Give higher bonus for full word matches
                        term_parts = term.split()
                        if len(term_parts) > 1:
                            # Multi-word term, check if all words appear
                            if all(part in check_skill.split() for part in term_parts):
                                domain_score += 0.25  # Increased from 0.2
                            else:
                                domain_score += 0.15  # Increased from 0.1
                        else:
                            # Single word term, check if it's a full word
                            skill_parts = check_skill.split()
                            if term in skill_parts:
                                domain_score += 0.25  # Increased from 0.2
                            else:
                                domain_score += 0.15  # Increased from 0.1
            
            if domain_score > 0:
                bonuses[domain] = min(domain_score, 0.5)  # Cap at 0.5
        
        # Special handling for "compliance" term - ensure proper classification
        compliance_terms = ["compliance", "regulatory compliance", "legal compliance"]
        if any(term in skill_lower for term in compliance_terms):
            # Examine the context to determine if it's legal or cybersecurity
            legal_indicators = ["regulatory", "legal", "law", "policy", "governance", "corporate", 
                              "regulation"]
            security_indicators = ["security", "cyber", "network", "it security", "technical", 
                                 "protection", "threat"]
            
            # Look for indicators in whole skill text
            has_legal_context = any(indicator in skill_lower for indicator in legal_indicators)
            has_security_context = any(indicator in skill_lower for indicator in security_indicators)
            
            # If clear legal indicators, strongly boost legal domain and reduce cybersecurity
            if has_legal_context and not has_security_context:
                bonuses["legal"] = max(bonuses.get("legal", 0), 0.45)
                # Reduce or eliminate cybersecurity bonus if present
                if "cybersecurity" in bonuses:
                    bonuses["cybersecurity"] *= 0.3  # Severely reduce cybersecurity score
            
            # If clear security indicators, boost cybersecurity domain
            elif has_security_context and not has_legal_context:
                bonuses["cybersecurity"] = max(bonuses.get("cybersecurity", 0), 0.45)
            
            # If ambiguous (both or neither context detected), slightly favor legal
            elif "compliance" in skill_lower and "regulatory" in skill_lower:
                bonuses["legal"] = max(bonuses.get("legal", 0), 0.4)
                # Reduce cybersecurity bonus if present
                if "cybersecurity" in bonuses:
                    bonuses["cybersecurity"] = min(bonuses.get("cybersecurity", 0) * 0.7, 0.25)
            elif "compliance" in skill_lower:
                # Default slightly to legal for plain "compliance" with no context
                bonuses["legal"] = max(bonuses.get("legal", 0), 0.3)
                # Slightly reduce cybersecurity if present
                if "cybersecurity" in bonuses:
                    bonuses["cybersecurity"] = min(bonuses.get("cybersecurity", 0) * 0.8, 0.25)
        
        # Handle "case management" disambiguation
        if "case management" in skill_lower or ("case" in skill_lower and "management" in skill_lower):
            # Check for legal context
            legal_indicators = ["legal", "law", "litigation", "court", "attorney", "client", "lawsuit"]
            tech_indicators = ["incident", "security", "cyber", "ticket", "technical", "it support"]
            
            has_legal_context = any(indicator in skill_lower for indicator in legal_indicators)
            has_tech_context = any(indicator in skill_lower for indicator in tech_indicators)
            
            if has_legal_context and not has_tech_context:
                # This is clearly legal case management
                bonuses["legal"] = max(bonuses.get("legal", 0), 0.45)
                
                # Remove cybersecurity bonus if present
                if "cybersecurity" in bonuses:
                    del bonuses["cybersecurity"]
            
            elif has_tech_context and not has_legal_context:
                # This is clearly technical case/incident management
                bonuses["cybersecurity"] = max(bonuses.get("cybersecurity", 0), 0.45)
            
            elif not has_legal_context and not has_tech_context:
                # Default behavior: favor legal interpretation for "case management" without context
                bonuses["legal"] = max(bonuses.get("legal", 0), 0.35)
                # Reduce cybersecurity score dramatically
                if "cybersecurity" in bonuses:
                    bonuses["cybersecurity"] *= 0.4
        
        # Deal with irrelevant skills - if skills like cooking, gardening, etc. are detected
        irrelevant_terms = ["cooking", "gardening", "hobbies", "sports", "gaming"]
        if any(term in skill_lower for term in irrelevant_terms):
            # Clear all technical domain bonuses - irrelevant skills should not match technical domains
            for domain in ["cybersecurity", "programming", "data_science", "engineering"]:
                if domain in bonuses:
                    del bonuses[domain]
            
            # Give small boost to general domains
            bonuses["general"] = 0.2
        
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
                   threshold: float = None, use_semantic: bool = True) -> Tuple[Optional[str], float]:
        """
        Match a skill against a list of reference skills with domain awareness
        
        Args:
            skill: The user's skill to match
            reference_skills: List of reference skills to match against
            threshold: Optional override for similarity threshold
            use_semantic: Whether to use semantic matching (vs fuzzy string matching)
            
        Returns:
            Tuple of (best_match, score) or (None, 0.0) if no match
        """
        # Use cached result if available - key includes all parameters
        cache_key = f"{skill}::{','.join(sorted(reference_skills))}::{threshold}::{use_semantic}"
        if cache_key in cls._match_cache:
            return cls._match_cache[cache_key]
            
        if not reference_skills:
            return None, 0.0
            
        # Use configured threshold or default
        match_threshold = threshold if threshold is not None else cls._matching_config["similarity_threshold"]
        exact_score = cls._matching_config["exact_match_score"]
        
        # Try exact match first (case insensitive)
        for ref_skill in reference_skills:
            if skill.lower() == ref_skill.lower():
                cls._match_cache[cache_key] = (ref_skill, exact_score)
                return ref_skill, exact_score
        
        # Check for special mappings
        special_mappings = cls.CROSS_DOMAIN_MAPPINGS
        mapping_score = cls._matching_config["special_mapping_score"]
        
        for ref_skill in reference_skills:
            if ref_skill in special_mappings and skill in special_mappings[ref_skill]:
                cls._match_cache[cache_key] = (ref_skill, mapping_score)
                return ref_skill, mapping_score
                
            # Also check the reverse direction
            if skill in special_mappings and ref_skill in special_mappings[skill]:
                cls._match_cache[cache_key] = (ref_skill, mapping_score)
                return ref_skill, mapping_score
        
        # If semantic matching is disabled, fall back to fuzzy string matching
        if not use_semantic:
            best_match = None
            best_score = 0.0
            
            for ref_skill in reference_skills:
                # Use token_sort_ratio to handle word order differences
                score = fuzz.token_sort_ratio(skill.lower(), ref_skill.lower()) / 100.0
                
                if score > best_score:
                    best_match = ref_skill
                    best_score = score
            
            # Only return a match if it exceeds the threshold
            if best_score >= match_threshold:
                cls._match_cache[cache_key] = (best_match, best_score)
                return best_match, best_score
            else:
                cls._match_cache[cache_key] = (None, 0.0)
                return None, 0.0
        
        # Use semantic similarity for improved matching
        best_match = None
        best_score = 0.0
        
        # Compare the user skill to each reference skill
        for ref_skill in reference_skills:
            # Compute semantic similarity between the user skill and reference skill
            similarity = cls.semantic_similarity(skill, ref_skill)
            
            # Get domain bonuses
            user_domains = cls.get_prioritized_domains(skill)
            ref_domains = cls.get_prioritized_domains(ref_skill)
            
            # Apply domain-specific logic
            final_score = similarity
            
            # If domains are available for both skills
            if user_domains and ref_domains:
                common_domains = set(user_domains).intersection(set(ref_domains))
                
                # Apply domain bonus for skills in the same domain
                if common_domains:
                    # Higher bonus for skills in same primary domain
                    if user_domains[0] == ref_domains[0]:
                        domain_bonus = min(cls._matching_config["domain_bonus_cap"], 
                                         0.20 + (similarity * 0.15))  # Scaled by base similarity
                        final_score = min(0.98, similarity + domain_bonus)
                    else:
                        # Lower bonus for skills in same secondary domain
                        domain_bonus = min(cls._matching_config["domain_bonus_cap"] / 2, 
                                         0.10 + (similarity * 0.10))
                        final_score = min(0.95, similarity + domain_bonus)
                else:
                    # Skills are from different domains - apply PENALTY if they're completely unrelated
                    # This is a key addition to prevent cross-domain matching
                    incompatible_domains = {
                        ("legal", "programming"),
                        ("legal", "cybersecurity"),
                        ("legal", "data_science"),
                        ("legal", "engineering"),
                        ("legal", "management"),  # Added management
                        ("legal", "supply_chain"),
                        ("legal", "operations"),  # Added operations
                        ("healthcare", "programming"),
                        ("healthcare", "cybersecurity"),  # Added healthcare-cybersecurity
                        ("legal", "supply_chain"),
                        ("arts", "cybersecurity"),
                        ("arts", "engineering"),  # Added arts-engineering
                        ("marketing", "cybersecurity"),  # Added marketing-cybersecurity
                        ("marketing", "programming")  # Added marketing-programming
                    }
                    
                    # Check for incompatible domain pairs - bidirectional check
                    for user_domain in user_domains[:2]:  # Consider only top 2 domains
                        for ref_domain in ref_domains[:2]:  # Consider only top 2 domains
                            if (user_domain, ref_domain) in incompatible_domains or \
                               (ref_domain, user_domain) in incompatible_domains:
                                # Apply penalty for incompatible domains
                                domain_penalty = cls._matching_config["cross_domain_penalty"]
                                final_score = max(0.05, similarity - domain_penalty)
                                break
                                
                    # Check for patent law special case
                    if "patent" in ref_skill.lower() and "legal" in user_domains:
                        # For patent law, require technical context
                        if not any(d in user_domains for d in ["engineering", "programming", "data_science", "scientific"]):
                            # Stronger penalty for non-technical legal skills matching to patent
                            final_score = max(0.05, similarity - (cls._matching_config["cross_domain_penalty"] * 1.5))
            
            # Special case for "case management" to avoid matching to "supply chain management"
            if "case management" in skill.lower() and "management" in ref_skill.lower():
                if any(term in ref_skill.lower() for term in ["supply chain", "operations", "logistics"]):
                    # Apply significant penalty for case management matching to supply chain
                    final_score = max(0.05, similarity - (cls._matching_config["cross_domain_penalty"] * 1.5))
            
            # Update best match if this is the highest score
            if final_score > best_score:
                best_match = ref_skill
                best_score = final_score
        
        # Apply threshold to ensure quality matches
        if best_score >= match_threshold:
            cls._match_cache[cache_key] = (best_match, best_score)
            return best_match, best_score
        else:
            cls._match_cache[cache_key] = (None, 0.0)
            return None, 0.0
    
    @classmethod
    def cluster_skills(cls, skills: List[str], n_clusters: Optional[int] = None, 
                      min_similarity: float = 0.7) -> Dict[str, List[str]]:
        """
        Cluster skills based on semantic similarity.
        
        Args:
            skills: List of skill names to cluster
            n_clusters: Number of clusters to create (if None, auto-determined)
            min_similarity: Minimum similarity to consider two skills in same cluster
            
        Returns:
            Dictionary mapping cluster names to lists of skills
        """
        if not skills:
            return {}
            
        try:
            # Generate embeddings for all skills
            skill_embeddings = cls.get_embeddings_batch(skills)
            
            # Filter out skills that couldn't be embedded
            valid_skills = []
            valid_embeddings = []
            
            for skill, embedding in skill_embeddings.items():
                if embedding is not None:
                    valid_skills.append(skill)
                    valid_embeddings.append(embedding)
            
            if not valid_skills:
                logger.warning("No valid embeddings could be generated for skills")
                return {"Unclustered": skills}
                
            # Convert to numpy array
            embeddings_array = np.array(valid_embeddings)
            
            # Import clustering algorithm
            try:
                from sklearn.cluster import AgglomerativeClustering
                from scipy.cluster.hierarchy import linkage, fcluster
                
                # If n_clusters is not specified, use hierarchical clustering with distance threshold
                if n_clusters is None:
                    # Convert similarity threshold to distance threshold
                    distance_threshold = 1.0 - min_similarity
                    
                    # Compute linkage matrix
                    Z = linkage(embeddings_array, method='ward')
                    
                    # Form flat clusters
                    labels = fcluster(Z, t=distance_threshold, criterion='distance') - 1
                    unique_clusters = len(set(labels))
                    logger.info(f"Auto-determined {unique_clusters} clusters with min_similarity={min_similarity}")
                else:
                    # Use agglomerative clustering with specified number of clusters
                    clustering = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = clustering.fit_predict(embeddings_array)
                
                # Group skills by cluster
                clusters = {}
                
                # First determine representative skill for each cluster
                for cluster_id in set(labels):
                    cluster_mask = labels == cluster_id
                    cluster_skills = [valid_skills[i] for i, is_in_cluster in enumerate(cluster_mask) if is_in_cluster]
                    
                    # Find centroid of the cluster
                    cluster_embeddings = embeddings_array[cluster_mask]
                    centroid = np.mean(cluster_embeddings, axis=0)
                    
                    # Find skill closest to centroid
                    closest_idx = np.argmin(np.sum((cluster_embeddings - centroid) ** 2, axis=1))
                    representative_skill = cluster_skills[closest_idx]
                    
                    # Create more descriptive cluster name based on domains
                    domains = cls.get_prioritized_domains(representative_skill)
                    cluster_name = f"{domains[0].capitalize() if domains else 'General'}: {representative_skill}"
                    
                    clusters[cluster_name] = cluster_skills
                
                # Add any skills that couldn't be embedded to an "Unclustered" group
                unclustered = [skill for skill in skills if skill not in valid_skills]
                if unclustered:
                    clusters["Unclustered"] = unclustered
                
                return clusters
                
            except ImportError:
                logger.warning("scikit-learn not available for clustering, falling back to simple method")
                # Fall back to simple clustering
                return cls._simple_clustering(skills, min_similarity)
                
        except Exception as e:
            logger.error(f"Error during skill clustering: {e}")
            return {"Unclustered": skills}
            
    @classmethod
    def _simple_clustering(cls, skills: List[str], min_similarity: float = 0.7) -> Dict[str, List[str]]:
        """
        Simple greedy clustering algorithm for when scikit-learn is not available.
        
        Args:
            skills: List of skill names to cluster
            min_similarity: Minimum similarity to consider two skills in same cluster
            
        Returns:
            Dictionary mapping cluster names to lists of skills
        """
        clusters = {}
        processed = set()
        
        for skill in skills:
            if skill in processed:
                continue
                
            # This skill becomes a new cluster center
            cluster = [skill]
            processed.add(skill)
            
            # Find similar skills
            for other_skill in skills:
                if other_skill in processed:
                    continue
                    
                similarity = cls.semantic_similarity(skill, other_skill)
                if similarity >= min_similarity:
                    cluster.append(other_skill)
                    processed.add(other_skill)
            
            # Name the cluster after the first skill and its domain
            domains = cls.get_prioritized_domains(skill)
            cluster_name = f"{domains[0].capitalize() if domains else 'General'}: {skill}"
            clusters[cluster_name] = cluster
        
        return clusters
        
    @classmethod
    def analyze_skill_trends(cls, skill_snapshots: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze skill trends over time.
        
        Args:
            skill_snapshots: Dictionary mapping timestamps to skill dictionaries
                Each skill dictionary maps skill names to proficiency levels
                
        Returns:
            Dictionary with trend analysis results
        """
        if not skill_snapshots or len(skill_snapshots) < 2:
            return {"error": "Need at least two time points for trend analysis"}
            
        # Extract timestamps and ensure they're in order
        timestamps = sorted(skill_snapshots.keys())
        
        # Get all unique skills across all snapshots
        all_skills = set()
        for snapshot in skill_snapshots.values():
            all_skills.update(snapshot.keys())
            
        # Initialize results dictionary
        trends = {
            "timestamps": timestamps,
            "skills": {},
            "emerging_skills": [],
            "declining_skills": [],
            "steady_skills": [],
            "new_skills": [],
            "abandoned_skills": []
        }
        
        # Analyze each skill
        for skill in all_skills:
            # Track proficiency over time
            proficiency_trend = []
            
            for timestamp in timestamps:
                snapshot = skill_snapshots[timestamp]
                proficiency = snapshot.get(skill, 0)
                proficiency_trend.append(proficiency)
            
            # Skip skills that aren't in either the first or last snapshot
            if proficiency_trend[0] == 0 and proficiency_trend[-1] == 0:
                continue
                
            # Calculate trend metrics
            is_new = proficiency_trend[0] == 0 and proficiency_trend[-1] > 0
            is_abandoned = proficiency_trend[0] > 0 and proficiency_trend[-1] == 0
            
            # For skills present in both first and last snapshots
            if not is_new and not is_abandoned:
                # Calculate growth rate
                initial = max(1, proficiency_trend[0])  # Avoid division by zero
                final = proficiency_trend[-1]
                growth_rate = (final - initial) / initial
                
                # Categorize based on growth rate
                if growth_rate > 0.1:  # More than 10% growth
                    trends["emerging_skills"].append((skill, growth_rate))
                elif growth_rate < -0.1:  # More than 10% decline
                    trends["declining_skills"].append((skill, growth_rate))
                else:  # Relatively stable
                    trends["steady_skills"].append(skill)
            elif is_new:
                trends["new_skills"].append(skill)
            elif is_abandoned:
                trends["abandoned_skills"].append(skill)
            
            # Add detailed trend data for this skill
            trends["skills"][skill] = {
                "trend": proficiency_trend,
                "is_new": is_new,
                "is_abandoned": is_abandoned,
                "growth_rate": (proficiency_trend[-1] - proficiency_trend[0]) / max(1, proficiency_trend[0])
                               if not is_new else float('inf')
            }
        
        # Sort emerging and declining skills by absolute growth rate
        trends["emerging_skills"] = [skill for skill, _ in 
                                    sorted(trends["emerging_skills"], 
                                          key=lambda x: x[1], reverse=True)]
        
        trends["declining_skills"] = [skill for skill, _ in 
                                     sorted(trends["declining_skills"], 
                                           key=lambda x: x[1])]
        
        return trends
        
    @classmethod
    def suggest_skill_development_path(cls, 
                                     current_skills: Dict[str, int],
                                     target_job_skills: Dict[str, int],
                                     max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """
        Suggest an optimal skill development path based on current skills and career goals.
        
        Args:
            current_skills: Dictionary of current skills and proficiency
            target_job_skills: Dictionary of skills needed for target job
            max_suggestions: Maximum number of skill suggestions
            
        Returns:
            List of suggested skills to develop with rationale
        """
        # First identify missing skills
        missing_skills = cls.prioritize_missing_skills(
            current_skills, 
            target_job_skills,
            threshold=cls._matching_config["similarity_threshold"]
        )
        
        # Only consider top missing skills
        top_missing = missing_skills[:max_suggestions*2]  # Get more than needed to allow for filtering
        
        suggestions = []
        
        for missing in top_missing:
            if len(suggestions) >= max_suggestions:
                break
                
            skill = missing["skill"]
            importance = missing["importance"]
            closest_user_skill = missing["closest_user_skill"]
            similarity = missing["similarity"]
            domain = missing["domain"]
            
            # Calculate how learnable this skill is based on existing skills
            learnability = similarity * 0.7  # Base learnability on similarity to closest skill
            
            # Bonus for skills in domains user already has skills in
            user_domains = set()
            for user_skill in current_skills:
                domains = cls.get_prioritized_domains(user_skill)
                if domains:
                    user_domains.add(domains[0])
            
            if domain in user_domains:
                learnability += 0.2
                
            # Calculate final priority score that balances importance with learnability
            priority = (importance / 100) * 0.7 + learnability * 0.3
            
            # Find prerequisite skills that might help learn this skill
            prerequisites = []
            for user_skill, proficiency in current_skills.items():
                user_skill_domains = cls.get_prioritized_domains(user_skill)
                if user_skill_domains and user_skill_domains[0] == domain:
                    # This skill is in the same domain and might be a prerequisite
                    prereq_sim = cls.semantic_similarity(user_skill, skill)
                    if prereq_sim > 0.4:  # Only consider relevant skills
                        prerequisites.append({
                            "skill": user_skill,
                            "proficiency": proficiency,
                            "similarity": prereq_sim
                        })
            
            # Sort prerequisites by relevance
            prerequisites.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Add suggestion
            suggestions.append({
                "skill": skill,
                "importance": importance,
                "learnability": learnability,
                "priority": priority,
                "domain": domain,
                "closest_existing_skill": closest_user_skill,
                "similarity_to_existing": similarity,
                "relevant_prerequisites": prerequisites[:3]  # Top 3 prerequisites
            })
            
        # Sort suggestions by priority
        suggestions.sort(key=lambda x: x["priority"], reverse=True)
        
        return suggestions[:max_suggestions]
    
    @classmethod
    def weighted_skill_match(cls, user_skills: Dict[str, int], 
                           target_skills: Dict[str, int],
                           threshold: float = None) -> List[Dict[str, Any]]:
        """
        Find matches between user skills and target skills with weighted scoring
        
        Args:
            user_skills: Dictionary of user skills and proficiency (1-100)
            target_skills: Dictionary of target skills and their importance (1-100)
            threshold: Minimum similarity score (0-1) to consider a match
            
        Returns:
            List of matched skills with details and scores
        """
        # Use configured threshold if none provided
        if threshold is None:
            threshold = cls._matching_config["similarity_threshold"]
            
        matches = []
        
        # Get all user skills and target skills for reference
        user_skill_names = list(user_skills.keys())
        target_skill_names = list(target_skills.keys())
        
        # Process in batch for better performance
        all_user_skills = list(user_skills.keys())
        
        for target_skill, importance in target_skills.items():
            # Find the best matching user skill
            best_match = None
            best_score = 0.0
            best_proficiency = 0
            
            # Get embeddings for the target skill and all user skills
            all_skills = [target_skill] + all_user_skills
            embeddings = cls.get_embeddings_batch(all_skills)
            
            # If we have valid embeddings
            if embeddings.get(target_skill) is not None:
                # Compare target skill with each user skill
                for user_skill, proficiency in user_skills.items():
                    if embeddings.get(user_skill) is not None:
                        # Use semantic similarity with embeddings we already retrieved
                        try:
                            from sentence_transformers import util
                            score = float(util.cos_sim(embeddings[target_skill], 
                                                     embeddings[user_skill])[0][0])
                        except Exception:
                            # Fallback to direct call if something went wrong
                            _, score = cls.match_skill(user_skill, [target_skill], threshold)
                    else:
                        # Fallback to direct call
                        _, score = cls.match_skill(user_skill, [target_skill], threshold)
                        
                    if score > best_score:
                        best_match = user_skill
                        best_score = score
                        best_proficiency = proficiency
            else:
                # Fallback to individual processing
                for user_skill, proficiency in user_skills.items():
                    matched_skill, score = cls.match_skill(user_skill, [target_skill], threshold)
                    if matched_skill and score > best_score:
                        best_match = user_skill
                        best_score = score
                        best_proficiency = proficiency
            
            # If we found a match
            if best_match and best_score >= threshold:
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
                                threshold: float = None) -> List[Dict[str, Any]]:
        """
        Analyze and prioritize missing skills based on importance and career relevance
        
        Args:
            user_skills: Dictionary of user skills and proficiency (1-100)
            target_skills: Dictionary of target skills and their importance (1-100)
            threshold: Minimum similarity score to consider a match
            
        Returns:
            List of missing skills sorted by priority with importance scores
        """
        # Use configured threshold if none provided
        if threshold is None:
            threshold = cls._matching_config["similarity_threshold"]
            
        # First find all matching skills
        matches = cls.weighted_skill_match(user_skills, target_skills, threshold)
        matched_target_skills = {match["target_skill"] for match in matches}
        
        # Identify missing skills
        missing = []
        
        # Get all user skill embeddings at once for better performance
        all_user_skills = list(user_skills.keys())
        user_embeddings = cls.get_embeddings_batch(all_user_skills)
        
        # Process missing skills
        for skill, importance in target_skills.items():
            if skill not in matched_target_skills:
                # Check if any user skill is somewhat similar but below threshold
                best_partial_match = None
                best_partial_score = 0
                
                # Get embedding for this target skill
                skill_embedding = cls.get_embedding(skill)
                
                if skill_embedding is not None:
                    # If we have a valid embedding for the target skill,
                    # compare with all user skill embeddings
                    for user_skill, user_embedding in user_embeddings.items():
                        if user_embedding is not None:
                            try:
                                from sentence_transformers import util
                                score = float(util.cos_sim(skill_embedding, user_embedding)[0][0])
                                if score > best_partial_score:
                                    best_partial_match = user_skill
                                    best_partial_score = score
                            except Exception:
                                # Fallback to individual matching
                                _, score = cls.match_skill(
                                    user_skill, [skill], 
                                    cls._matching_config["partial_match_threshold"]
                                )
                                if score > best_partial_score:
                                    best_partial_match = user_skill
                                    best_partial_score = score
                else:
                    # Fallback to individual processing
                    for user_skill in user_skills:
                        _, score = cls.match_skill(
                            user_skill, [skill], 
                            cls._matching_config["partial_match_threshold"]
                        )
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
        
    @classmethod
    def translate_skill(cls, skill: str, source_lang: str, target_lang: str) -> str:
        """
        Translate a skill name from source language to target language.
        Uses multilingual model capabilities if available.
        
        Args:
            skill: Skill name to translate
            source_lang: Source language code (e.g., 'en', 'es', 'fr')
            target_lang: Target language code
            
        Returns:
            Translated skill name or original if translation not possible
        """
        # Check if we're using a multilingual model
        multilingual = cls._model_config.model_version == "multilingual"
        
        if not multilingual:
            logger.warning("Skill translation requires a multilingual model")
            return skill
            
        try:
            # Try to import translation services
            try:
                # First try using external translation libraries if available
                import googletrans
                translator = googletrans.Translator()
                result = translator.translate(skill, src=source_lang, dest=target_lang).text
                return result
            except ImportError:
                # Try using sentence-transformers cross-lingual capabilities
                # For this to work, we need language-prefixed examples
                source_prefix = f"[{source_lang}] "
                target_prefix = f"[{target_lang}] "
                
                # Lookup table of common skills in different languages
                # This helps guide the translation by providing examples
                TRANSLATION_EXAMPLES = {
                    "en-es": [
                        ("programming", "programación"),
                        ("machine learning", "aprendizaje automático"),
                        ("data analysis", "análisis de datos"),
                        ("project management", "gestión de proyectos")
                    ],
                    "en-fr": [
                        ("programming", "programmation"),
                        ("machine learning", "apprentissage automatique"),
                        ("data analysis", "analyse de données"),
                        ("project management", "gestion de projet")
                    ],
                    "en-de": [
                        ("programming", "Programmierung"),
                        ("machine learning", "maschinelles Lernen"),
                        ("data analysis", "Datenanalyse"),
                        ("project management", "Projektmanagement")
                    ]
                }
                
                # Get language pair key
                lang_pair = f"{source_lang}-{target_lang}"
                reverse_pair = f"{target_lang}-{source_lang}"
                
                # Get examples for this language pair
                examples = TRANSLATION_EXAMPLES.get(lang_pair, [])
                if not examples:
                    # Try reverse pair if direct pair not found
                    examples = [(b, a) for a, b in TRANSLATION_EXAMPLES.get(reverse_pair, [])]
                
                if not examples:
                    logger.warning(f"No translation examples for {lang_pair}")
                    return skill
                    
                # Create example texts with language prefixes
                example_sources = [source_prefix + src for src, _ in examples]
                example_targets = [target_prefix + tgt for _, tgt in examples]
                
                # Add our query
                query = source_prefix + skill
                all_texts = example_sources + example_targets + [query]
                
                # Get embeddings
                embeddings = cls.get_embeddings_batch(all_texts)
                
                if not all(embeddings.get(text) is not None for text in all_texts):
                    logger.warning("Could not get embeddings for translation")
                    return skill
                    
                # Find closest target language example
                query_embedding = embeddings[query]
                
                best_score = -1
                best_match = None
                
                # For each source-target example pair
                for i in range(len(examples)):
                    source_text = example_sources[i]
                    target_text = example_targets[i]
                    
                    source_embedding = embeddings[source_text]
                    target_embedding = embeddings[target_text]
                    
                    # Calculate similarity between query and source example
                    from sentence_transformers import util
                    source_sim = float(util.cos_sim(query_embedding, source_embedding)[0][0])
                    
                    if source_sim > best_score:
                        best_score = source_sim
                        best_match = examples[i]
                
                if best_match is None:
                    return skill
                    
                # Use the best matching example's target translation as a basis
                source_example, target_example = best_match
                
                # Simple replacement-based translation
                # This is a very basic approach - in a real system you would use
                # a proper translation service or more sophisticated methods
                words_to_replace = set(source_example.lower().split()) & set(skill.lower().split())
                
                result = skill
                for word in words_to_replace:
                    if word in source_example.lower() and word in skill.lower():
                        idx = source_example.lower().find(word)
                        if idx >= 0:
                            end_idx = idx + len(word)
                            replacement = target_example[idx:end_idx]
                            result = result.replace(word, replacement)
                
                return result
                
        except Exception as e:
            logger.error(f"Error during skill translation: {e}")
            return skill
    
    @classmethod
    def detect_skill_language(cls, skill: str) -> str:
        """
        Detect the language of a skill name.
        
        Args:
            skill: Skill name to detect language for
            
        Returns:
            Language code (e.g., 'en', 'es', 'fr')
        """
        try:
            # Try using langdetect if available
            from langdetect import detect
            return detect(skill)
        except ImportError:
            # Fall back to simple heuristics
            # These are very basic rules - a real implementation would use a proper
            # language detection library or API
            
            # Check for characters common in specific languages
            if any(c in 'áéíóúñ¿¡' for c in skill):
                return 'es'  # Spanish
            elif any(c in 'àâçéèêëîïôùûüÿ' for c in skill):
                return 'fr'  # French
            elif any(c in 'äöüß' for c in skill):
                return 'de'  # German
            else:
                return 'en'  # Default to English 