import joblib
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from fuzzywuzzy import fuzz, process
from utils.semantic_matcher import SemanticMatcher, ModelConfig
import logging

class CareerRecommender:
    """
    Career path recommender that matches user skills to fields and specializations.
    Uses machine learning models to make predictions.
    """
    
    def __init__(self, data_path: str = "data", model_path: str = "model", fuzzy_threshold: int = 70, use_semantic: bool = True, semantic_model: str = "small"):
        """
        Initialize the recommender with data paths
        
        Args:
            data_path: Path to data directory containing field and specialization data
            model_path: Path to directory containing trained models
            fuzzy_threshold: Threshold score (0-100) for fuzzy matching, higher is stricter
            use_semantic: Whether to use semantic matching for skills
            semantic_model: Model size to use for semantic matching (small, medium, large, multilingual)
        """
        self.data_path = data_path
        self.model_path = model_path
        self.fields = {}
        self.specializations = {}
        self.skill_weights = {}
        self.models_loaded = False
        self._model_cache = {}  # Cache for loaded models
        self.fuzzy_threshold = fuzzy_threshold
        self.use_semantic = use_semantic
        
        # Initialize semantic matcher with configuration if semantic matching is enabled
        if self.use_semantic:
            # Configure semantic matcher
            SemanticMatcher.configure_model(ModelConfig(
                model_name=semantic_model,
                warmup_on_init=True,
                enable_progress_bars=False
            ))
            
            # Set matching configuration
            SemanticMatcher.configure_matching({
                "similarity_threshold": self.fuzzy_threshold / 100,
                "partial_match_threshold": 0.4,  # Lower threshold for partial matches
            })
        
        # Load data
        self.load_data()
        
        # Try to load models
        try:
            self.load_models()
        except Exception as e:
            print(f"Warning: Could not load ML models, falling back to rule-based recommendations: {e}")
    
    def load_data(self) -> None:
        """Load all field, specialization, and skill weight data"""
        try:
            # Load fields
            field_path = os.path.join(self.data_path, "fields.json")
            if os.path.exists(field_path):
                with open(field_path, "r") as f:
                    self.fields = json.load(f)
            else:
                raise FileNotFoundError(f"Field data file not found: {field_path}")
                
            # Load specializations
            spec_path = os.path.join(self.data_path, "specializations.json")
            if os.path.exists(spec_path):
                with open(spec_path, "r") as f:
                    self.specializations = json.load(f)
            else:
                raise FileNotFoundError(f"Specialization data file not found: {spec_path}")
                
            # Load skill weights
            weights_path = os.path.join(self.data_path, "skill_weights.json")
            if os.path.exists(weights_path):
                with open(weights_path, "r") as f:
                    self.skill_weights = json.load(f)
            else:
                raise FileNotFoundError(f"Skill weights file not found: {weights_path}")
                
            # Validate data consistency
            self._validate_data_consistency()
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in data files: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")
    
    def _validate_data_consistency(self) -> None:
        """
        Validate consistency between fields and specializations
        
        Raises:
            ValueError: If inconsistencies are found
        """
        # Check that all specialization fields exist in fields
        invalid_specs = []
        for spec_name, spec_data in self.specializations.items():
            if "field" not in spec_data:
                invalid_specs.append(f"{spec_name} (missing field)")
            elif spec_data["field"] not in self.fields:
                invalid_specs.append(f"{spec_name} (references non-existent field '{spec_data['field']}')")
                
        if invalid_specs:
            raise ValueError(f"Data inconsistency found in specializations: {', '.join(invalid_specs)}")
            
        # Check that all skills in specializations exist in skill_weights
        missing_skills = []
        for spec_name, spec_data in self.specializations.items():
            if "core_skills" in spec_data:
                for skill in spec_data["core_skills"]:
                    if skill not in self.skill_weights:
                        missing_skills.append(f"{skill} (used in {spec_name})")
                        
        if missing_skills:
            print(f"Warning: Some skills in specializations are not in skill_weights: {', '.join(missing_skills)}")
            
        # Validate that all confidence-related fields exist
        for field, data in self.fields.items():
            if "core_skills" not in data:
                print(f"Warning: Field '{field}' missing core_skills definition")
                
        for spec, data in self.specializations.items():
            if "core_skills" not in data:
                print(f"Warning: Specialization '{spec}' missing core_skills definition")
            elif not isinstance(data["core_skills"], dict):
                print(f"Warning: Specialization '{spec}' has invalid core_skills format")
    
    def load_models(self) -> None:
        """Load trained models and encoders"""
        try:
            # Check if models are already cached
            if self._model_cache:
                self.field_clf = self._model_cache.get("field_clf")
                self.spec_clf = self._model_cache.get("spec_clf")
                self.le_field = self._model_cache.get("le_field")
                self.le_spec = self._model_cache.get("le_spec")
                self.feature_names = self._model_cache.get("feature_names")
                self.models_loaded = True
                return
                
            # Load models from disk
            required_files = ["field_clf.joblib", "spec_clf.joblib", "le_field.joblib", 
                             "le_spec.joblib", "feature_names.joblib"]
            
            # Check if all required files exist
            missing_files = []
            for filename in required_files:
                if not os.path.exists(os.path.join(self.model_path, filename)):
                    missing_files.append(filename)
            
            if missing_files:
                raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")
            
            self.field_clf = joblib.load(os.path.join(self.model_path, "field_clf.joblib"))
            self.spec_clf = joblib.load(os.path.join(self.model_path, "spec_clf.joblib"))
            self.le_field = joblib.load(os.path.join(self.model_path, "le_field.joblib"))
            self.le_spec = joblib.load(os.path.join(self.model_path, "le_spec.joblib"))
            self.feature_names = joblib.load(os.path.join(self.model_path, "feature_names.joblib"))
            
            # Cache models for future use
            self._model_cache = {
                "field_clf": self.field_clf,
                "spec_clf": self.spec_clf,
                "le_field": self.le_field,
                "le_spec": self.le_spec,
                "feature_names": self.feature_names
            }
            
            # Validate model consistency with data
            self._validate_model_consistency()
            
            self.models_loaded = True
        except (FileNotFoundError, IOError) as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False
            raise
        except Exception as e:
            print(f"Unexpected error loading models: {e}")
            self.models_loaded = False
            raise
    
    def _validate_model_consistency(self) -> None:
        """
        Validate consistency between trained models and current data
        
        Warns if inconsistencies are found
        """
        # Check that all encoded fields exist in current data
        for field in self.le_field.classes_:
            if field not in self.fields:
                print(f"Warning: Field '{field}' in model does not exist in current data")
                
        # Check that all encoded specializations exist in current data
        for spec in self.le_spec.classes_:
            if spec not in self.specializations:
                print(f"Warning: Specialization '{spec}' in model does not exist in current data")
                
        # Check that all feature names exist in current skill weights
        for feature in self.feature_names:
            if feature not in self.skill_weights:
                print(f"Warning: Feature '{feature}' in model does not exist in current skill_weights")
    
    def _match_skill_improved(self, user_skill: str, target_skill: str) -> Tuple[bool, int]:
        """
        Improved skill matching that considers partial token matches for specialized terms
        
        Args:
            user_skill: User's skill
            target_skill: Target skill to match against
            
        Returns:
            Tuple of (is_match, score)
        """
        # Use semantic matching if enabled
        if self.use_semantic:
            # The threshold is now configured globally in __init__
            matched, score = SemanticMatcher.match_skill(
                user_skill, 
                [target_skill],
                use_semantic=True
            )
            return matched is not None, int(score * 100)
            
        # Fall back to the original implementation
        # Try exact match first (case-insensitive)
        if user_skill.lower() == target_skill.lower():
            return True, 100
        
        # Try token matching for multi-word skills
        user_tokens = set(user_skill.lower().split())
        target_tokens = set(target_skill.lower().split())
        
        # If there's significant token overlap, increase the score
        common_tokens = user_tokens.intersection(target_tokens)
        if common_tokens and len(common_tokens) >= min(len(user_tokens), len(target_tokens)) / 2:
            # Calculate token similarity ratio
            token_ratio = len(common_tokens) / max(len(user_tokens), len(target_tokens))
            token_score = int(token_ratio * 100)
            
            # Give bonus to technical/specialized terms
            specialized_terms = {"laboratory", "techniques", "toxicology", "analytical", 
                              "chemistry", "chemical", "organic", "synthesis", "spectroscopy"}
            if any(term in common_tokens for term in specialized_terms):
                token_score += 10
                
            # Use the token score if it's higher than the fuzzy ratio
            ratio_score = fuzz.ratio(user_skill.lower(), target_skill.lower())
            score = max(token_score, ratio_score)
            
            return score >= self.fuzzy_threshold, score
        
        # Fall back to regular fuzzy matching
        score = fuzz.ratio(user_skill.lower(), target_skill.lower())
        return score >= self.fuzzy_threshold, score
    
    def prepare_input(self, skills: Dict[str, int]) -> np.ndarray:
        """
        Convert user skills to feature vector based on trained model's feature names
        
        Args:
            skills: Dictionary mapping skill names to proficiency levels (1-100)
            
        Returns:
            Feature vector for model prediction
        """
        if not self.feature_names:
            raise ValueError("Model feature names not loaded. Please load models first.")
            
        skill_vector = np.zeros(len(self.feature_names))
        
        if self.use_semantic:
            # Use batch processing for better performance
            all_skills = list(skills.keys())
            feature_skills = self.feature_names
            
            # Use weighted skill match which now uses batch processing internally
            matches = SemanticMatcher.weighted_skill_match(
                skills,
                {feature: 100 for feature in feature_skills}  # All features with equal importance
            )
            
            # Process matches to fill the skill vector
            matched_features = {}
            for match in matches:
                target_skill = match["target_skill"]
                user_skill = match["user_skill"]
                proficiency = match["proficiency"]
                
                # Find the index of this feature in feature_names
                if target_skill in self.feature_names:
                    idx = self.feature_names.index(target_skill)
                    skill_vector[idx] = proficiency / 100.0  # Normalize to 0-1
                    matched_features[target_skill] = True
            
            # For any features without matches, try individual matching
            for i, feature_skill in enumerate(self.feature_names):
                if feature_skill not in matched_features:
                    best_match = None
                    best_score = 0
                    best_proficiency = 0
                    
                    for user_skill, proficiency in skills.items():
                        matched, score = SemanticMatcher.match_skill(
                            user_skill, 
                            [feature_skill]
                        )
                        if matched and score > best_score:
                            best_match = user_skill
                            best_score = score
                            best_proficiency = proficiency
                    
                    if best_match:
                        skill_vector[i] = best_proficiency / 100.0  # Normalize to 0-1
        else:
            # Original implementation without semantic matching
            for i, feature_skill in enumerate(self.feature_names):
                best_match = None
                best_score = 0
                best_proficiency = 0
                
                for user_skill, proficiency in skills.items():
                    is_match, score = self._match_skill_improved(user_skill, feature_skill)
                    if is_match and score > best_score:
                        best_match = user_skill
                        best_score = score
                        best_proficiency = proficiency
                
                if best_match:
                    skill_vector[i] = best_proficiency / 100.0  # Normalize to 0-1
                    
        return skill_vector
    
    def recommend_field(self, skills: Dict[str, int], top_n: int = 1) -> List[Dict[str, Any]]:
        """
        Recommend the most suitable field based on user skills
        
        Args:
            skills: Dictionary of skills and proficiency (1-100)
            top_n: Number of top fields to return
            
        Returns:
            List of recommended fields with confidence scores
        """
        if not skills:
            raise ValueError("No skills provided. Please provide at least one skill.")
            
        if not self.models_loaded:
            raise ValueError("Models not loaded. Please train the models first by running src/train.py.")
            
        try:
            # Convert skills to feature vector
            X = self.prepare_input(skills)
            
            # Log input shape before reshape
            logger = logging.getLogger(__name__)
            logger.debug(f"Input shape before reshape: {np.array(X).shape}")
            
            # Ensure proper shape: 2D array with a single sample
            X = np.array(X).reshape(1, -1)
            
            # Log input shape after reshape
            logger.debug(f"Input shape after reshape: {X.shape}")
            
            # Make prediction
            probas = self.field_clf.predict_proba(X)[0]
            
            recommendations = []
            for field_idx in range(len(self.le_field.classes_)):
                field_name = self.le_field.classes_[field_idx]
                confidence = round(float(probas[field_idx]) * 100, 2)
                
                # Only add fields with confidence > 5%
                if confidence > 5:
                    # Validate field domain alignment
                    if self._validate_field_domain(skills, field_name):
                        recommendations.append({
                            "field": field_name,
                            "confidence": confidence,
                            "matched_skills": self._get_matching_skills_for_field(skills, field_name)
                        })
                    else:
                        # Log rejection due to domain mismatch
                        logger.debug(f"Field {field_name} rejected due to domain mismatch with skills")
                        
                        # For Cybersecurity specifically, reduce confidence significantly if no tech skills
                        if field_name.lower() == "cybersecurity":
                            # Add with greatly reduced confidence if it would have been recommended
                            if confidence > 30:
                                recommendations.append({
                                    "field": field_name,
                                    "confidence": confidence * 0.3,  # Reduce confidence by 70%
                                    "matched_skills": self._get_matching_skills_for_field(skills, field_name)
                                })
            
            # Add a check for the "minimal skills" case - if there's only one or two skills
            if len(skills) <= 2:
                # Adjust confidence scores - penalize cybersecurity for minimal skills
                for rec in recommendations:
                    if rec["field"].lower() == "cybersecurity":
                        rec["confidence"] *= 0.5  # Reduce cybersecurity confidence by 50% for minimal skills
            
            # Sort by confidence score
            recommendations.sort(key=lambda x: x["confidence"], reverse=True)
            
            # If no recommendations passed validation, return the top fields without validation
            if not recommendations:
                logger.warning("No recommendations passed domain validation, falling back to raw predictions.")
                for field_idx in range(len(self.le_field.classes_)):
                    field_name = self.le_field.classes_[field_idx]
                    confidence = round(float(probas[field_idx]) * 100, 2)
                    
                    # Only add fields with confidence > 5%
                    if confidence > 5:
                        recommendations.append({
                            "field": field_name,
                            "confidence": confidence,
                            "matched_skills": self._get_matching_skills_for_field(skills, field_name)
                        })
                
                # Sort by confidence
                recommendations.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Apply post-processing filters to recommendations
            filtered_recommendations = self._filter_recommendations(recommendations, skills)
            
            return filtered_recommendations[:top_n]
        except Exception as e:
            raise RuntimeError(f"Error recommending fields: {e}")
    
    def _get_matching_skills_for_field(self, skills: Dict[str, int], field_name: str) -> int:
        """Count matching skills for a field using improved fuzzy matching"""
        if field_name not in self.fields:
            return 0
            
        field_skills = self.fields[field_name].get("core_skills", {})
        
        # Count matches
        matches = 0
        for skill in field_skills:
            match_found = False
            for user_skill in skills:
                is_match, _ = self._match_skill_improved(user_skill, skill)
                if is_match:
                    matches += 1
                    match_found = True
                    break
                
        return matches
    
    def _filter_recommendations(self, recommendations: List[Dict[str, Any]], skills: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Apply post-processing filters to recommendations to improve accuracy
        
        Args:
            recommendations: List of recommendation dictionaries
            skills: Dictionary of user skills
            
        Returns:
            Filtered list of recommendations
        """
        if not recommendations:
            return recommendations
            
        # Check for legal-specific skills
        legal_terms = {"legal", "law", "litigation", "corporate", "compliance", "regulatory", 
                     "contracts", "attorney", "lawyer", "legal research", "case law"}
        has_legal_skills = any(any(term in skill.lower() for term in legal_terms) 
                              for skill in skills.keys())
        
        # Count skills with legal terms
        legal_skill_count = sum(1 for skill in skills.keys() 
                             if any(term in skill.lower() for term in legal_terms))
        
        # Check for irrelevant skills (skills that don't match any professional field)
        irrelevant_terms = {"cooking", "gardening", "hobbies", "gaming", "sports"}
        has_irrelevant_skills = any(any(term in skill.lower() for term in irrelevant_terms) 
                                   for skill in skills.keys())
        
        # Minimal skills case - only 1-2 skills provided
        has_minimal_skills = len(skills) <= 2
        
        # Apply hotfix for legal vs. cybersecurity confusion
        if has_legal_skills:
            # Check if cybersecurity is recommended
            has_cyber_rec = any(r["field"] == "Cybersecurity" for r in recommendations)
            
            # If more than 50% of skills are legal-related, remove cybersecurity
            if legal_skill_count / len(skills) > 0.5 and has_cyber_rec:
                recommendations = [r for r in recommendations if r["field"] != "Cybersecurity"]
                logging.debug("Removed Cybersecurity recommendation due to predominant legal skills")
        
        # Irrelevant skills case - boost confidence for general fields, reduce for specialized
        if has_irrelevant_skills:
            for rec in recommendations:
                if rec["field"] in ["Cybersecurity", "Data Science", "Software Engineering"]:
                    rec["confidence"] *= 0.5  # Reduce technical field confidence for irrelevant skills
        
        # For minimal skills case with cybersecurity
        if has_minimal_skills and any(r["field"] == "Cybersecurity" for r in recommendations):
            for rec in recommendations:
                if rec["field"] == "Cybersecurity":
                    rec["confidence"] *= 0.4  # Significantly reduce confidence for minimal skills
        
        # Re-sort by confidence
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        return recommendations
    
    def recommend_specializations(self, skills: Dict[str, int], field: Optional[str] = None, 
                                 top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend specializations based on user skills
        
        Args:
            skills: Dictionary of skills and proficiency (1-100)
            field: Optional field to filter specializations
            top_n: Number of top specializations to return
            
        Returns:
            List of recommended specializations with confidence scores
        """
        if not skills:
            raise ValueError("No skills provided. Please provide at least one skill.")
            
        if field and field not in self.fields:
            raise ValueError(f"Unknown field: {field}. Please provide a valid field name.")
            
        if not self.models_loaded:
            raise ValueError("Models not loaded. Please train the models first by running src/train.py.")
            
        try:
            # Convert skills to feature vector
            X = self.prepare_input(skills)
            
            # Log input shape before reshape
            logger = logging.getLogger(__name__)
            logger.debug(f"Input shape before reshape: {np.array(X).shape}")
            
            # Ensure proper shape: 2D array with a single sample
            X = np.array(X).reshape(1, -1)
            
            # Log input shape after reshape
            logger.debug(f"Input shape after reshape: {X.shape}")
            
            # Make prediction
            probas = self.spec_clf.predict_proba(X)[0]
            
            recommendations = []
            for spec_idx in range(len(self.le_spec.classes_)):
                spec_name = self.le_spec.classes_[spec_idx]
                spec_data = self.specializations.get(spec_name, {})
                
                # Skip if not in the requested field
                if field and spec_data.get("field") != field:
                    continue
                    
                confidence = round(float(probas[spec_idx]) * 100, 2)
                
                # Only add specializations with confidence > 5%
                if confidence > 5:
                    # Get matched and missing skills
                    matched_skill_details, missing_skills, match_score, confidence = self._get_skill_details(skills, spec_name)
                    
                    recommendations.append({
                        "specialization": spec_name,
                        "field": spec_data.get("field", "Unknown"),
                        "confidence": confidence,
                        "matched_skills": matched_skill_details,
                        "missing_skills": missing_skills,
                        "total_skills_required": len(spec_data.get("core_skills", {})),
                        "skills_matched": len(matched_skill_details)
                    })
            
            # Sort by confidence score
            recommendations.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Apply post-processing filters
            filtered_recommendations = self._filter_recommendations(recommendations, skills)
            
            return filtered_recommendations[:top_n]
        except Exception as e:
            raise RuntimeError(f"Error recommending specializations: {e}")
    
    def _get_skill_details(self, skills: Dict[str, int], spec_name: str) -> tuple:
        """
        Get detailed information about skill matches for a specialization
        
        Args:
            skills: Dictionary of user skills and proficiency
            spec_name: Name of the specialization
            
        Returns:
            Tuple of (matched_skill_details, missing_skill_details, match_score, confidence)
        """
        if spec_name not in self.specializations:
            return [], [], 0.0, 0.0
            
        spec_data = self.specializations[spec_name]
        required_skills = spec_data.get("core_skills", {})
        
        if not required_skills:
            return [], [], 0.0, 0.0
            
        matched_skill_details = []
        missing_skill_details = []
        
        # For each required skill
        for skill, importance in required_skills.items():
            # Get the weight/importance of this skill (default to 70)
            weight = self.skill_weights.get(skill, 70)
            
            # Find the best matching user skill
            best_match = None
            best_score = 0
            best_proficiency = 0
            
            if self.use_semantic:
                # Use semantic matching with batch processing
                # This is a more efficient approach using the enhanced SemanticMatcher
                
                # For a single skill, we can use semantic_matcher directly
                matched, score = SemanticMatcher.match_skill(
                    skill,
                    list(skills.keys())
                )
                
                if matched:
                    best_match = matched
                    best_score = int(score * 100)
                    best_proficiency = skills[matched]
            else:
                # Original implementation
                for user_skill, proficiency in skills.items():
                    is_match, score = self._match_skill_improved(user_skill, skill)
                    if is_match and score > best_score:
                        best_match = user_skill
                        best_score = score
                        best_proficiency = proficiency
            
            # Record the match or missing skill
            matched = False
            if best_match:
                matched = True
                matched_skill_details.append({
                    "skill": skill,
                    "user_skill": best_match,
                    "match_score": best_score,
                    "proficiency": best_proficiency,
                    "importance": importance,
                    "weight": weight if isinstance(weight, (int, float)) else 
                             weight.get("global_weight", 70) if isinstance(weight, dict) else 70
                })
            
            if not matched:
                # Get domain information for better categorization
                domains = SemanticMatcher.get_prioritized_domains(skill) if self.use_semantic else []
                primary_domain = domains[0] if domains else "general"
                
                # For missing skills, check if any user skill is somewhat similar but below threshold
                best_partial_match = None
                best_partial_score = 0
                
                if self.use_semantic:
                    # Use the prioritize_missing_skills method which is optimized for this task
                    missing_analysis = SemanticMatcher.prioritize_missing_skills(
                        skills,
                        {skill: importance}  # Just this single skill
                    )
                    
                    if missing_analysis:
                        missing_info = missing_analysis[0]
                        best_partial_match = missing_info["closest_user_skill"]
                        best_partial_score = int(missing_info["similarity"] * 100)
                        priority_score = missing_info["priority_score"]
                    else:
                        priority_score = importance / 100
                else:
                    # Original implementation
                    for user_skill, proficiency in skills.items():
                        is_match, score = self._match_skill_improved(user_skill, skill)
                        if score > best_partial_score:
                            best_partial_match = user_skill
                            best_partial_score = score
                            
                    # Calculate priority based on importance and partial match score
                    # Higher importance and lower partial match means higher priority
                    priority_score = (importance / 100) * (1 - (best_partial_score / 100 * 0.5))
                
                missing_skill_details.append({
                    "skill": skill,
                    "importance": importance,
                    "weight": weight if isinstance(weight, (int, float)) else 
                             weight.get("global_weight", 70) if isinstance(weight, dict) else 70,
                    "closest_match": best_partial_match,
                    "closest_score": best_partial_score,
                    "priority": priority_score,
                    "domain": primary_domain if self.use_semantic else "general"
                })
        
        # Calculate overall match score and confidence
        match_score = self._calculate_match_score(matched_skill_details)
        confidence = self._calculate_specialization_confidence(spec_name, matched_skill_details)
        
        # Sort missing skills by priority
        missing_skill_details.sort(key=lambda x: x["priority"], reverse=True)
        
        return matched_skill_details, missing_skill_details, match_score, confidence
    
    def _calculate_match_score(self, matched_skill_details: List[Dict[str, Any]]) -> float:
        """
        Calculate an overall match score based on matched skills
        
        Args:
            matched_skill_details: List of matched skill details
        
        Returns:
            Match score between 0.0 and 1.0
        """
        if not matched_skill_details:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for skill in matched_skill_details:
            # Normalize scores between 0 and 1
            normalized_match_score = skill["match_score"] / 100
            normalized_proficiency = skill["proficiency"] / 100
            
            # Get skill weight (default to 70 if not found or invalid)
            weight = skill["weight"]
            if isinstance(weight, dict):
                # If weight is a dictionary, use a default value
                weight = 70
            elif not isinstance(weight, (int, float)):
                # If weight is not numeric, use a default value
                weight = 70
            
            importance = skill.get("importance", 1.0)
            
            # Weight the skill by its match score, proficiency, and importance
            skill_score = normalized_match_score * normalized_proficiency * importance
            total_score += skill_score * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0
    
    def _calculate_specialization_confidence(self, spec_name: str, 
                                          matched_skill_details: List[Dict[str, Any]]) -> float:
        """
        Improved confidence scoring with better normalization and weighting
        
        Args:
            spec_name: Name of specialization
            matched_skill_details: List of matched skill details
            
        Returns:
            Normalized confidence score between 0.0 and 1.0
        """
        if not matched_skill_details:
            return 0.0
        
        # Get specialization data
        spec_data = self.specializations.get(spec_name, {})
        core_skills = spec_data.get("core_skills", {})
        
        # Base case - if no required skills, return neutral confidence
        if not core_skills:
            return 0.5
        
        # Calculate three main components
        # 1. Skill Coverage (how many required skills are matched)
        matched_count = len(matched_skill_details)
        coverage_ratio = matched_count / len(core_skills)
        
        # 2. Quality of Matches (proficiency and match scores)
        total_quality = 0.0
        total_weight = 0.0
        
        for skill in matched_skill_details:
            # Normalize all components to 0-1 range
            match_quality = skill["match_score"] / 100
            proficiency = skill["proficiency"] / 100
            importance = skill.get("importance", 70) / 100
            weight = skill.get("weight", 70) / 100
            
            # Combined quality metric
            skill_score = match_quality * proficiency * importance
            total_quality += skill_score * weight
            total_weight += weight
        
        # Avoid division by zero
        if total_weight > 0:
            quality_score = total_quality / total_weight
        else:
            quality_score = 0.0
        
        # 3. Specialization Coverage (critical skills matched)
        critical_skills = [s for s, v in core_skills.items() 
                         if self.skill_weights.get(s, {}).get("global_weight", 70) > 80 
                         or v > 80]
        
        matched_critical = sum(1 for skill in matched_skill_details 
                             if skill["skill"] in critical_skills)
        
        # Calculate critical coverage with smoothing (add 1 to denominator)
        critical_coverage = matched_critical / (len(critical_skills) if critical_skills else 0.5)
        
        # Final weighted composition
        weights = {
            'coverage': 0.3,      # How many skills are matched
            'quality': 0.5,       # How well they're matched
            'critical': 0.2       # Critical skills matched
        }
        
        confidence = (
            weights['coverage'] * coverage_ratio +
            weights['quality'] * quality_score +
            weights['critical'] * critical_coverage
        )
        
        # Apply softmax-style scaling to prevent extreme values
        confidence = 1 / (1 + np.exp(-10 * (confidence - 0.5)))
        
        # Apply domain-specific adjustments
        if self._is_ui_ux_heavy(matched_skill_details):
            if spec_name == "Civil Engineer" and not self._has_technical_skills(matched_skill_details):
                confidence *= 0.5  # Reduce confidence for non-technical UI/UX matches
        
        # Ensure reasonable bounds
        return min(0.95, max(0.05, confidence))

    def _is_ui_ux_heavy(self, matched_skills: List[Dict[str, Any]]) -> bool:
        """Check if matches are predominantly UI/UX skills"""
        ui_ux_terms = {"ui", "ux", "user interface", "user experience", "design"}
        return any(any(term in skill["user_skill"].lower() for term in ui_ux_terms)
                for skill in matched_skills)

    def _has_technical_skills(self, matched_skills: List[Dict[str, Any]]) -> bool:
        """Check for technical engineering skills"""
        tech_terms = {"structural", "civil", "mechanical", "construction", "engineering"}
        return any(any(term in skill["user_skill"].lower() for term in tech_terms)
                for skill in matched_skills)
    
    def _validate_field_domain(self, skills: Dict[str, int], field: str) -> bool:
        """
        Check if skills align with the field's domain
        
        Args:
            skills: Dictionary of skills and proficiency
            field: Field name to validate against
            
        Returns:
            True if skills align with the field's domain, False otherwise
        """
        if not self.use_semantic:
            return True  # Skip domain validation if semantic matching is disabled
            
        # Define domain groupings
        legal_domains = {"legal", "law", "litigation", "corporate", "compliance", "regulatory"}
        tech_domains = {"cybersecurity", "programming", "networking", "encryption", "security", "technical"}
        business_domains = {"business", "management", "finance", "accounting", "marketing"}
        
        # Get skill domains using SemanticMatcher
        skill_domains = set()
        for skill in skills.keys():
            # Get prioritized domains for this skill
            domains = SemanticMatcher.get_prioritized_domains(skill)
            if domains:
                skill_domains.update(domains)
                
        # Field-specific validation
        if field.lower() == "cybersecurity":
            # Check if at least one cybersecurity or tech skill exists
            has_tech_skill = any(domain in tech_domains for domain in skill_domains)
            if not has_tech_skill:
                # If no tech skills, check skill names explicitly for tech terms
                has_tech_term = any(any(term in skill.lower() for term in tech_domains) 
                                 for skill in skills.keys())
                return has_tech_term
                
        elif field.lower() in {"law", "corporate law", "litigation"}:
            # Check if at least one legal skill exists
            has_legal_skill = any(domain in legal_domains for domain in skill_domains) 
            if not has_legal_skill:
                # If no legal skills, check skill names explicitly for legal terms
                has_legal_term = any(any(term in skill.lower() for term in legal_domains) 
                                  for skill in skills.keys())
                return has_legal_term
                
        # Default: no specific domain requirements
        return True
    
    def full_recommendation(self, skills: Dict[str, int], top_fields: int = 1, 
                         top_specs: int = 3) -> Dict[str, Any]:
        """
        Generate complete career recommendations including fields, specializations, and skill gaps
        
        Args:
            skills: Dictionary mapping skill names to proficiency levels (1-100)
            top_fields: Number of top fields to return
            top_specs: Number of top specializations per field to return
            
        Returns:
            Dictionary with recommendation results
        """
        # Validate input
        if not isinstance(skills, dict):
            raise ValueError("Skills must be a dictionary")
            
        # Get field recommendations
        field_recommendations = self.recommend_field(skills, top_n=top_fields)
        
        # Apply post-processing filters to field recommendations
        field_recommendations = self._filter_recommendations(field_recommendations, skills)
        
        # Get specialization recommendations for each field
        all_specialization_recommendations = []
        field_specializations = {}
        
        for field_rec in field_recommendations:
            field_name = field_rec["field"]
            spec_recommendations = self.recommend_specializations(skills, field=field_name, top_n=top_specs)
            
            field_specializations[field_name] = spec_recommendations
            all_specialization_recommendations.extend(spec_recommendations)
        
        # Sort all specializations by confidence
        all_specialization_recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Get skill development suggestions
        development_suggestions = []
        
        if self.use_semantic and all_specialization_recommendations:
            # Get the top specialization's required skills
            top_spec = all_specialization_recommendations[0]
            top_spec_name = top_spec["specialization"]
            
            if top_spec_name in self.specializations:
                required_skills = self.specializations[top_spec_name].get("core_skills", {})
                
                # Use the enhanced skill development path feature
                development_path = SemanticMatcher.suggest_skill_development_path(
                    skills,
                    required_skills,
                    max_suggestions=5
                )
                
                for suggestion in development_path:
                    development_suggestions.append({
                        "skill": suggestion["skill"],
                        "importance": suggestion["importance"],
                        "learnability": suggestion["learnability"],
                        "priority": suggestion["priority"],
                        "domain": suggestion["domain"],
                        "closest_existing_skill": suggestion["closest_existing_skill"],
                        "similarity_to_existing": suggestion["similarity_to_existing"]
                    })
        else:
            # Fallback to simpler approach
            # Get missing skills from top specializations
            for spec_rec in all_specialization_recommendations[:2]:  # Just use top 2 specializations
                for missing in spec_rec["missing_skills"][:3]:  # Just use top 3 missing skills per spec
                    # Avoid duplicates
                    if not any(s["skill"] == missing["skill"] for s in development_suggestions):
                        development_suggestions.append(missing)
            
            # Sort by priority
            development_suggestions.sort(key=lambda x: x.get("priority", 0), reverse=True)
            development_suggestions = development_suggestions[:5]  # Just top 5
        
        # Return comprehensive results
        return {
            "fields": field_recommendations,
            "specializations_by_field": field_specializations,
            "top_specializations": all_specialization_recommendations[:top_specs],
            "skill_development": development_suggestions
        } 