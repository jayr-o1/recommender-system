import joblib
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from fuzzywuzzy import fuzz, process
from utils.semantic_matcher import SemanticMatcher

class CareerRecommender:
    """
    Career path recommender that matches user skills to fields and specializations.
    Uses machine learning models to make predictions.
    """
    
    def __init__(self, data_path: str = "data", model_path: str = "model", fuzzy_threshold: int = 70, use_semantic: bool = True):
        """
        Initialize the recommender with data paths
        
        Args:
            data_path: Path to data directory containing field and specialization data
            model_path: Path to directory containing trained models
            fuzzy_threshold: Threshold score (0-100) for fuzzy matching, higher is stricter
            use_semantic: Whether to use semantic matching for skills
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
            matched, score = SemanticMatcher.match_skill(
                user_skill, 
                [target_skill], 
                threshold=self.fuzzy_threshold / 100, 
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
            # Use semantic matching for better accuracy
            for i, feature_skill in enumerate(self.feature_names):
                best_match = None
                best_score = 0
                best_proficiency = 0
                
                for user_skill, proficiency in skills.items():
                    matched, score = SemanticMatcher.match_skill(
                        user_skill, 
                        [feature_skill], 
                        threshold=self.fuzzy_threshold / 100
                    )
                    if matched and score > best_score:
                        best_match = user_skill
                        best_score = score
                        best_proficiency = proficiency
                
                if best_match:
                    skill_vector[i] = best_proficiency / 100.0  # Normalize to 0-1
        else:
            # Original implementation
            for i, skill in enumerate(self.feature_names):
                # Find the skill in user skills using improved fuzzy matching
                best_match = None
                best_score = 0
                
                for user_skill, proficiency in skills.items():
                    # Try improved matching
                    is_match, score = self._match_skill_improved(user_skill, skill)
                    if is_match and score > best_score:
                        best_match = user_skill
                        best_score = score
                
                # If we found a match, use its proficiency
                if best_match:
                    skill_vector[i] = skills[best_match] / 100.0  # Normalize to 0-1
        
        return skill_vector.reshape(1, -1)
    
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
            X = self.prepare_input(skills)
            probas = self.field_clf.predict_proba(X)[0]
            
            recommendations = []
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
            
            # Sort by confidence score
            recommendations.sort(key=lambda x: x["confidence"], reverse=True)
            return recommendations[:top_n]
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
            X = self.prepare_input(skills)
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
                    matched_skill_details, missing_skills = self._get_skill_details(skills, spec_name)
                    
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
            return recommendations[:top_n]
        except Exception as e:
            raise RuntimeError(f"Error recommending specializations: {e}")
    
    def _get_skill_details(self, skills: Dict[str, int], spec_name: str) -> tuple:
        """
        Get detailed skill matching information for a specialization
        
        Args:
            skills: Dictionary of user skills and proficiency
            spec_name: Name of specialization to check
            
        Returns:
            Tuple of (matched_skill_details, missing_skills, match_score)
        """
        if spec_name not in self.specializations:
            raise ValueError(f"Specialization '{spec_name}' not found")
            
        spec_data = self.specializations[spec_name]
        if "core_skills" not in spec_data:
            return [], [], 0.0
            
        matched_skill_details = []
        missing_skills = []
        
        for skill in spec_data["core_skills"]:
            # Get the weight/importance of this skill (default to 70)
            weight = self.skill_weights.get(skill, 70)
            
            # Find the best matching user skill
            best_match = None
            best_score = 0
            best_proficiency = 0
            
            if self.use_semantic:
                # Use semantic matching for better accuracy
                for user_skill, proficiency in skills.items():
                    matched, score = SemanticMatcher.match_skill(
                        user_skill, 
                        [skill], 
                        threshold=self.fuzzy_threshold / 100
                    )
                    score_int = int(score * 100)
                    if matched and score_int > best_score:
                        best_match = user_skill
                        best_score = score_int
                        best_proficiency = proficiency
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
                # For specialized skills, increase their impact on confidence
                skill_importance = 1.0
                
                # Check if this is a specialized skill using the semantic matcher's domain knowledge
                domain_bonuses = SemanticMatcher.get_domain_bonus(skill) if self.use_semantic else {}
                if domain_bonuses:
                    skill_importance = 1.5  # 50% more weight for specialized skills
                else:
                    # Fallback to the original specialized terms check
                    specialized_terms = {"laboratory", "techniques", "toxicology", "analytical", 
                                     "chemistry", "chemical", "organic", "synthesis", "spectroscopy",
                                     "biochemistry", "instrumentation"}
                    for term in specialized_terms:
                        if term in skill.lower():
                            skill_importance = 1.5  # 50% more weight for specialized skills
                            break
                
                matched_skill_details.append({
                    "skill": skill,
                    "proficiency": best_proficiency,
                    "weight": weight,
                    "match_score": best_score,
                    "matched_to": best_match,
                    "importance": skill_importance  # Track importance for confidence calculation
                })
                matched = True
            
            if not matched:
                missing_skills.append({
                    "skill": skill,
                    "weight": weight
                })
                
        return matched_skill_details, missing_skills, self._calculate_match_score(matched_skill_details)
    
    def _calculate_match_score(self, matched_skill_details: List[Dict[str, Any]]) -> float:
        """
        Calculate an overall match score based on matched skills
        
        Args:
            matched_skill_details: List of matched skill details
            
        Returns:
            Match score as a float between 0 and 1
        """
        if not matched_skill_details:
            return 0.0
            
        total_score = 0.0
        total_weight = 0.0
        
        for skill in matched_skill_details:
            # Consider match quality, proficiency, and importance
            normalized_match_score = skill["match_score"] / 100
            normalized_proficiency = skill["proficiency"] / 100
            
            # Ensure weight is a numeric value
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
        Calculate confidence score for specialization match
        
        Args:
            spec_name: Name of specialization
            matched_skill_details: List of matched skill details
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not matched_skill_details:
            return 0.0
            
        # Use the new match score calculation
        match_score = self._calculate_match_score(matched_skill_details)
        
        # Get the specialization data
        spec_data = self.specializations.get(spec_name, {})
        
        # Calculate what percentage of core skills are matched
        core_skills = spec_data.get("core_skills", [])
        if not core_skills:
            return match_score
            
        # Get matched skill count
        matched_count = len(matched_skill_details)
        total_count = len(core_skills)
        coverage_ratio = matched_count / total_count
        
        # Weight the match score by coverage
        confidence = match_score * (0.5 + 0.5 * coverage_ratio)  # Coverage impacts 50% of confidence
        
        return min(1.0, confidence)
    
    def full_recommendation(self, skills: Dict[str, int], top_fields: int = 1, 
                             top_specs: int = 3) -> Dict[str, Any]:
        """
        Get full recommendation including fields and specializations
        
        Args:
            skills: Dictionary mapping skill names to proficiency levels (1-100)
            top_fields: Number of top fields to return
            top_specs: Number of top specializations to return
            
        Returns:
            Dictionary with recommendations
        """
        # Process skills first
        processed_skills = {k: v for k, v in skills.items() if v > 0}
        
        # Try to use ML models if loaded
        if self.models_loaded:
            # Prepare input for prediction
            X = self.prepare_input(processed_skills)
            
            # Predict fields
            field_confidences = {}
            try:
                field_probs = self.field_clf.predict_proba(X)[0]
                for i, field_name in enumerate(self.le_field.classes_):
                    confidence = field_probs[i] * 100
                    if confidence > 1.0:  # Only include if confidence is meaningful
                        field_confidences[field_name] = confidence
            except Exception as e:
                print(f"Error predicting fields: {e}")
                # Fall back to rule-based approach for fields
                for field_name in self.fields:
                    matches = self._get_matching_skills_for_field(processed_skills, field_name)
                    if matches > 0:
                        total_skills = len(self.fields[field_name].get("core_skills", []))
                        confidence = min((matches / max(total_skills, 1)) * 100, 100)
                        field_confidences[field_name] = confidence
                
            # Get top fields
            top_field_names = sorted(field_confidences.keys(), 
                                   key=lambda x: field_confidences[x], 
                                   reverse=True)[:top_fields]
            
            # Get specializations for top fields
            specialization_results = []
            for field_name in top_field_names:
                # Filter specializations by field
                field_specs = {spec_name: spec_data for spec_name, spec_data in self.specializations.items()
                             if spec_data.get("field") == field_name}
                
                for spec_name, spec_data in field_specs.items():
                    # Get matched and missing skills - now returns 3 values
                    matched_skill_details, missing_skills, match_score = self._get_skill_details(processed_skills, spec_name)
                    
                    # Calculate confidence using new method with specialized skill weighting
                    confidence = self._calculate_specialization_confidence(spec_name, matched_skill_details)
                    
                    if confidence > 0:
                        specialization_results.append({
                            "specialization": spec_name,
                            "field": field_name,
                            "confidence": confidence,
                            "matched_skill_details": matched_skill_details,
                            "missing_skills": missing_skills,
                            "description": spec_data.get("description", "")
                        })
            
            # Sort specializations by confidence
            specialization_results.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Build result dictionary
            result = {
                "success": True,
                "fields": [
                    {
                        "field": field_name,
                        "confidence": round(field_confidences[field_name], 2),
                        "description": self.fields[field_name].get("description", "")
                    }
                    for field_name in top_field_names
                ],
                "specializations": specialization_results[:top_specs]
            }
            
            return result
        else:
            # Fall back to rule-based approach
            raise ValueError("Models not loaded. Use rule-based recommendation instead.") 