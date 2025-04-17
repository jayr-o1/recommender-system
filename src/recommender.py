import joblib
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class CareerRecommender:
    """
    Career path recommender that matches user skills to fields and specializations.
    Uses machine learning models to make predictions.
    """
    
    def __init__(self, data_path: str = "data", model_path: str = "model"):
        """
        Initialize the recommender with data paths
        
        Args:
            data_path: Path to data directory containing field and specialization data
            model_path: Path to directory containing trained models
        """
        self.data_path = data_path
        self.model_path = model_path
        self.fields = {}
        self.specializations = {}
        self.skill_weights = {}
        self.models_loaded = False
        self._model_cache = {}  # Cache for loaded models
        
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
        
        for i, skill in enumerate(self.feature_names):
            # Find the skill in user skills (case insensitive)
            for user_skill, proficiency in skills.items():
                if user_skill.lower() == skill.lower():
                    skill_vector[i] = proficiency / 100.0  # Normalize to 0-1
                    break
        
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
        """Count matching skills for a field"""
        if field_name not in self.fields:
            return 0
            
        field_skills = self.fields[field_name].get("core_skills", {})
        
        # Count matches
        matches = 0
        for skill in field_skills:
            if any(user_skill.lower() == skill.lower() for user_skill in skills):
                matches += 1
                
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
        """Get matched and missing skills for a specialization"""
        if spec_name not in self.specializations:
            return [], []
            
        spec_data = self.specializations[spec_name]
        spec_skills = spec_data.get("core_skills", {})
        
        matched_skill_details = []
        missing_skills = []
        
        for skill, weight in spec_skills.items():
            # Check if skill is in user skills (case insensitive)
            matched = False
            for user_skill, proficiency in skills.items():
                if user_skill.lower() == skill.lower():
                    matched_skill_details.append({
                        "skill": skill,
                        "proficiency": proficiency,
                        "weight": weight
                    })
                    matched = True
                    break
                    
            if not matched:
                missing_skills.append({
                    "skill": skill,
                    "weight": weight
                })
                
        # Sort by weight
        matched_skill_details.sort(key=lambda x: x["weight"], reverse=True)
        missing_skills.sort(key=lambda x: x["weight"], reverse=True)
        
        return matched_skill_details, missing_skills
    
    def full_recommendation(self, skills: Dict[str, int], top_fields: int = 1, 
                           top_specs: int = 3) -> Dict[str, Any]:
        """
        Perform a full recommendation including fields and specializations
        
        Args:
            skills: Dictionary of skills and proficiency (1-100)
            top_fields: Number of top fields to return
            top_specs: Number of top specializations to return per field
            
        Returns:
            Dictionary with top fields and specializations
        """
        # Get top fields
        top_field_results = self.recommend_field(skills, top_n=top_fields)
        
        results = {
            "top_fields": top_field_results,
            "specializations": []
        }
        
        # Get specializations for top field
        if top_field_results:
            top_field = top_field_results[0]["field"]
            specs = self.recommend_specializations(skills, field=top_field, top_n=top_specs)
            results["specializations"] = specs
            
            # Also add specializations across all fields if we have fewer than requested
            if len(specs) < top_specs:
                all_specs = self.recommend_specializations(skills, field=None, top_n=top_specs)
                
                # Filter out duplicates
                spec_names = [s["specialization"] for s in specs]
                additional_specs = [s for s in all_specs 
                                   if s["specialization"] not in spec_names]
                
                results["specializations"].extend(additional_specs[:top_specs-len(specs)])
        
        return results 