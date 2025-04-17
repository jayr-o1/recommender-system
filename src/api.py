from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import uvicorn
import os

from src.recommender import CareerRecommender

app = FastAPI(
    title="Career Recommender API",
    description="API for recommending career paths based on skills and proficiency",
    version="1.0.0",
)

# Initialize the recommender
recommender = CareerRecommender()

class SkillInput(BaseModel):
    """Skills input model with validation"""
    skills: Dict[str, int] = Field(
        ...,
        description="Dictionary of skills and proficiency levels (1-100)",
        example={"Python": 90, "SQL": 80, "Data Analysis": 85, "Machine Learning": 80, "Excel": 75}
    )
    
    @validator('skills')
    def validate_proficiency(cls, skills):
        """Validate that proficiency levels are between 1 and 100"""
        for skill, proficiency in skills.items():
            if not isinstance(proficiency, int):
                raise ValueError(f"Proficiency for skill '{skill}' must be an integer")
            if proficiency < 1 or proficiency > 100:
                raise ValueError(f"Proficiency for skill '{skill}' must be between 1 and 100")
        return skills


class RecommendationParams(BaseModel):
    """Parameters for recommendation"""
    top_fields: Optional[int] = Field(1, ge=1, le=5, description="Number of top fields to return")
    top_specializations: Optional[int] = Field(3, ge=1, le=10, description="Number of top specializations to return")
    fuzzy_threshold: Optional[int] = Field(80, ge=0, le=100, description="Threshold for fuzzy matching (0-100)")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Career Recommender API",
        "version": "1.0.0",
        "description": "Recommend career paths based on skills and proficiency",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "This information"},
            {"path": "/recommend", "method": "POST", "description": "Get career recommendations based on skills"},
            {"path": "/fields", "method": "GET", "description": "Get all available career fields"},
            {"path": "/specializations", "method": "GET", "description": "Get all available specializations"}
        ]
    }


@app.post("/recommend")
async def recommend(skills_input: SkillInput, params: Optional[RecommendationParams] = None):
    """
    Get career recommendations based on skills and proficiency
    
    Takes a dictionary of skills and proficiency levels (1-100) and returns
    recommended fields and specializations with confidence scores
    """
    if params is None:
        params = RecommendationParams()
        
    try:
        # Set fuzzy threshold in recommender
        recommender.fuzzy_threshold = params.fuzzy_threshold
        
        # Try to use the ML-based recommendation if models are loaded
        try:
            result = recommender.full_recommendation(
                skills=skills_input.skills,
                top_fields=params.top_fields,
                top_specs=params.top_specializations
            )
            return result
        except ValueError as e:
            if "Models not loaded" in str(e):
                # Fall back to a rule-based approach when models aren't loaded
                return fallback_recommendation(skills_input.skills, params.top_fields, params.top_specializations)
            else:
                # Re-raise other ValueErrors
                raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


def fallback_recommendation(skills: Dict[str, int], top_fields: int = 1, top_specs: int = 3):
    """
    Fallback recommendation when ML models aren't loaded
    
    Args:
        skills: Dictionary of skills and proficiency
        top_fields: Number of top fields to return
        top_specs: Number of top specializations to return
        
    Returns:
        Dictionary with top fields and specializations
    """
    # Calculate field matches based on skill overlap
    field_scores = {}
    for field_name, field_data in recommender.fields.items():
        field_skills = field_data.get("core_skills", {})
        matches = recommender._get_matching_skills_for_field(skills, field_name)
        
        # Calculate a confidence score based on matches
        if matches > 0:
            total_skills = len(field_skills)
            confidence = min(round((matches / max(total_skills, 1)) * 100, 2), 100)
            field_scores[field_name] = {
                "field": field_name, 
                "confidence": confidence,
                "matched_skills": matches
            }
    
    # Sort fields by confidence
    top_field_results = sorted(
        field_scores.values(), 
        key=lambda x: x["confidence"], 
        reverse=True
    )[:top_fields]
    
    # Get specializations for top field if any fields were found
    specialization_results = []
    if top_field_results:
        top_field = top_field_results[0]["field"]
        
        # Find specializations in the top field
        for spec_name, spec_data in recommender.specializations.items():
            if spec_data.get("field") == top_field:
                # Get matched and missing skills using recommender's fuzzy matching
                matched_skill_details, missing_skills = recommender._get_skill_details(skills, spec_name)
                
                # Calculate confidence based on matches
                total_skills = len(spec_data.get("core_skills", {}))
                matches = len(matched_skill_details)
                confidence = min(round((matches / max(total_skills, 1)) * 100, 2), 100)
                
                specialization_results.append({
                    "specialization": spec_name,
                    "field": top_field,
                    "confidence": confidence,
                    "matched_skills": matched_skill_details,
                    "missing_skills": missing_skills,
                    "total_skills_required": total_skills,
                    "skills_matched": matches
                })
        
        # Sort by confidence
        specialization_results.sort(key=lambda x: x["confidence"], reverse=True)
        specialization_results = specialization_results[:top_specs]
    
    return {
        "top_fields": top_field_results,
        "specializations": specialization_results
    }


@app.get("/fields")
async def get_fields():
    """Get all available career fields"""
    return recommender.fields


@app.get("/specializations")
async def get_specializations():
    """Get all available specializations"""
    return recommender.specializations


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True) 