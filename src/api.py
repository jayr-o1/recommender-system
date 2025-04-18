from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any
import uvicorn
from datetime import datetime
import logging
import os
import traceback
import asyncio

# Import the actual CareerRecommender
from .recommender import CareerRecommender
# Import the SemanticMatcher to configure progress bars
from utils.semantic_matcher import SemanticMatcher

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable progress bars for API mode
SemanticMatcher.set_progress_bars(False)

# Initialize the FastAPI app with enhanced metadata
app = FastAPI(
    title="Career Pathway Recommender API",
    description="""A comprehensive API for career recommendations based on skills analysis.
                Provides career recommendations based on user's skills and proficiency levels.""",
    version="3.0.0",
    contact={
        "name": "API Support",
        "email": "support@careerpath.com",
    },
    license_info={
        "name": "MIT",
    },
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the recommender with default paths
recommender = CareerRecommender(
    data_path=os.environ.get("DATA_PATH", "data"),
    model_path=os.environ.get("MODEL_PATH", "model")
)

# Data Models
class SkillInput(BaseModel):
    """Skills input model with validation"""
    skills: Dict[str, int] = Field(
        ...,
        description="Dictionary of skills and proficiency levels (1-100)",
        example={"Python": 90, "SQL": 80, "Data Analysis": 85}
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
    top_fields: Optional[int] = Field(3, ge=1, le=5, description="Number of top fields to return")
    top_specializations: Optional[int] = Field(3, ge=1, le=10, description="Number of top specializations to return")
    fuzzy_threshold: Optional[int] = Field(80, ge=0, le=100, description="Threshold for fuzzy matching (0-100)")
    simplified_response: Optional[bool] = Field(False, description="Whether to return a simplified response for frontend")
    use_semantic: Optional[bool] = Field(True, description="Whether to use semantic matching for skills")

class APIStatus(BaseModel):
    """API status model"""
    status: str
    version: str
    timestamp: str
    uptime: float
    dependencies: Dict[str, str]

# Health Check Endpoint
@app.get("/health", response_model=APIStatus)
async def health_check():
    """Check API health and status"""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": 0,  # Replace with actual uptime calculation
        "dependencies": {
            "recommender": "operational" if recommender.models_loaded else "limited",
            "database": "connected"
        }
    }

# Root Endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirects to docs"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

async def get_recommendations(skills: Dict[str, int], params: RecommendationParams) -> dict:
    """
    Get recommendations based on skills and parameters.
    Helper function to handle the recommendation logic for both endpoints.
    """
    logger.info(f"Processing recommendation for {len(skills)} skills")
    
    try:
        # Set fuzzy threshold in recommender and configure semantic matching
        recommender.fuzzy_threshold = params.fuzzy_threshold
        recommender.use_semantic = params.use_semantic
        
        # Try to use the ML-based recommendation if models are loaded
        try:
            # Run the synchronous recommendation function in a separate thread
            # to prevent blocking the event loop
            result = await asyncio.to_thread(
                recommender.full_recommendation,
                skills=skills,
                top_fields=params.top_fields,
                top_specs=params.top_specializations
            )
            
            # Return either a simplified response for frontend or the detailed response
            if params.simplified_response:
                return format_frontend_response(result, list(skills.keys()))
            else:
                return result
                
        except ValueError as e:
            if "Models not loaded" in str(e):
                logger.warning("Falling back to rule-based recommendations")
                # Run the fallback recommendation in a separate thread as well
                fallback_result = await asyncio.to_thread(
                    fallback_recommendation,
                    skills,
                    params.top_fields,
                    params.top_specializations
                )
                
                # Return either a simplified response for frontend or the detailed response
                if params.simplified_response:
                    return format_frontend_response(fallback_result, list(skills.keys()))
                else:
                    return fallback_result
            raise
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        logger.error(traceback.format_exc())
        error_response = {"success": False, "message": f"Error generating recommendations: {str(e)}"}
        if params.simplified_response:
            raise HTTPException(status_code=500, detail=error_response)
        else:
            raise HTTPException(status_code=500, detail=str(e))

# Unified Recommendation Endpoint
@app.post("/recommend", summary="Get career recommendations based on skills")
async def recommend(
    request: Dict[str, Any] = Body(...),
    params: Optional[RecommendationParams] = None
):
    """
    Get career recommendations based on skills and proficiency.
    
    This endpoint accepts a dictionary of skills and their proficiency levels (1-100)
    and returns recommendations for career fields and specializations.
    
    You can provide skills in two formats:
    
    1. Direct format:
    ```json
    {
        "Python": 90,
        "JavaScript": 85,
        "Data Analysis": 70
    }
    ```
    
    2. Nested format:
    ```json
    {
        "skills": {
            "Python": 90,
            "JavaScript": 85, 
            "Data Analysis": 70
        }
    }
    ```
    
    You can control the response format with the params.simplified_response parameter.
    """
    if params is None:
        params = RecommendationParams()
    
    # Check if request contains direct skills or a nested skills object
    if "skills" in request and isinstance(request["skills"], dict):
        # Nested format: {"skills": {"Python": 90, ...}}
        skills = request["skills"]
    elif all(isinstance(val, int) for val in request.values()):
        # Direct format: {"Python": 90, ...}
        skills = request
    else:
        raise HTTPException(
            status_code=422, 
            detail="Invalid request format. Provide skills either directly or in a 'skills' object."
        )
    
    return await get_recommendations(skills, params)

def format_simplified_specialization_response(engine_result, original_skills):
    """
    Format the engine result with specializations as the primary focus
    and fields as supporting information. This creates a flatter structure
    optimized for card-based UI display.
    
    Args:
        engine_result: The raw recommendation engine result
        original_skills: The original skills input by the user
        
    Returns:
        Formatted response with specialization-focused structure
    """
    # Extract all specializations
    spec_list = []
    
    # Get specializations from either key that might exist
    specializations = engine_result.get("specializations", []) or engine_result.get("top_specializations", [])
    
    for spec in specializations:
        # Get basic info
        spec_name = spec.get("specialization", "Unknown")
        field_name = spec.get("field", "General")
        confidence = spec.get("confidence", 0)
        
        # Process matched skills with more detail
        matched_skills = []
        if spec.get("matched_skills"):
            for skill in spec.get("matched_skills", []):
                if isinstance(skill, dict) and "skill" in skill:
                    matched_skills.append({
                        "name": skill.get("skill", ""),
                        "user_skill": skill.get("user_skill", skill.get("skill", "")),
                        "proficiency": skill.get("proficiency", 0),
                        "match_score": skill.get("match_score", 0)
                    })
                elif isinstance(skill, str):
                    # If it's just a string, create a simpler entry
                    matched_skills.append({
                        "name": skill,
                        "proficiency": 50,  # Default value
                        "match_score": 100  # Assume perfect match if no details
                    })
        
        # Also check matched_skill_details if matched_skills is empty
        if not matched_skills and spec.get("matched_skill_details"):
            for skill in spec.get("matched_skill_details", []):
                matched_skills.append({
                    "name": skill.get("skill", ""),
                    "user_skill": skill.get("user_skill", skill.get("skill", "")),
                    "proficiency": skill.get("proficiency", 0),
                    "match_score": skill.get("match_score", 0)
                })
        
        # Process missing skills with priority
        missing_skills = []
        if spec.get("missing_skills"):
            for skill in spec.get("missing_skills", []):
                if isinstance(skill, str):
                    missing_skills.append({
                        "name": skill,
                        "priority": 50  # Default priority
                    })
                elif isinstance(skill, dict) and "skill" in skill:
                    missing_skills.append({
                        "name": skill["skill"],
                        "priority": skill.get("priority", 50)
                    })
        
        # Create the specialization card entry
        spec_card = {
            "title": spec_name,
            "field": field_name,
            "confidence": confidence,
            "matching_skills": matched_skills,
            "missing_skills": missing_skills
        }
        
        spec_list.append(spec_card)
    
    # Sort by confidence
    spec_list.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Return the simplified card-based format
    return {
        "success": True,
        "recommendations": {
            "specialization_cards": spec_list,
            "timestamp": datetime.now().isoformat()
        }
    }

def format_frontend_response(engine_result, original_skills):
    """Format the engine result for frontend consumption"""
    # Use the new specialization-focused format
    return format_simplified_specialization_response(engine_result, original_skills)

def fallback_recommendation(skills: Dict[str, int], top_fields: int = 3, top_specs: int = 3):
    """
    Fallback recommendation when ML models aren't loaded
    """
    field_scores = {}
    for field_name, field_data in recommender.fields.items():
        matches = recommender._get_matching_skills_for_field(skills, field_name)
        
        if matches > 0:
            total_skills = len(field_data.get("core_skills", []))
            confidence = min(round((matches / max(total_skills, 1)) * 100, 2), 100)
            field_scores[field_name] = {
                "field": field_name, 
                "confidence": confidence,
                "matched_skills": matches
            }
    
    top_field_results = sorted(
        field_scores.values(), 
        key=lambda x: x["confidence"], 
        reverse=True
    )[:top_fields]
    
    specialization_results = []
    if top_field_results:
        top_field = top_field_results[0]["field"]
        
        for spec_name, spec_data in recommender.specializations.items():
            if spec_data.get("field") == top_field:
                matched_skill_details, missing_skills = recommender._get_skill_details(skills, spec_name)
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

# New model for skill matching
class SkillMatchInput(BaseModel):
    """Skill matching input model"""
    user_skill: str = Field(..., description="The user's skill to match")
    standard_skill: str = Field(..., description="The standard skill to match against")
    use_semantic: Optional[bool] = Field(True, description="Whether to use semantic matching")

class SkillMatchResult(BaseModel):
    """Skill matching result model"""
    is_match: bool
    score: float
    user_skill: str
    standard_skill: str
    method: str

# Add a new endpoint for skill matching
@app.post("/match_skill", response_model=SkillMatchResult)
async def match_skill(input_data: SkillMatchInput):
    """
    Match a user skill against a standard skill using either fuzzy or semantic matching.
    
    Returns whether it's a match and the match score (0-100).
    """
    try:
        # Set use_semantic in the recommender
        recommender.use_semantic = input_data.use_semantic
        
        # Call the skill matching method in a separate thread to avoid blocking
        is_match, score = await asyncio.to_thread(
            recommender._match_skill_improved,
            input_data.user_skill, 
            input_data.standard_skill
        )
        
        # Format and return the result
        return {
            "is_match": is_match,
            "score": score,
            "user_skill": input_data.user_skill,
            "standard_skill": input_data.standard_skill,
            "method": "semantic" if input_data.use_semantic else "fuzzy"
        }
    except Exception as e:
        logger.error(f"Skill matching error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/output_recommendation", summary="Get career recommendations and output to JSON file")
async def output_recommendation(
    request: Dict[str, Any] = Body(...),
    params: Optional[RecommendationParams] = None
):
    """
    Get career recommendations based on skills and proficiency and output to a JSON file.
    
    This endpoint accepts a dictionary of skills and their proficiency levels (1-100)
    and returns recommendations for career fields and specializations.
    
    You can provide skills in two formats:
    
    1. Direct format:
    ```json
    {
        "Python": 90,
        "JavaScript": 85,
        "Data Analysis": 70
    }
    ```
    
    2. Nested format:
    ```json
    {
        "skills": {
            "Python": 90,
            "JavaScript": 85, 
            "Data Analysis": 70
        }
    }
    ```
    
    The response will include fields, specializations, matching skills, and missing skills
    with confidence scores. The result will also be saved to a JSON file.
    """
    if params is None:
        params = RecommendationParams()
    
    # Check if request contains direct skills or a nested skills object
    if "skills" in request and isinstance(request["skills"], dict):
        # Nested format: {"skills": {"Python": 90, ...}}
        skills = request["skills"]
    elif all(isinstance(val, int) for val in request.values()):
        # Direct format: {"Python": 90, ...}
        skills = request
    else:
        raise HTTPException(
            status_code=422, 
            detail="Invalid request format. Provide skills either directly or in a 'skills' object."
        )
    
    # Get recommendations
    result = await get_recommendations(skills, params)
    
    # Save to JSON file
    try:
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"career_recommendation_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        # Add filename to result
        result["output_file"] = filename
        
        return result
    except Exception as e:
        logger.error(f"Error saving to JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving to JSON: {str(e)}")

# Keep this endpoint for backward compatibility, but it will redirect to the main endpoint
@app.post("/api/recommend", summary="Get simplified career recommendations for frontend")
async def simple_recommend(
    request: Dict[str, Any] = Body(...)
):
    """
    Frontend-friendly endpoint for career recommendations based on skills and proficiency.
    This is a compatibility endpoint that redirects to the main /recommend endpoint.
    
    You can provide skills in the same formats as the /recommend endpoint.
    """
    # Create RecommendationParams with simplified_response=True
    params = RecommendationParams(simplified_response=True)
    
    # Check if request contains direct skills or a nested skills object
    if "skills" in request and isinstance(request["skills"], dict):
        # Nested format: {"skills": {"Python": 90, ...}}
        skills = request["skills"]
    elif all(isinstance(val, int) for val in request.values()):
        # Direct format: {"Python": 90, ...}
        skills = request
    else:
        raise HTTPException(
            status_code=422, 
            detail="Invalid request format. Provide skills either directly or in a 'skills' object."
        )
    
    logger.info(f"Received frontend recommendation request with {len(skills)} skills")
    return await get_recommendations(skills, params)

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout"
                }
            },
            "formatters": {
                "simple": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["console"]
            }
        }
    )