import requests
import json
import argparse
import time
import subprocess
import sys
import os
import signal
import atexit
from datetime import datetime

# Global variable to store the server process
server_process = None

def start_api_server(port=8000, host="127.0.0.1"):
    """Start the API server using Uvicorn"""
    global server_process
    
    print(f"Starting API server at {host}:{port}...")
    try:
        # Using subprocess to start the server as a background process
        cmd = [sys.executable, "-m", "uvicorn", "src.api:app", "--host", host, "--port", str(port)]
        server_process = subprocess.Popen(cmd)
        
        # Give the server a moment to start
        time.sleep(2)
        
        # Register a function to kill the server when the script exits
        atexit.register(stop_api_server)
        
        # Check if the server is running by making a request to the health endpoint
        response = requests.get(f"http://{host}:{port}/health")
        if response.status_code == 200:
            print(f"API server started successfully at http://{host}:{port}")
            return True
        else:
            print(f"Failed to verify server is running: {response.status_code}")
            stop_api_server()
            return False
    except Exception as e:
        print(f"Error starting API server: {e}")
        if server_process:
            stop_api_server()
        return False

def stop_api_server():
    """Stop the API server"""
    global server_process
    if server_process:
        print("Stopping API server...")
        try:
            if os.name == 'nt':  # Windows
                server_process.terminate()
            else:  # Unix/Linux
                os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
            server_process.wait(timeout=5)
            print("API server stopped.")
        except Exception as e:
            print(f"Error stopping server: {e}")
            # Force kill if terminate doesn't work
            server_process.kill()
        finally:
            server_process = None

def get_recommendation_to_json(skills, output_file=None, use_semantic=True, fuzzy_threshold=65, top_fields=3, top_specs=5, port=8000):
    """
    Get a recommendation from the API and save it to a JSON file
    
    Args:
        skills: Dictionary of skills and proficiency
        output_file: Path to save the JSON output (defaults to timestamp-based filename)
        use_semantic: Whether to use semantic matching
        fuzzy_threshold: Fuzzy matching threshold
        top_fields: Number of top fields to return
        top_specs: Number of top specializations to return
        port: Port where the API is running
    """
    # Default output file if none provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"recommendation_{timestamp}.json"
    
    # Prepare the API request
    url = f"http://127.0.0.1:{port}/recommend"
    
    # Create parameters object
    params = {
        "top_fields": top_fields,
        "top_specializations": top_specs,
        "fuzzy_threshold": fuzzy_threshold,
        "use_semantic": use_semantic,
        "simplified_response": False
    }
    
    # Setup the request payload - wrap in request object according to API expectation
    data = {"request": skills}
    
    # Get recommendations
    try:
        start_time = time.time()
        
        print("\n===== GETTING CAREER RECOMMENDATION FROM API =====")
        print(f"Semantic Matching: {'Enabled' if use_semantic else 'Disabled'}")
        print(f"Fuzzy Threshold: {fuzzy_threshold}")
        
        print("\nSkills input:")
        for skill, level in skills.items():
            print(f"- {skill}: {level}")
        
        # Make the API request
        response = requests.post(url, json=data, params=params)
        
        # If the main endpoint fails, try the legacy endpoint
        if response.status_code != 200:
            print(f"Main endpoint failed with status {response.status_code}, trying legacy endpoint...")
            legacy_url = f"http://127.0.0.1:{port}/api/recommend"
            # Try the legacy endpoint with both formats
            response = requests.post(legacy_url, json=data)
            if response.status_code != 200:
                response = requests.post(legacy_url, json=skills)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"\nProcessing time: {processing_time:.2f} seconds")
        
        if response.status_code == 200:
            recommendations = response.json()
            
            # Save to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(recommendations, f, indent=2)
            
            print(f"\nRecommendation saved to: {output_file}")
            
            # Print a summary of the recommendations
            print("\n===== RECOMMENDATION SUMMARY =====")
            
            # Handle different response formats
            if "recommendations" in recommendations:
                # Legacy endpoint format
                print("\nResponse from legacy endpoint (/api/recommend)")
                
                top_fields = recommendations.get("recommendations", {}).get("top_fields", [])
                top_specializations = recommendations.get("recommendations", {}).get("top_specializations", [])
                
                print("\nTOP FIELDS:")
                for i, field in enumerate(top_fields, 1):
                    print(f"{i}. {field.get('field', 'Unknown')} (Confidence: {field.get('match_percentage', 0) * 100:.1f}%)")
                
                print("\nTOP SPECIALIZATIONS:")
                for i, spec in enumerate(top_specializations, 1):
                    print(f"{i}. {spec.get('specialization', 'Unknown')} (Field: {spec.get('field', 'Unknown')}, Confidence: {spec.get('match_percentage', 0) * 100:.1f}%)")
                    print(f"   Matching Skills: {len(spec.get('matching_skills', []))}, Missing Skills: {len(spec.get('missing_skills', []))}")
            else:
                # Standard endpoint format
                print("\nResponse from standard endpoint (/recommend)")
                
                print("\nTOP FIELDS:")
                for i, field in enumerate(recommendations.get("fields", []), 1):
                    print(f"{i}. {field['field']} (Confidence: {field['confidence']:.1f}%)")
                    
                print("\nTOP SPECIALIZATIONS:")
                for i, spec in enumerate(recommendations.get("specializations", []), 1):
                    print(f"{i}. {spec['specialization']} (Field: {spec.get('field', 'Unknown')}, Confidence: {spec['confidence'] * 100:.1f}%)")
                    print(f"   Matched Skills: {len(spec.get('matched_skill_details', []))}, Missing Skills: {len(spec.get('missing_skills', []))}")
            
            return output_file
        else:
            print(f"Error getting recommendations: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return None

def parse_custom_skills(skills_string):
    """Parse a string of skills into a dictionary"""
    if not skills_string:
        return None
        
    skills = {}
    parts = skills_string.split(',')
    
    for part in parts:
        part = part.strip()
        if ':' in part:
            skill, prof = part.split(':', 1)
            try:
                skills[skill.strip()] = int(prof.strip())
            except ValueError:
                skills[skill.strip()] = 70  # Default if not a valid number
        else:
            skills[part] = 70  # Default proficiency
            
    return skills

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get career recommendations and save to JSON file')
    parser.add_argument('--skills', type=str, help='Comma-separated list of skills with optional proficiency (e.g., "Python:90,Data Analysis:85")')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--no-semantic', action='store_true', help='Disable semantic matching')
    parser.add_argument('--threshold', type=int, default=65, help='Fuzzy matching threshold (0-100)')
    parser.add_argument('--top-fields', type=int, default=3, help='Number of top fields to return')
    parser.add_argument('--top-specs', type=int, default=5, help='Number of top specializations to return')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the API on')
    parser.add_argument('--host', type=str, default="127.0.0.1", help='Host to run the API on')
    parser.add_argument('--no-server', action='store_true', help='Don\'t start the server (assume it\'s already running)')
    
    args = parser.parse_args()
    
    # Parse custom skills if provided, otherwise use default skills
    if args.skills:
        skills = parse_custom_skills(args.skills)
    else:
        # Default test skills
        skills = {
            "Python": 90,
            "JavaScript": 85,
            "Data Analysis": 80,
            "Machine Learning": 75,
            "Project Management": 70
        }
    
    # Start the API server if needed
    server_started = False
    if not args.no_server:
        server_started = start_api_server(port=args.port, host=args.host)
        if not server_started:
            print("Failed to start API server. Exiting.")
            sys.exit(1)
    
    try:
        # Get recommendation and save to JSON
        output_file = get_recommendation_to_json(
            skills=skills,
            output_file=args.output,
            use_semantic=not args.no_semantic,
            fuzzy_threshold=args.threshold,
            top_fields=args.top_fields,
            top_specs=args.top_specs,
            port=args.port
        )
        
        if output_file:
            print(f"\nSuccess! Recommendation data saved to {output_file}")
        else:
            print("\nFailed to get recommendations.")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        # Stop the server if we started it
        if server_started:
            stop_api_server() 