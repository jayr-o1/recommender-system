import requests
import json

def test_recommend_direct():
    """Test the recommend endpoint with direct skill-proficiency pairs"""
    url = "http://127.0.0.1:8000/recommend"
    
    # Direct skill-proficiency dictionary wrapped in "request" as expected by FastAPI
    data = {
        "request": {
            "Python": 90,
            "JavaScript": 85,
            "Data Analysis": 70
        }
    }
    
    try:
        response = requests.post(url, json=data)
        print("\n=== Testing recommend endpoint with direct skill-proficiency pairs ===")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_recommend_nested():
    """Test the recommend endpoint with nested skills object"""
    url = "http://127.0.0.1:8000/recommend"
    
    # Nested skills object wrapped in "request" as expected by FastAPI
    data = {
        "request": {
            "skills": {
                "Python": 90,
                "JavaScript": 85,
                "Data Analysis": 70
            }
        }
    }
    
    try:
        response = requests.post(url, json=data)
        print("\n=== Testing recommend endpoint with nested skills object ===")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_api_recommend():
    """Test the /api/recommend endpoint"""
    url = "http://127.0.0.1:8000/api/recommend"
    
    # Direct skill-proficiency dictionary for the legacy endpoint
    # This one seems to be working correctly already
    data = {
        "Python": 90,
        "JavaScript": 85,
        "Data Analysis": 70
    }
    
    try:
        response = requests.post(url, json=data)
        print("\n=== Testing /api/recommend endpoint ===")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_recommend_direct()
    test_recommend_nested()
    test_api_recommend() 