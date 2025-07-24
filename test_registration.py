#!/usr/bin/env python3
"""
Quick test script to verify registration endpoint functionality
"""
import requests
import json

def test_registration():
    url = "http://127.0.0.1:8000/register"
    
    # Test data
    test_user = {
        "username": "testuser123",
        "email": "test@example.com", 
        "password": "testpassword123",
        "role": "user"
    }
    
    try:
        print("Testing registration endpoint...")
        print(f"URL: {url}")
        print(f"Data: {test_user}")
        
        # Send POST request
        response = requests.post(url, data=test_user)
        
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        try:
            response_json = response.json()
            print(f"Response JSON: {json.dumps(response_json, indent=2)}")
        except:
            print(f"Response Text: {response.text}")
            
        if response.status_code == 200:
            print("✅ Registration successful!")
        else:
            print("❌ Registration failed!")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is FastAPI running on port 8000?")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_server_health():
    try:
        response = requests.get("http://127.0.0.1:8000/")
        print(f"Server health check: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Server health check failed: {str(e)}")

if __name__ == "__main__":
    print("=== FastAPI Registration Test ===")
    test_server_health()
    print("\n" + "="*40)
    test_registration()
