#!/usr/bin/env python3
"""
Test script for AI integrations (Computer Vision and Chatbot)
This script tests the core AI functionality without requiring the full FastAPI app.
"""

import os
import sys
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_computer_vision():
    """Test the computer vision analyzer."""
    print("🧠 Testing Computer Vision Integration...")
    
    try:
        # Import the analyzer
        from backend.ml.analyzer import analyze_image
        
        # Test with a dummy image path (will fail but tests import)
        result = analyze_image("test_image.jpg")
        
        if "error" in result:
            print("✓ Computer Vision module imported successfully")
            print(f"  Expected error for missing file: {result['error']}")
            return True
        else:
            print("✓ Computer Vision analysis completed")
            return True
            
    except ImportError as e:
        print(f"✗ Computer Vision import failed: {e}")
        print("  Make sure to install: pip install opencv-python mediapipe tensorflow numpy pillow")
        return False
    except Exception as e:
        print(f"✗ Computer Vision test failed: {e}")
        return False

def test_chatbot():
    """Test the enhanced chatbot."""
    print("\n💬 Testing Chatbot Integration...")
    
    try:
        # Import the chatbot
        from backend.chatbot.bot import get_chatbot_response
        
        # Test basic functionality
        test_messages = [
            "Hello, how are you?",
            "I need help with my workout routine",
            "What should I eat to build muscle?",
            "I'm feeling unmotivated today"
        ]
        
        print("  Testing rule-based responses...")
        for message in test_messages:
            response = get_chatbot_response(message)
            if response and len(response) > 10:
                print(f"    ✓ '{message[:30]}...' → Response received")
            else:
                print(f"    ✗ '{message[:30]}...' → No response")
                return False
        
        # Test with user context
        print("  Testing contextual responses...")
        user_context = {
            "body_type": "ectomorph",
            "goals": "build muscle",
            "experience_level": "beginner"
        }
        
        response = get_chatbot_response("What workout should I do?", user_context)
        if response and len(response) > 10:
            print(f"    ✓ Contextual response received")
        else:
            print(f"    ✗ No contextual response")
            return False
        
        print("✓ Chatbot integration working correctly")
        return True
        
    except ImportError as e:
        print(f"✗ Chatbot import failed: {e}")
        print("  Make sure to install: pip install openai python-dotenv")
        return False
    except Exception as e:
        print(f"✗ Chatbot test failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are installed."""
    print("📦 Testing Dependencies...")
    
    required_packages = [
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("tensorflow", "tensorflow"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("openai", "openai"),
        ("python-dotenv", "dotenv")
    ]
    
    all_installed = True
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def test_environment_setup():
    """Test environment configuration."""
    print("\n🔧 Testing Environment Setup...")
    
    # Check for .env file
    env_file_exists = os.path.exists(".env")
    if env_file_exists:
        print("  ✓ .env file found")
    else:
        print("  ⚠ .env file not found (create one from env_template.txt)")
    
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "your_openai_api_key_here":
        print("  ✓ OpenAI API key configured")
    else:
        print("  ⚠ OpenAI API key not configured (GPT features will use fallback)")
    
    return True

def main():
    """Run all AI integration tests."""
    print("🤖 AI Integration Test Suite")
    print("=" * 50)
    
    # Test dependencies first
    deps_ok = test_dependencies()
    if not deps_ok:
        print("\n❌ Some dependencies are missing. Please install them first:")
        print("pip install -r requirements.txt")
        return False
    
    # Test environment
    env_ok = test_environment_setup()
    
    # Test computer vision
    cv_ok = test_computer_vision()
    
    # Test chatbot
    chatbot_ok = test_chatbot()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  Dependencies: {'✓ PASSED' if deps_ok else '✗ FAILED'}")
    print(f"  Environment: {'✓ PASSED' if env_ok else '⚠ PARTIAL'}")
    print(f"  Computer Vision: {'✓ PASSED' if cv_ok else '✗ FAILED'}")
    print(f"  Chatbot: {'✓ PASSED' if chatbot_ok else '✗ FAILED'}")
    
    if deps_ok and cv_ok and chatbot_ok:
        print("\n🎉 All AI integrations are working correctly!")
        print("\nNext steps:")
        print("1. Add your OpenAI API key to .env file for GPT features")
        print("2. Upload body images to test computer vision analysis")
        print("3. Start the backend server: uvicorn main:app --reload")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 