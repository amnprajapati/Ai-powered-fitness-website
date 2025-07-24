#!/usr/bin/env python3
"""
Lightweight test for AI integration structure
Tests the code structure without requiring heavy dependencies
"""

import os
import sys
import importlib.util

def test_file_structure():
    """Test that all required files exist."""
    print("📁 Testing File Structure...")
    
    required_files = [
        "ml/analyzer.py",
        "chatbot/bot.py",
        "requirements.txt",
        "env_template.txt",
        "AI_INTEGRATION_README.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def test_import_structure():
    """Test that modules can be imported (without heavy deps)."""
    print("\n📦 Testing Import Structure...")
    
    # Test basic imports
    try:
        import typing
        print("  ✓ typing")
    except ImportError:
        print("  ✗ typing")
        return False
    
    try:
        import logging
        print("  ✓ logging")
    except ImportError:
        print("  ✗ logging")
        return False
    
    # Test our modules (without heavy deps)
    try:
        # Test analyzer structure
        spec = importlib.util.spec_from_file_location("analyzer", "ml/analyzer.py")
        if spec:
            print("  ✓ ml/analyzer.py structure")
        else:
            print("  ✗ ml/analyzer.py structure")
            return False
    except Exception as e:
        print(f"  ✗ ml/analyzer.py: {e}")
        return False
    
    try:
        # Test chatbot structure
        spec = importlib.util.spec_from_file_location("bot", "chatbot/bot.py")
        if spec:
            print("  ✓ chatbot/bot.py structure")
        else:
            print("  ✗ chatbot/bot.py structure")
            return False
    except Exception as e:
        print(f"  ✗ chatbot/bot.py: {e}")
        return False
    
    return True

def test_requirements():
    """Test requirements.txt content."""
    print("\n📋 Testing Requirements...")
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
            
        required_packages = [
            "opencv-python",
            "mediapipe", 
            "tensorflow",
            "numpy",
            "pillow",
            "openai",
            "python-dotenv"
        ]
        
        missing = []
        for package in required_packages:
            if package in content:
                print(f"  ✓ {package}")
            else:
                print(f"  ✗ {package} - MISSING")
                missing.append(package)
        
        return len(missing) == 0
        
    except FileNotFoundError:
        print("  ✗ requirements.txt not found")
        return False

def test_environment_template():
    """Test environment template."""
    print("\n🔧 Testing Environment Template...")
    
    try:
        with open("env_template.txt", "r") as f:
            content = f.read()
            
        required_vars = [
            "OPENAI_API_KEY",
            "DATABASE_URL",
            "SECRET_KEY"
        ]
        
        missing = []
        for var in required_vars:
            if var in content:
                print(f"  ✓ {var}")
            else:
                print(f"  ✗ {var} - MISSING")
                missing.append(var)
        
        return len(missing) == 0
        
    except FileNotFoundError:
        print("  ✗ env_template.txt not found")
        return False

def main():
    """Run all structure tests."""
    print("🤖 AI Integration Structure Test")
    print("=" * 50)
    
    # Test file structure
    files_ok = test_file_structure()
    
    # Test import structure
    imports_ok = test_import_structure()
    
    # Test requirements
    reqs_ok = test_requirements()
    
    # Test environment template
    env_ok = test_environment_template()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Structure Test Results:")
    print(f"  File Structure: {'✓ PASSED' if files_ok else '✗ FAILED'}")
    print(f"  Import Structure: {'✓ PASSED' if imports_ok else '✗ FAILED'}")
    print(f"  Requirements: {'✓ PASSED' if reqs_ok else '✗ FAILED'}")
    print(f"  Environment Template: {'✓ PASSED' if env_ok else '✗ FAILED'}")
    
    if files_ok and imports_ok and reqs_ok and env_ok:
        print("\n🎉 AI integration structure is correct!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up environment: cp env_template.txt .env")
        print("3. Add OpenAI API key to .env file")
        print("4. Run full tests: python test_ai_integration.py")
        return True
    else:
        print("\n❌ Some structure tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 