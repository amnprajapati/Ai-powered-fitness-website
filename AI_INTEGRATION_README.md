# 🤖 AI Integration Guide

This guide explains the AI-powered features integrated into the Fitness App backend.

## 🧠 Computer Vision Integration

### Features
- **Body Landmark Detection**: Uses MediaPipe to detect 33 body landmarks
- **Body Measurements**: Extracts shoulder width, hip width, torso length, arm length
- **Body Composition Analysis**: Estimates muscle and fat percentages using color analysis
- **Body Type Classification**: Classifies users as ectomorph, mesomorph, or endomorph
- **Personalized Recommendations**: Generates workout and nutrition plans based on analysis

### Technical Implementation
- **OpenCV**: Image processing and analysis
- **MediaPipe**: Pose detection and body landmark extraction
- **TensorFlow**: Ready for custom body type classification models
- **NumPy**: Numerical computations and array operations

### How It Works
1. **Image Upload**: User uploads body image
2. **Pose Detection**: MediaPipe detects body landmarks
3. **Measurement Extraction**: Calculates body proportions
4. **Composition Analysis**: Analyzes color/texture for body composition
5. **Classification**: Determines body type based on measurements
6. **Recommendations**: Generates personalized fitness advice

### Example Response
```json
{
  "body_type": "mesomorph",
  "muscle_percentage": 35.2,
  "fat_percentage": 18.5,
  "bmi_estimate": 24.1,
  "body_measurements": {
    "shoulder_width": 450.5,
    "hip_width": 380.2,
    "torso_length": 600.0,
    "arm_length": 350.0,
    "shoulder_to_hip_ratio": 1.18
  },
  "recommendations": {
    "workout_focus": ["Balanced training", "Hypertrophy focus"],
    "nutrition_focus": ["Maintenance calories", "Moderate protein"],
    "training_frequency": "4-5 times per week",
    "key_exercises": ["Compound movements", "Isolation exercises"],
    "body_composition_goals": ["Maintain muscle", "Improve definition"]
  },
  "analysis_confidence": 0.85,
  "landmarks_detected": 33
}
```

## 💬 Enhanced Chatbot Integration

### Features
- **GPT-3.5 Integration**: Uses OpenAI's GPT for intelligent responses
- **Intent Detection**: Recognizes user intent (workout, nutrition, motivation, etc.)
- **Contextual Responses**: Provides personalized advice based on user profile
- **Fallback System**: Rule-based responses when GPT is unavailable
- **Conversation Memory**: Maintains context across multiple messages

### Intent Categories
- **Greeting**: Hello, hi, how are you
- **Workout**: Exercise, training, gym, cardio, strength
- **Nutrition**: Diet, food, protein, calories, weight loss/gain
- **Motivation**: Plateau, bored, stuck, consistency
- **Progress**: Tracking, measurements, photos, results
- **Recovery**: Rest, sleep, injury, overtraining
- **Body Type**: Ectomorph, mesomorph, endomorph

### Technical Implementation
- **OpenAI API**: GPT-3.5-turbo for intelligent responses
- **Intent Detection**: Regex patterns for fallback system
- **Context Management**: Maintains conversation history
- **Personalization**: Uses user body type and goals for tailored advice

### Example Usage
```python
# Basic response
response = get_chatbot_response("I need help with my workout routine")

# Contextual response
user_context = {
    "body_type": "ectomorph",
    "goals": "build muscle",
    "experience_level": "beginner"
}
response = get_chatbot_response("What should I eat?", user_context)
```

## 🚀 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the backend directory:
```bash
cp env_template.txt .env
```

Edit `.env` and add your OpenAI API key:
```env
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 3. Get OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an account or sign in
3. Generate a new API key
4. Add it to your `.env` file

### 4. Test AI Integrations
```bash
python test_ai_integration.py
```

### 5. Start the Server
```bash
uvicorn main:app --reload
```

## 📊 API Endpoints

### Computer Vision
- `POST /upload-image`: Upload body image
- `POST /analyze`: Analyze uploaded image for body composition

### Chatbot
- `POST /chatbot`: Get AI-powered fitness advice
  - Parameters: `message`, `user_id` (optional)

## 🔧 Configuration Options

### Computer Vision Settings
```python
# In ml/analyzer.py
self.pose = self.mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,  # 0, 1, or 2
    enable_segmentation=True,
    min_detection_confidence=0.5
)
```

### Chatbot Settings
```python
# In chatbot/bot.py
response = self.openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    max_tokens=300,
    temperature=0.7,  # 0.0-1.0 (creativity)
    top_p=0.9
)
```

## 🧪 Testing

### Run AI Tests
```bash
python test_ai_integration.py
```

### Test Computer Vision
```bash
# Upload an image and test analysis
curl -X POST "http://localhost:8000/upload-image" \
  -F "user_id=1" \
  -F "file=@your_image.jpg"
```

### Test Chatbot
```bash
# Test basic response
curl -X POST "http://localhost:8000/chatbot" \
  -F "message=Hello, I need workout advice"

# Test with user context
curl -X POST "http://localhost:8000/chatbot" \
  -F "message=What should I eat to build muscle?" \
  -F "user_id=1"
```

## 🔒 Security Considerations

### API Key Security
- Never commit API keys to version control
- Use environment variables for sensitive data
- Rotate API keys regularly
- Monitor API usage and costs

### Image Privacy
- Images are processed locally
- No images are stored permanently (unless configured)
- Consider implementing image encryption
- Add user consent for image processing

## 📈 Performance Optimization

### Computer Vision
- Use appropriate image sizes (recommended: 640x480)
- Implement caching for repeated analyses
- Consider batch processing for multiple images
- Optimize MediaPipe settings for your use case

### Chatbot
- Implement response caching
- Use conversation summarization for long chats
- Monitor API response times
- Implement rate limiting

## 🐛 Troubleshooting

### Common Issues

1. **OpenCV Import Error**
   ```bash
   pip install opencv-python-headless  # For server environments
   ```

2. **MediaPipe Not Working**
   ```bash
   pip install mediapipe==0.10.3  # Use specific version
   ```

3. **OpenAI API Errors**
   - Check API key is correct
   - Verify account has credits
   - Check rate limits

4. **Memory Issues**
   - Reduce image resolution
   - Implement image cleanup
   - Monitor memory usage

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Future Enhancements

### Computer Vision
- [ ] Custom TensorFlow model for body type classification
- [ ] Muscle group detection and analysis
- [ ] Progress tracking with image comparison
- [ ] Real-time pose estimation

### Chatbot
- [ ] Voice interaction
- [ ] Multi-language support
- [ ] Integration with fitness trackers
- [ ] Personalized workout generation

### General
- [ ] Machine learning model training pipeline
- [ ] A/B testing for response optimization
- [ ] Analytics dashboard
- [ ] Mobile app integration

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the test output
3. Check API documentation
4. Monitor server logs

---

**Note**: This AI integration provides a solid foundation for intelligent fitness assistance. The computer vision system uses advanced pose detection, while the chatbot offers both GPT-powered and rule-based responses for reliability. 