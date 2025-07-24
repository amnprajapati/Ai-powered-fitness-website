import re
import random
import os
import json
from typing import Dict, List, Optional, Tuple
import logging
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FitnessChatbot:
    def __init__(self):
        """Initialize the fitness chatbot with GPT integration."""
        self.openai_client = None
        self.use_gpt = False
        
        # Try to initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
                self.use_gpt = True
                logger.info("OpenAI GPT integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.use_gpt = False
        else:
            logger.info("No OpenAI API key found, using rule-based responses")
        
        # Intent patterns for fallback
        self.intent_patterns = {
            "greeting": [
                r"\b(hello|hi|hey|start|good morning|good afternoon|good evening)\b",
                r"\b(how are you|what's up)\b"
            ],
            "workout": [
                r"\b(workout|exercise|training|gym|fitness|routine)\b",
                r"\b(cardio|strength|muscle|weight|lifting)\b",
                r"\b(beginner|advanced|intermediate)\b"
            ],
            "nutrition": [
                r"\b(diet|nutrition|food|meal|protein|calories|macros)\b",
                r"\b(weight loss|weight gain|bulking|cutting)\b",
                r"\b(supplements|vitamins|minerals)\b"
            ],
            "motivation": [
                r"\b(motivation|motivated|stuck|plateau|bored|tired)\b",
                r"\b(consistency|discipline|habit|routine)\b",
                r"\b(goal|target|achievement|success)\b"
            ],
            "progress": [
                r"\b(progress|results|improvement|change|transformation)\b",
                r"\b(measure|track|monitor|log)\b",
                r"\b(photo|picture|before|after)\b"
            ],
            "recovery": [
                r"\b(rest|recovery|sleep|rest day|overtraining)\b",
                r"\b(injury|pain|sore|stiff)\b",
                r"\b(stretch|flexibility|mobility)\b"
            ],
            "body_type": [
                r"\b(body type|ectomorph|mesomorph|endomorph|somatotype)\b",
                r"\b(thin|muscular|stocky|lean|athletic)\b"
            ]
        }
        
        # Context for GPT conversations
        self.conversation_context = []
        
    def get_response(self, message: str, user_context: Optional[Dict] = None) -> Dict:
        """
        Get chatbot response using GPT or fallback to rule-based system.
        
        Args:
            message: User's message
            user_context: Optional user context (body type, goals, etc.)
            
        Returns:
            Dictionary with response and metadata
        """
        session_log = []
        try:
            # Try GPT first if available
            if self.use_gpt and self.openai_client:
                response = self._get_gpt_response(message, user_context)
                if response:
                    result = {
                        "response": response,
                        "source": "gpt",
                        "confidence": 0.9
                    }
                    session_log.append({"user": message, "bot": response, "source": "gpt"})
                    self._log_session(session_log)
                    return result
                else:
                    # GPT failed at runtime, fallback
                    logger.warning("GPT API failed at runtime, using rule-based fallback.")
                    fallback_msg = "AI is busy, using standard advice. "
                    response = fallback_msg + self._get_rule_based_response(message, user_context)
                    result = {
                        "response": response,
                        "source": "rule_based_fallback",
                        "confidence": 0.5
                    }
                    session_log.append({"user": message, "bot": response, "source": "rule_based_fallback"})
                    self._log_session(session_log)
                    return result
            # Fallback to rule-based system
            response = self._get_rule_based_response(message, user_context)
            result = {
                "response": response,
                "source": "rule_based",
                "confidence": 0.7
            }
            session_log.append({"user": message, "bot": response, "source": "rule_based"})
            self._log_session(session_log)
            return result
        except Exception as e:
            logger.error(f"Error getting chatbot response: {e}")
            error_msg = "I'm having trouble processing your request right now. Please try again later."
            session_log.append({"user": message, "bot": error_msg, "source": "error"})
            self._log_session(session_log)
            return {
                "response": error_msg,
                "source": "error",
                "confidence": 0.0
            }
    
    def _get_gpt_response(self, message: str, user_context: Optional[Dict] = None) -> Optional[str]:
        """Get response from OpenAI GPT."""
        try:
            # Build system prompt with fitness context
            system_prompt = self._build_system_prompt(user_context)
            
            # Build conversation history
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add recent conversation context (last 5 messages)
            for context_msg in self.conversation_context[-5:]:
                messages.append(context_msg)
            
            # Add current user message
            messages.append({"role": "user", "content": message})
            
            # Get response from GPT
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=300,
                temperature=0.7,
                top_p=0.9
            )
            
            gpt_response = response.choices[0].message.content.strip()
            
            # Update conversation context
            self.conversation_context.append({"role": "user", "content": message})
            self.conversation_context.append({"role": "assistant", "content": gpt_response})
            
            # Keep only last 10 messages to manage context length
            if len(self.conversation_context) > 10:
                self.conversation_context = self.conversation_context[-10:]
            
            return gpt_response
            
        except Exception as e:
            logger.error(f"GPT API error: {e}")
            return None
    
    def _build_system_prompt(self, user_context: Optional[Dict] = None) -> str:
        """Build system prompt for GPT with fitness context."""
        base_prompt = """You are an AI fitness assistant with expertise in:
- Workout planning and exercise techniques
- Nutrition and diet advice
- Body type analysis (ectomorph, mesomorph, endomorph)
- Motivation and goal setting
- Progress tracking and recovery

Provide helpful, accurate, and encouraging fitness advice. Keep responses concise but informative. Always prioritize safety and recommend consulting professionals for medical concerns."""

        if user_context:
            context_info = []
            if user_context.get("body_type"):
                context_info.append(f"User's body type: {user_context['body_type']}")
            if user_context.get("goals"):
                context_info.append(f"User's goals: {user_context['goals']}")
            if user_context.get("experience_level"):
                context_info.append(f"Experience level: {user_context['experience_level']}")
            
            if context_info:
                base_prompt += f"\n\nUser context: {', '.join(context_info)}"
        
        return base_prompt
    
    def _get_rule_based_response(self, message: str, user_context: Optional[Dict] = None) -> str:
        """Get response using rule-based system with intent detection."""
        message_lower = message.lower().strip()
        
        # Detect intent
        intent = self._detect_intent(message_lower)
        
        # Get contextual response based on intent and user context
        response = self._get_contextual_response(intent, message_lower, user_context)
        
        return response
    
    def _detect_intent(self, message: str) -> str:
        """Detect user intent from message."""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return intent
        
        return "general"
    
    def _get_contextual_response(self, intent: str, message: str, user_context: Optional[Dict] = None) -> str:
        """Get contextual response based on intent and user context."""
        
        if intent == "greeting":
            return self._get_greeting_response(user_context)
        elif intent == "workout":
            return self._get_workout_response(message, user_context)
        elif intent == "nutrition":
            return self._get_nutrition_response(message, user_context)
        elif intent == "motivation":
            return self._get_motivation_response(message, user_context)
        elif intent == "progress":
            return self._get_progress_response(message, user_context)
        elif intent == "recovery":
            return self._get_recovery_response(message, user_context)
        elif intent == "body_type":
            return self._get_body_type_response(message, user_context)
        else:
            return self._get_general_response(message, user_context)
    
    def _get_greeting_response(self, user_context: Optional[Dict] = None) -> str:
        """Get personalized greeting response."""
        greetings = [
            "Hello! I'm your AI fitness assistant. How can I help you today?",
            "Hi there! Ready to crush your fitness goals? What would you like to know?",
            "Hey! I'm here to support your fitness journey. What's on your mind?"
        ]
        
        if user_context and user_context.get("body_type"):
            body_type = user_context["body_type"]
            if body_type == "ectomorph":
                return f"Hello! I see you're an ectomorph. Let's work on building that muscle mass! What can I help you with today?"
            elif body_type == "mesomorph":
                return f"Hi there! As a mesomorph, you have great potential for both strength and definition. What's your focus today?"
            elif body_type == "endomorph":
                return f"Hey! I'm here to help you optimize your endomorph body type for your goals. What would you like to work on?"
        
        return random.choice(greetings)
    
    def _get_workout_response(self, message: str, user_context: Optional[Dict] = None) -> str:
        """Get workout-related response."""
        if "beginner" in message or "start" in message:
            return "For beginners, start with bodyweight exercises: push-ups, squats, planks, and lunges. Begin with 3 sets of 10 reps each, 3 times per week. Focus on proper form over weight. Would you like a specific beginner program?"
        
        elif "cardio" in message:
            return "Great cardio options: brisk walking (30 min), cycling, swimming, or HIIT workouts. Start with 20-30 minutes, 3-4 times per week. What's your current fitness level and what cardio do you enjoy?"
        
        elif "strength" in message or "muscle" in message:
            return "For strength training, focus on compound movements: squats, deadlifts, bench press, rows, and overhead press. Start with 3-4 sets of 8-12 reps, 2-3 times per week. Do you have access to a gym or equipment?"
        
        else:
            body_type = user_context.get("body_type") if user_context else None
            if body_type == "ectomorph":
                return "As an ectomorph, focus on compound movements with progressive overload. Train 3-4 times per week with adequate rest. What specific area would you like to work on?"
            elif body_type == "mesomorph":
                return "Mesomorphs respond well to both strength and hypertrophy training. Mix compound movements with isolation exercises. What's your current goal?"
            elif body_type == "endomorph":
                return "For endomorphs, prioritize cardio and circuit training. Include strength training 2-3 times per week. What type of workout interests you most?"
            else:
                return "I can help with various workout types! Are you interested in cardio, strength training, flexibility, or something specific?"
    
    def _get_nutrition_response(self, message: str, user_context: Optional[Dict] = None) -> str:
        """Get nutrition-related response."""
        if "protein" in message:
            return "Aim for 0.8-1.2g of protein per pound of body weight. Good sources: lean meats, fish, eggs, legumes, dairy, and protein powder. Are you looking to build muscle or lose weight?"
        
        elif "calories" in message:
            return "Your daily calorie needs depend on goals and activity level. For weight loss: create a 300-500 calorie deficit. For muscle gain: add 200-300 calories above maintenance. What's your primary goal?"
        
        elif "weight loss" in message:
            return "Weight loss requires a calorie deficit, regular exercise, and consistency. Start with a 300-500 calorie daily deficit and 150 minutes of cardio per week. What's your current eating pattern?"
        
        else:
            body_type = user_context.get("body_type") if user_context else None
            if body_type == "ectomorph":
                return "Ectomorphs need a calorie surplus to build muscle. Focus on high protein, complex carbs, and healthy fats. Eat 5-6 meals per day. What's your current calorie intake?"
            elif body_type == "mesomorph":
                return "Mesomorphs can maintain or slightly increase calories. Focus on balanced macros: 40% carbs, 30% protein, 30% fats. What's your current nutrition goal?"
            elif body_type == "endomorph":
                return "Endomorphs benefit from a calorie deficit and low-carb approach. Prioritize protein and vegetables, limit refined carbs. What's your current diet like?"
            else:
                return "Nutrition is key to fitness success! Are you looking for meal planning, macro guidance, or specific dietary advice?"
    
    def _get_motivation_response(self, message: str, user_context: Optional[Dict] = None) -> str:
        """Get motivation-related response."""
        if "plateau" in message:
            return "Plateaus are normal! Try changing your routine, increasing intensity, or adding new exercises. Sometimes a deload week helps too. What specific plateau are you experiencing?"
        
        elif "bored" in message:
            return "Try new activities: different workout styles, outdoor training, group classes, or sports. Variety keeps things exciting and challenges your body in new ways. What sounds fun to you?"
        
        else:
            motivational_quotes = [
                "Progress is progress, no matter how small. Keep going!",
                "Your future self is watching you right now through memories. Make them proud.",
                "The only bad workout is the one that didn't happen.",
                "Consistency beats perfection every time.",
                "You don't have to be great to start, but you have to start to be great."
            ]
            return random.choice(motivational_quotes) + " What's your biggest challenge right now?"
    
    def _get_progress_response(self, message: str, user_context: Optional[Dict] = None) -> str:
        """Get progress-tracking response."""
        if "measure" in message or "track" in message:
            return "Track progress with: photos (monthly), measurements (weekly), weight (weekly), and performance metrics. Focus on trends, not daily fluctuations. What metrics are you currently tracking?"
        
        elif "photo" in message or "picture" in message:
            return "Progress photos are great! Take them monthly in consistent lighting and poses. Front, side, and back views. Remember, changes happen gradually - be patient with the process."
        
        else:
            return "Progress tracking helps stay motivated! Consider tracking: weight, measurements, photos, strength gains, and how clothes fit. What's your preferred way to measure progress?"
    
    def _get_recovery_response(self, message: str, user_context: Optional[Dict] = None) -> str:
        """Get recovery-related response."""
        if "rest" in message or "recovery" in message:
            return "Rest is crucial for progress! Aim for 7-9 hours of sleep, take 1-2 rest days per week, and listen to your body. Signs you need rest: fatigue, decreased performance, mood changes."
        
        elif "injury" in message or "pain" in message:
            return "If you're experiencing pain, stop the exercise immediately. Rest, ice, and consult a healthcare professional if pain persists. Prevention: proper form, gradual progression, and adequate warm-up."
        
        else:
            return "Recovery includes: sleep, nutrition, hydration, stretching, and rest days. What aspect of recovery would you like to improve?"
    
    def _get_body_type_response(self, message: str, user_context: Optional[Dict] = None) -> str:
        """Get body type-related response."""
        if "ectomorph" in message:
            return "Ectomorphs are naturally thin with difficulty gaining weight. Focus on: calorie surplus, compound movements, progressive overload, and adequate protein. Train 3-4 times per week."
        
        elif "mesomorph" in message:
            return "Mesomorphs build muscle easily and have athletic builds. Focus on: balanced training, moderate cardio, and maintenance calories. Train 4-5 times per week."
        
        elif "endomorph" in message:
            return "Endomorphs gain weight easily and have difficulty losing fat. Focus on: calorie deficit, cardio training, and strength training. Train 5-6 times per week."
        
        else:
            return "Body types (somatotypes) are: Ectomorph (thin, hard to gain weight), Mesomorph (athletic, builds muscle easily), and Endomorph (stocky, gains weight easily). What's your body type?"
    
    def _get_general_response(self, message: str, user_context: Optional[Dict] = None) -> str:
        """Get general fitness advice response."""
        general_tips = [
            "Start slow and gradually increase intensity to avoid injury.",
            "Consistency beats perfection - aim for 3-4 workouts per week.",
            "Stay hydrated and fuel your body with nutritious foods.",
            "Find activities you enjoy - fitness should be fun!",
            "Set realistic goals and celebrate small wins along the way."
        ]
        
        return random.choice(general_tips) + " I can help with workouts, nutrition, motivation, or progress tracking. What interests you most?"

    def _log_session(self, session_log):
        """Append session log to a local JSON file (chatbot_sessions.json)."""
        log_file = os.path.join(os.path.dirname(__file__), "chatbot_sessions.json")
        try:
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []
            data.extend(session_log)
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to log chatbot session: {e}")

class PlanUpdate(BaseModel):
    workout: str | None = None
    meal: str | None = None

class ProgressUpdate(BaseModel):
    weight: float | None = None
    photo: str | None = None  # Path or URL to photo

# Global chatbot instance
chatbot = FitnessChatbot()

def get_chatbot_response(message: str, user_context: Optional[Dict] = None) -> str:
    """
    Main function to get chatbot response.
    This is the interface used by the API.
    """
    result = chatbot.get_response(message, user_context)
    return result["response"] 