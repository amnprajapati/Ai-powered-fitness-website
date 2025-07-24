import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import os
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import logging
import imghdr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BodyAnalyzer:
    def __init__(self):
        """Initialize the body analyzer with MediaPipe and TensorFlow models."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        # Initialize body type classifier (placeholder for trained model)
        self.body_type_model = self._load_body_type_model()
        
    def _load_body_type_model(self):
        """Load the trained body type classification model."""
        # For now, we'll use a simple rule-based classifier
        # In production, this would load a trained TensorFlow model
        logger.info("Loading body type classification model...")
        return None  # Placeholder for actual model
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze body image using computer vision and return detailed analysis.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing body analysis results
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return {
                    "error": "Image file not found",
                    "body_type": "unknown",
                    "muscle_percentage": 0.0,
                    "fat_percentage": 0.0
                }
            # Check file type (using imghdr)
            file_type = imghdr.what(image_path)
            if file_type not in ["jpeg", "png", "jpg", "bmp", "webp"]:
                return {"error": f"Unsupported image type: {file_type or 'unknown'}"}
            # Check file size (limit to 10MB)
            if os.path.getsize(image_path) > 10 * 1024 * 1024:
                return {"error": "Image file is too large (max 10MB)"}
            # Load and preprocess image (with error handling)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    return {"error": "Could not load image (file may be corrupt)"}
            except Exception as e:
                return {"error": f"Error loading image: {str(e)}"}
            # Resize image if too large (max 640x480)
            height, width = image.shape[:2]
            max_dim = 640
            if max(height, width) > max_dim:
                scale = max_dim / float(max(height, width))
                new_size = (int(width * scale), int(height * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # Analyze pose and body landmarks
            pose_results = self.pose.process(image_rgb)
            
            if not pose_results.pose_landmarks:
                return {"error": "No body pose detected in image"}
            
            # Extract body measurements and features
            body_measurements = self._extract_body_measurements(pose_results, width, height)
            
            # Analyze body composition
            body_composition = self._analyze_body_composition(image, pose_results)
            
            # Classify body type
            body_type = self._classify_body_type(body_measurements, body_composition)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(body_type, body_composition)
            
            return {
                "body_type": body_type,
                "muscle_percentage": body_composition["muscle_percentage"],
                "fat_percentage": body_composition["fat_percentage"],
                "bmi_estimate": body_composition["bmi_estimate"],
                "body_measurements": body_measurements,
                "recommendations": recommendations,
                "analysis_confidence": body_composition["confidence"],
                "landmarks_detected": len(pose_results.pose_landmarks.landmark) if pose_results.pose_landmarks else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _extract_body_measurements(self, pose_results, width: int, height: int) -> Dict:
        """Extract body measurements from pose landmarks."""
        landmarks = pose_results.pose_landmarks.landmark
        
        # Get key body points
        measurements = {}
        
        # Shoulder width
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_width = abs(right_shoulder.x - left_shoulder.x) * width
        
        # Hip width
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        hip_width = abs(right_hip.x - left_hip.x) * width
        
        # Torso length
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        torso_length = abs(nose.y - left_hip.y) * height
        
        # Arm length
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        arm_length = abs(left_shoulder.y - left_wrist.y) * height
        
        measurements = {
            "shoulder_width": shoulder_width,
            "hip_width": hip_width,
            "torso_length": torso_length,
            "arm_length": arm_length,
            "shoulder_to_hip_ratio": shoulder_width / hip_width if hip_width > 0 else 0
        }
        
        return measurements
    
    def _analyze_body_composition(self, image: np.ndarray, pose_results) -> Dict:
        """Analyze body composition using image processing techniques."""
        # Create segmentation mask
        segmentation_mask = pose_results.segmentation_mask
        
        if segmentation_mask is None:
            return {
                "muscle_percentage": 25.0,
                "fat_percentage": 15.0,
                "bmi_estimate": 22.0,
                "confidence": 0.5
            }
        
        # Convert mask to binary
        mask = (segmentation_mask > 0.5).astype(np.uint8)
        
        # Apply mask to original image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Analyze skin tone and texture for body composition estimation
        # This is a simplified approach - in production, use more sophisticated ML models
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
        
        # Calculate average values in masked region
        valid_pixels = masked_image[mask > 0]
        if len(valid_pixels) == 0:
            return {
                "muscle_percentage": 25.0,
                "fat_percentage": 15.0,
                "bmi_estimate": 22.0,
                "confidence": 0.3
            }
        
        # Analyze color distribution for body composition hints
        avg_b = np.mean(valid_pixels[:, 0])
        avg_g = np.mean(valid_pixels[:, 1])
        avg_r = np.mean(valid_pixels[:, 2])
        
        # Simple heuristics for body composition estimation
        # These would be replaced with trained ML models in production
        
        # Estimate muscle percentage based on color and texture
        muscle_percentage = self._estimate_muscle_percentage(avg_r, avg_g, avg_b, mask)
        
        # Estimate fat percentage
        fat_percentage = self._estimate_fat_percentage(avg_r, avg_g, avg_b, mask)
        
        # Estimate BMI (simplified)
        bmi_estimate = self._estimate_bmi(muscle_percentage, fat_percentage)
        
        # Calculate confidence based on mask quality
        confidence = np.mean(mask) if mask.size > 0 else 0.3
        
        return {
            "muscle_percentage": round(muscle_percentage, 1),
            "fat_percentage": round(fat_percentage, 1),
            "bmi_estimate": round(bmi_estimate, 1),
            "confidence": round(confidence, 2)
        }
    
    def _estimate_muscle_percentage(self, r: float, g: float, b: float, mask: np.ndarray) -> float:
        """Estimate muscle percentage based on color analysis."""
        # Simplified muscle estimation based on color values
        # In production, this would use a trained model
        
        # Muscle typically has more red and less blue
        muscle_score = (r - b) / 255.0
        
        # Base muscle percentage with color adjustment
        base_muscle = 25.0
        muscle_adjustment = muscle_score * 20.0
        
        muscle_percentage = base_muscle + muscle_adjustment
        
        # Clamp to reasonable range
        return max(15.0, min(45.0, muscle_percentage))
    
    def _estimate_fat_percentage(self, r: float, g: float, b: float, mask: np.ndarray) -> float:
        """Estimate fat percentage based on color analysis."""
        # Simplified fat estimation
        # Fat typically has more yellow/white tones
        
        # Calculate brightness
        brightness = (r + g + b) / 3.0
        
        # Fat tends to be lighter
        fat_score = brightness / 255.0
        
        # Base fat percentage with brightness adjustment
        base_fat = 15.0
        fat_adjustment = fat_score * 15.0
        
        fat_percentage = base_fat + fat_adjustment
        
        # Clamp to reasonable range
        return max(8.0, min(35.0, fat_percentage))
    
    def _estimate_bmi(self, muscle_percentage: float, fat_percentage: float) -> float:
        """Estimate BMI based on body composition."""
        # Simplified BMI estimation
        # In production, this would use actual height/weight data
        
        # Higher muscle percentage typically correlates with higher BMI
        muscle_factor = muscle_percentage / 100.0
        
        # Higher fat percentage also correlates with higher BMI
        fat_factor = fat_percentage / 100.0
        
        # Base BMI with adjustments
        base_bmi = 22.0
        bmi_adjustment = (muscle_factor + fat_factor) * 8.0
        
        bmi = base_bmi + bmi_adjustment
        
        # Clamp to reasonable range
        return max(18.0, min(30.0, bmi))
    
    def _classify_body_type(self, measurements: Dict, composition: Dict) -> str:
        """Classify body type based on measurements and composition."""
        shoulder_to_hip_ratio = measurements.get("shoulder_to_hip_ratio", 1.0)
        muscle_percentage = composition.get("muscle_percentage", 25.0)
        fat_percentage = composition.get("fat_percentage", 15.0)
        
        # Body type classification logic
        if shoulder_to_hip_ratio > 1.1 and muscle_percentage > 30:
            return "mesomorph"
        elif shoulder_to_hip_ratio < 0.9 and fat_percentage > 20:
            return "endomorph"
        else:
            return "ectomorph"
    
    def _generate_recommendations(self, body_type: str, composition: Dict) -> Dict:
        """Generate personalized recommendations based on body type and composition."""
        muscle_percentage = composition.get("muscle_percentage", 25.0)
        fat_percentage = composition.get("fat_percentage", 15.0)
        
        recommendations = {
            "workout_focus": [],
            "nutrition_focus": [],
            "training_frequency": "",
            "key_exercises": [],
            "body_composition_goals": []
        }
        
        if body_type == "ectomorph":
            recommendations["workout_focus"] = ["Strength training", "Compound movements", "Progressive overload"]
            recommendations["nutrition_focus"] = ["Calorie surplus", "High protein", "Complex carbohydrates"]
            recommendations["training_frequency"] = "3-4 times per week"
            recommendations["key_exercises"] = ["Squats", "Deadlifts", "Bench press", "Rows"]
            recommendations["body_composition_goals"] = ["Build muscle mass", "Increase strength"]
            
        elif body_type == "mesomorph":
            recommendations["workout_focus"] = ["Balanced training", "Hypertrophy focus", "Moderate cardio"]
            recommendations["nutrition_focus"] = ["Maintenance calories", "Moderate protein", "Balanced macros"]
            recommendations["training_frequency"] = "4-5 times per week"
            recommendations["key_exercises"] = ["Compound movements", "Isolation exercises", "HIIT"]
            recommendations["body_composition_goals"] = ["Maintain muscle", "Improve definition"]
            
        else:  # endomorph
            recommendations["workout_focus"] = ["Cardio training", "High-intensity workouts", "Strength training"]
            recommendations["nutrition_focus"] = ["Calorie deficit", "High protein", "Low refined carbs"]
            recommendations["training_frequency"] = "5-6 times per week"
            recommendations["key_exercises"] = ["Cardio", "Circuit training", "Compound movements"]
            recommendations["body_composition_goals"] = ["Reduce body fat", "Improve cardiovascular fitness"]
        
        # Add specific recommendations based on current composition
        if muscle_percentage < 20:
            recommendations["workout_focus"].append("Muscle building focus")
        if fat_percentage > 25:
            recommendations["nutrition_focus"].append("Fat loss priority")
        
        return recommendations

# Global analyzer instance
analyzer = BodyAnalyzer()

def analyze_image(image_path: str) -> Dict:
    """
    Main function to analyze body image.
    This is the interface used by the API.
    """
    return analyzer.analyze_image(image_path) 