from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Body, Path, Security, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from backend.database import SessionLocal, engine, Base
from backend.models import User, Plan, Progress, WorkoutSession, Goal
from backend.ml.analyzer import analyze_image
from backend.chatbot.bot import get_chatbot_response, chatbot
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
import shutil
import os
from typing import Optional, Union, Tuple
from dotenv import load_dotenv
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from fastapi import status
from fastapi.exception_handlers import http_exception_handler
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import json
from sqlalchemy import desc
import time

load_dotenv()
# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "type": "HTTPException"},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": exc.errors(), "type": "ValidationError"},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": str(exc), "type": "InternalServerError"},
    )

security = HTTPBearer()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Update verify_token to return both username and role
def verify_token(token: str) -> Optional[Tuple[str, str]]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        role: Optional[str] = payload.get("role")
        if username is None or role is None:
            return None
        return username, role
    except JWTError:
        return None

def get_current_user_role(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    result = verify_token(token)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    username, role = result
    return {"username": username, "role": role}

def require_role(required_role: str):
    def role_dependency(user=Depends(get_current_user_role)):
        if user["role"] != required_role:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return role_dependency

@app.get("/")
def read_root():
    return {"message": "AI Fitness App Backend Running"}

@app.post("/register")
def register(username: str = Form(...), email: str = Form(...), password: str = Form(...), role: str = Form('user'), db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Hash password
    hashed_password = pwd_context.hash(password)
    user = User(username=username, email=email, hashed_password=hashed_password, role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User registered", "user_id": user.id, "role": user.role}

@app.post("/login")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not pwd_context.verify(password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    # Include role in JWT
    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {
        "message": "Login successful", 
        "user_id": user.id,
        "role": user.role,
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.post("/upload-image")
def upload_image(user_id: int = Form(...), file: UploadFile = File(...)):
    # Validate file name
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid file name")
    
    upload_dir = "uploaded_images"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "Image uploaded", "file_path": file_path}

@app.post("/analyze")
def analyze(file_path: str = Form(...)):
    result = analyze_image(file_path)
    return result

@app.post("/plan")
def generate_plan(user_id: int = Form(...), body_type: str = Form(...), db: Session = Depends(get_db)):
    # Generate personalized workout and meal plans based on body type
    if body_type == "ectomorph":
        workout = """
        ECTOMORPH WORKOUT PLAN (3-4 times per week):
        
        Day 1 - Upper Body:
        - Bench Press: 4 sets x 8-10 reps
        - Overhead Press: 3 sets x 8-10 reps
        - Rows: 3 sets x 10-12 reps
        - Bicep Curls: 3 sets x 12 reps
        - Tricep Dips: 3 sets x 10 reps
        
        Day 2 - Lower Body:
        - Squats: 4 sets x 8-10 reps
        - Deadlifts: 3 sets x 6-8 reps
        - Lunges: 3 sets x 12 reps each leg
        - Calf Raises: 4 sets x 15 reps
        
        Day 3 - Full Body:
        - Compound movements with moderate weights
        - Focus on progressive overload
        """
        
        meal = """
        ECTOMORPH NUTRITION PLAN:
        
        Daily Calories: 2500-3000
        Protein: 1.2g per pound bodyweight
        Carbs: 3-4g per pound bodyweight
        Fats: 0.5g per pound bodyweight
        
        Sample Day:
        Breakfast: Oatmeal with protein powder, banana, nuts
        Snack: Greek yogurt with berries
        Lunch: Chicken breast, rice, vegetables
        Snack: Protein shake with peanut butter
        Dinner: Salmon, sweet potato, green vegetables
        """
        
    elif body_type == "mesomorph":
        workout = """
        MESOMORPH WORKOUT PLAN (4-5 times per week):
        
        Day 1 - Chest/Triceps:
        - Bench Press: 4 sets x 8-10 reps
        - Incline Press: 3 sets x 8-10 reps
        - Dips: 3 sets x 10-12 reps
        - Tricep Extensions: 3 sets x 12 reps
        
        Day 2 - Back/Biceps:
        - Deadlifts: 4 sets x 6-8 reps
        - Pull-ups: 3 sets x 8-10 reps
        - Rows: 3 sets x 10-12 reps
        - Bicep Curls: 3 sets x 12 reps
        
        Day 3 - Legs:
        - Squats: 4 sets x 8-10 reps
        - Leg Press: 3 sets x 10-12 reps
        - Lunges: 3 sets x 12 reps each leg
        - Calf Raises: 4 sets x 15 reps
        
        Day 4 - Shoulders/Arms:
        - Overhead Press: 4 sets x 8-10 reps
        - Lateral Raises: 3 sets x 12 reps
        - Arm isolation exercises
        """
        
        meal = """
        MESOMORPH NUTRITION PLAN:
        
        Daily Calories: 2200-2800
        Protein: 1g per pound bodyweight
        Carbs: 2-3g per pound bodyweight
        Fats: 0.4g per pound bodyweight
        
        Sample Day:
        Breakfast: Eggs, whole grain toast, avocado
        Snack: Protein bar or nuts
        Lunch: Turkey sandwich, salad
        Snack: Greek yogurt
        Dinner: Lean beef, quinoa, vegetables
        """
        
    else:  # endomorph
        workout = """
        ENDOMORPH WORKOUT PLAN (5-6 times per week):
        
        Day 1 - Cardio + Upper Body:
        - 20 min HIIT cardio
        - Push-ups: 3 sets x max reps
        - Rows: 3 sets x 12-15 reps
        - Overhead Press: 3 sets x 10-12 reps
        
        Day 2 - Cardio + Lower Body:
        - 30 min steady state cardio
        - Squats: 4 sets x 12-15 reps
        - Lunges: 3 sets x 15 reps each leg
        - Step-ups: 3 sets x 12 reps each leg
        
        Day 3 - Full Body Circuit:
        - Circuit training with minimal rest
        - 45-60 minutes total
        - High intensity, moderate weights
        """
        
        meal = """
        ENDOMORPH NUTRITION PLAN:
        
        Daily Calories: 1800-2200 (calorie deficit)
        Protein: 1.2g per pound bodyweight
        Carbs: 1-2g per pound bodyweight
        Fats: 0.3g per pound bodyweight
        
        Sample Day:
        Breakfast: Protein smoothie with berries
        Snack: Handful of almonds
        Lunch: Grilled chicken salad
        Snack: Greek yogurt
        Dinner: Baked fish, steamed vegetables
        """
    
    plan = Plan(user_id=user_id, workout=workout, meal=meal)
    db.add(plan)
    db.commit()
    db.refresh(plan)
    
    return {
        "workout": workout,
        "meal": meal,
        "body_type": body_type,
        "plan_id": plan.id
    }

@app.post("/progress")
def log_progress(user_id: int = Form(...), weight: float = Form(...), photo: Optional[UploadFile] = File(None), db: Session = Depends(get_db)):
    photo_path: Optional[str] = None
    if photo and photo.filename:
        upload_dir = "progress_photos"
        os.makedirs(upload_dir, exist_ok=True)
        photo_path = os.path.join(upload_dir, photo.filename)
        with open(photo_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)
    
    progress = Progress(user_id=user_id, weight=weight, photo=photo_path)
    db.add(progress)
    db.commit()
    db.refresh(progress)
    return {"message": "Progress logged"}

@app.post("/workout-session")
def log_workout_session(
    user_id: int = Form(...),
    exercises: str = Form(...),  # JSON stringified list
    duration: int = Form(...),   # seconds
    calories: float = Form(None),
    workout_type: str = Form(...),
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    # Estimate calories if not provided (simple heuristic)
    if calories is None:
        if "cardio" in workout_type.lower():
            calories = duration * 0.13  # ~8 cal/min
        else:
            calories = duration * 0.09  # ~5.5 cal/min
    session = WorkoutSession(
        user_id=user_id,
        exercises=exercises,
        duration=duration,
        calories=calories,
        workout_type=workout_type
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return {"message": "Workout session logged", "session_id": session.id}

@app.get("/workout-session/{user_id}")
def get_workout_sessions(user_id: int, db: Session = Depends(get_db), credentials: HTTPAuthorizationCredentials = Security(security)):
    sessions = db.query(WorkoutSession).filter(WorkoutSession.user_id == user_id).order_by(desc(WorkoutSession.date)).all()
    return sessions

# Store ML analyzer results per user (latest only for now)
@app.post("/user-analysis")
def save_user_analysis(
    user_id: int = Form(...),
    analysis: str = Form(...),  # JSON stringified dict
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    analysis_dir = "user_analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    analysis_file = os.path.join(analysis_dir, f"{user_id}.json")
    entry = json.loads(analysis)
    entry["timestamp"] = int(time.time())
    if os.path.exists(analysis_file):
        with open(analysis_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
    else:
        data = []
    data.append(entry)
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"message": "Analysis saved"}

@app.get("/user-analysis/{user_id}")
def get_user_analysis(user_id: int, credentials: HTTPAuthorizationCredentials = Security(security)):
    analysis_file = os.path.join("user_analysis", f"{user_id}.json")
    if not os.path.exists(analysis_file):
        return []
    with open(analysis_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    # Sort by timestamp ascending
    data.sort(key=lambda x: x.get("timestamp", 0))
    return data

# Update /progress and /progress/{user_id} to include latest analysis
@app.get("/progress/{user_id}")
def get_progress(user_id: int, db: Session = Depends(get_db)):
    progress = db.query(Progress).filter(Progress.user_id == user_id).all()
    # Attach latest analysis if available
    analysis_file = os.path.join("user_analysis", f"{user_id}.json")
    analysis = None
    if os.path.exists(analysis_file):
        with open(analysis_file, "r", encoding="utf-8") as f:
            analysis = json.load(f)
    return {"progress": [p.__dict__ for p in progress], "analysis": analysis}

@app.post("/chatbot")
def chatbot(message: str = Form(...), user_id: Optional[int] = Form(None), db: Session = Depends(get_db)):
    # Get user context if user_id is provided
    user_context = None
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            # Get user's latest plan to understand their body type and goals
            latest_plan = db.query(Plan).filter(Plan.user_id == user_id).order_by(Plan.created_at.desc()).first()
            if latest_plan:
                user_context = {
                    "user_id": user_id,
                    "username": user.username,
                    "body_type": "unknown"  # This would be extracted from plan analysis
                }
    
    response = get_chatbot_response(message, user_context)
    return {"response": response}

@app.post("/motivation")
def get_motivation(
    data: dict = Body(...)
):
    """
    Returns a personalized motivational quote based on user context.
    Accepts JSON: {goal, body_type, progress, mood (optional)}
    """
    goal = data.get("goal")
    body_type = data.get("body_type")
    progress = data.get("progress")
    mood = data.get("mood")

    # Predefined quotes by context
    quotes = {
        ("bulking", None): [
            "Growth doesn’t come easy. You’re fueling the beast.",
            "Eat big, lift big, get big!"
        ],
        ("cutting", None): [
            "You don’t find willpower — you forge it. Every rep counts.",
            "Shred the excuses, not just the fat."
        ],
        ("maintenance", None): [
            "Consistency is the secret sauce to lasting results.",
            "Strong today, stronger tomorrow."
        ],
        (None, "ectomorph"): [
            "Small steps, big gains. Ectomorphs can build muscle too!",
            "Your frame is your foundation. Build it strong."
        ],
        (None, "mesomorph"): [
            "You’re built for greatness. Push your limits!",
            "Athletic by nature, unstoppable by choice."
        ],
        (None, "endomorph"): [
            "Every drop of sweat is a step closer. You’ve got this!",
            "Endomorphs: turning challenge into triumph."
        ],
        ("plateau", None): [
            "Progress isn’t just numbers — it’s not giving up.",
            "Plateaus are where champions are made."
        ],
        (None, None): [
            "Your only limit is you. Show up and show out!",
            "The journey is tough, but so are you."
        ]
    }

    # Try to match the most specific quote
    selected = None
    if progress == "plateau":
        selected = quotes.get(("plateau", None))
    elif goal in ["bulking", "cutting", "maintenance"]:
        selected = quotes.get((goal, None))
    elif body_type in ["ectomorph", "mesomorph", "endomorph"]:
        selected = quotes.get((None, body_type))
    if not selected:
        selected = quotes.get((None, None))

    import random
    quote = random.choice(selected)

    # If GPT is available, try to generate a custom quote
    try:
        from backend.chatbot.bot import chatbot
        if chatbot.use_gpt and chatbot.openai_client:
            prompt = f"Generate a short, powerful motivational quote for someone with goal: {goal}, body type: {body_type}, progress: {progress}, mood: {mood}. Make it sound like a fitness coach."
            gpt_result = chatbot._get_gpt_response(prompt)
            if gpt_result:
                quote = gpt_result.strip().strip('"')
    except Exception:
        pass

    return {"quote": quote}

@app.post("/goal")
def set_goal(
    user_id: int = Form(...),
    metric: str = Form(...),
    target_value: float = Form(...),
    direction: str = Form(...),  # 'increase', 'decrease', 'reach'
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    goal = Goal(user_id=user_id, metric=metric, target_value=target_value, direction=direction)
    db.add(goal)
    db.commit()
    db.refresh(goal)
    return {"message": "Goal set", "goal_id": goal.id}

@app.get("/goal/{user_id}")
def get_goals(user_id: int, db: Session = Depends(get_db), credentials: HTTPAuthorizationCredentials = Security(security)):
    goals = db.query(Goal).filter(Goal.user_id == user_id, Goal.active == 1).all()
    return [g.__dict__ for g in goals]

@app.put("/goal/{goal_id}")
def update_goal(goal_id: int, achieved: bool = Form(None), db: Session = Depends(get_db), credentials: HTTPAuthorizationCredentials = Security(security)):
    goal = db.query(Goal).filter(Goal.id == goal_id).first()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    if achieved is not None:
        goal.achieved_at = datetime.utcnow() if achieved else None
        goal.active = 0 if achieved else 1
    db.commit()
    db.refresh(goal)
    return goal

# On new analysis, check goals and return alerts
@app.post("/user-analysis-with-goal-alerts")
def save_analysis_with_goal_alerts(
    user_id: int = Form(...),
    analysis: str = Form(...),
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    # Save analysis as before
    analysis_dir = "user_analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    analysis_file = os.path.join(analysis_dir, f"{user_id}.json")
    entry = json.loads(analysis)
    entry["timestamp"] = int(time.time())
    if os.path.exists(analysis_file):
        with open(analysis_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
    else:
        data = []
    data.append(entry)
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    # Check goals
    goals = db.query(Goal).filter(Goal.user_id == user_id, Goal.active == 1).all()
    alerts = []
    for goal in goals:
        metric = goal.metric
        direction = goal.direction
        target = goal.target_value
        values = [a.get(metric) for a in data if a.get(metric) is not None]
        if not values:
            continue
        current = values[-1]
        # Check if goal met
        met = False
        if direction == 'decrease' and current <= target:
            met = True
        elif direction == 'increase' and current >= target:
            met = True
        elif direction == 'reach' and abs(current - target) < 0.01:
            met = True
        if met:
            goal.achieved_at = datetime.utcnow()
            goal.active = 0
            db.commit()
            alerts.append({"type": "goal_met", "goal": goal.metric, "target": target, "current": current})
        else:
            # Check for stall (no progress in last 2 entries)
            if len(values) >= 3 and abs(values[-1] - values[-2]) < 0.01 and abs(values[-2] - values[-3]) < 0.01:
                alerts.append({"type": "progress_stalled", "goal": goal.metric, "target": target, "current": current})
            # AI-based suggestion: estimate date to goal
            if len(values) >= 2:
                rate = (values[-1] - values[0]) / max((data[-1]["timestamp"] - data[0]["timestamp"]) / (7*24*3600), 0.01)
                if rate != 0:
                    weeks_needed = (target - values[-1]) / rate
                    if weeks_needed > 0:
                        est_date = datetime.utcnow() + timedelta(weeks=weeks_needed)
                        alerts.append({"type": "ai_suggestion", "goal": goal.metric, "target": target, "current": current, "estimated_date": est_date.strftime('%Y-%m-%d')})
    return {"message": "Analysis saved", "alerts": alerts}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# --- Pydantic Schemas ---
class PlanUpdate(BaseModel):
    workout: str | None = None
    meal: str | None = None

class ProgressUpdate(BaseModel):
    weight: float | None = None
    photo: str | None = None  # Path or URL to photo

# --- PLAN CRUD ENDPOINTS ---
@app.get("/plan/{user_id}")
def get_plans(user_id: int, db: Session = Depends(get_db)):
    plans = db.query(Plan).filter(Plan.user_id == user_id).all()
    return plans

@app.put("/plan/{plan_id}")
def update_plan(plan_id: int, update: PlanUpdate, db: Session = Depends(get_db)):
    plan = db.query(Plan).filter(Plan.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    if update.workout is not None:
        plan.workout = update.workout
    if update.meal is not None:
        plan.meal = update.meal
    db.commit()
    db.refresh(plan)
    return plan

@app.delete("/plan/{plan_id}")
def delete_plan(plan_id: int, db: Session = Depends(get_db)):
    plan = db.query(Plan).filter(Plan.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    db.delete(plan)
    db.commit()
    return {"message": "Plan deleted"}

# --- PROGRESS CRUD ENDPOINTS ---
@app.put("/progress/{progress_id}")
def update_progress(progress_id: int, update: ProgressUpdate, db: Session = Depends(get_db)):
    progress = db.query(Progress).filter(Progress.id == progress_id).first()
    if not progress:
        raise HTTPException(status_code=404, detail="Progress not found")
    if update.weight is not None:
        progress.weight = update.weight
    if update.photo is not None:
        progress.photo = update.photo
    db.commit()
    db.refresh(progress)
    return progress

@app.delete("/progress/{progress_id}")
def delete_progress(progress_id: int, db: Session = Depends(get_db)):
    progress = db.query(Progress).filter(Progress.id == progress_id).first()
    if not progress:
        raise HTTPException(status_code=404, detail="Progress not found")
    db.delete(progress)
    db.commit()
    return {"message": "Progress deleted"}

# Example admin-only endpoint
@app.get("/admin-only")
def admin_only(user=Depends(require_role("admin"))):
    return {"message": f"Hello, {user['username']}! You are an admin."}

@app.get("/admin/events")
def admin_events(user=Depends(require_role("admin"))):
    return {"message": f"Admin events data for {user['username']}"}

@app.get("/admin/dashboard")
def admin_dashboard(user=Depends(require_role("admin"))):
    return {"message": f"Admin dashboard data for {user['username']}"}

@app.get("/admin/chatbot-sessions")
def get_chatbot_sessions(
    user=Depends(require_role("admin")),
    username: Optional[str] = Query(None, description="Filter by username (user message)"),
    source: Optional[str] = Query(None, description="Filter by response source (gpt, rule_based, etc.)"),
    start_date: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)")
):
    """
    Retrieve and filter chatbot session logs for admin review and insight generation.
    """
    log_file = os.path.join(os.path.dirname(__file__), "chatbot", "chatbot_sessions.json")
    if not os.path.exists(log_file):
        return {"sessions": [], "count": 0}
    with open(log_file, "r", encoding="utf-8") as f:
        sessions = json.load(f)
    filtered = []
    for entry in sessions:
        # Filter by username (user message)
        if username and username.lower() not in entry.get("user", "").lower():
            continue
        # Filter by source
        if source and entry.get("source") != source:
            continue
        # Filter by date range (if bot response includes a timestamp in the future, add it)
        # For now, skip date filtering unless timestamps are added
        filtered.append(entry)
    # Optionally, add date filtering if timestamps are present in the future
    return {"sessions": filtered, "count": len(filtered)}

# Placeholder routes for user, image, ML, plan, progress, chatbot
# Implementations will be added in separate modules/files 