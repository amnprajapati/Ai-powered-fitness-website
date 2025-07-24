from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float
from sqlalchemy.orm import relationship
from backend.database import Base
import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="user")
    plans = relationship("Plan", back_populates="user")
    progress = relationship("Progress", back_populates="user")

class Plan(Base):
    __tablename__ = "plans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    workout = Column(String)
    meal = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship("User", back_populates="plans")

class Progress(Base):
    __tablename__ = "progress"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(DateTime, default=datetime.datetime.utcnow)
    weight = Column(Float)
    photo = Column(String)
    user = relationship("User", back_populates="progress")

class WorkoutSession(Base):
    __tablename__ = "workout_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(DateTime, default=datetime.datetime.utcnow)
    exercises = Column(String)  # JSON stringified list
    duration = Column(Integer)  # seconds
    calories = Column(Float)
    workout_type = Column(String)
    user = relationship("User")

class Goal(Base):
    __tablename__ = "goals"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    metric = Column(String)  # e.g., 'fat_percentage', 'bmi', 'muscle_percentage'
    target_value = Column(Float)
    direction = Column(String)  # 'increase' or 'decrease' or 'reach'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    achieved_at = Column(DateTime, nullable=True)
    active = Column(Integer, default=1)  # 1=active, 0=archived
    user = relationship("User") 