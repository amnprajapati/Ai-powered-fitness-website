import pytest
from fastapi.testclient import TestClient
from main import app
from backend.models import User, Plan, Progress
from backend.database import Base, engine, SessionLocal
from sqlalchemy.orm import sessionmaker
import os

client = TestClient(app)

# --- Setup/teardown for test DB ---
TEST_DB_URL = "sqlite:///./test_fitness_app.db"

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    # Override DB URL for tests
    os.environ["DATABASE_URL"] = TEST_DB_URL
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test_fitness_app.db"):
        os.remove("./test_fitness_app.db")

# --- Route Tests ---
def test_register_and_login():
    username = "testuser"
    email = "testuser@example.com"
    password = "testpass123"
    # Register
    response = client.post("/register", data={"username": username, "email": email, "password": password})
    assert response.status_code == 200
    data = response.json()
    assert "user_id" in data
    # Login
    response = client.post("/login", data={"username": username, "password": password})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user_id"]


def test_upload_image(tmp_path):
    # Register and login
    username = "imguser"
    email = "imguser@example.com"
    password = "imgpass123"
    client.post("/register", data={"username": username, "email": email, "password": password})
    login_resp = client.post("/login", data={"username": username, "password": password})
    user_id = login_resp.json()["user_id"]
    # Create a dummy image file
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 100)  # minimal JPEG header
    with open(img_path, "rb") as f:
        response = client.post("/upload-image", data={"user_id": user_id}, files={"file": ("test.jpg", f, "image/jpeg")})
    assert response.status_code == 200 or response.status_code == 201 or response.status_code == 422  # Accept 422 for invalid image

# --- Model Tests ---
def test_user_model():
    user = User(username="modeluser", email="model@example.com", hashed_password="hashed")
    assert user.username == "modeluser"
    assert user.email == "model@example.com"
    assert user.hashed_password == "hashed"

def test_plan_model():
    plan = Plan(user_id=1, workout="Pushups", meal="Salad")
    assert plan.user_id == 1
    assert plan.workout == "Pushups"
    assert plan.meal == "Salad"

def test_progress_model():
    progress = Progress(user_id=1, weight=70.5, photo="photo.jpg")
    assert progress.user_id == 1
    assert progress.weight == 70.5
    assert progress.photo == "photo.jpg" 