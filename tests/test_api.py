from fastapi.testclient import TestClient
import io
from PIL import Image

# Import the FastAPI 'app' from your app.py file
from app import app

# Create a simulated web client to attack our API
client = TestClient(app)

def test_read_root():
    """Test if the home page health-check is alive"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Defect Detection API! Send a POST request to /predict/"}

def test_predict_image():
    """Test if the AI prediction endpoint works with a valid image"""
    
    # 1. Generate a fake 224x224 gray image purely in memory
    fake_image_bytes = io.BytesIO()
    image = Image.new('RGB', (224, 224), color='gray')
    image.save(fake_image_bytes, format='JPEG')
    fake_image_bytes.seek(0) # Reset pointer to the start of the file

    # 2. POST the fake image to the API
    response = client.post(
        "/predict/",
        files={"file": ("fake_test_image.jpg", fake_image_bytes, "image/jpeg")}
    )
    
    # 3. Assertions (The actual tests)
    # Did the server return a 200 OK success code?
    assert response.status_code == 200
    
    # Does the response contain our expected JSON keys?
    json_data = response.json()
    assert "prediction" in json_data
    assert "confidence" in json_data
    assert "filename" in json_data
    
    # Did it return the correct filename?
    assert json_data["filename"] == "fake_test_image.jpg"
def test_predict_invalid_file():
    """Test how the API handles a non-image file (e.g., a text file)"""
    fake_text_bytes = io.BytesIO(b"This is a text document, not an image.")
    
    response = client.post(
        "/predict/",
        files={"file": ("fake_doc.txt", fake_text_bytes, "text/plain")}
    )
    
    # We expect a 400 Bad Request, not a 500 Internal Server Error crash
    assert response.status_code == 400
    assert "detail" in response.json()

def test_predict_missing_file():
    """Test how the API handles a request with no file attached"""
    response = client.post("/predict/")
    
    # FastAPI automatically guards against missing files with a 422 Unprocessable Entity
    assert response.status_code == 422