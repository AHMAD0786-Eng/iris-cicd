import joblib
from sklearn.datasets import load_iris
from app import app

def test_model_accuracy():
    model = joblib.load("model.pkl")
    X, y = load_iris(return_X_y=True)
    assert model.score(X, y) > 0.90          # model must be at least 90% accurate

def test_predict_endpoint():
    client = app.test_client()
    response = client.get("/predict?values=5.1,3.5,1.4,0.2")
    assert response.status_code == 200
    assert response.get_json()["prediction"] == "setosa"
