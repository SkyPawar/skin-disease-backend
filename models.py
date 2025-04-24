from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255))
    predicted_disease = db.Column(db.String(100))
    confidence = db.Column(db.String(20))
    medicamento = db.Column(db.String(255))
    recurso = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
