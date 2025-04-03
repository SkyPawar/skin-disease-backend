import os
import cv2
import numpy as np
import base64
import random
import string
import mediapipe as mp
import tensorflow as tf
from flask_cors import CORS
from flask import Flask, request, render_template, session, jsonify
from tensorflow.keras.preprocessing import image 


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CORS(app, resources={r"/*": {"origins": "*"}})

app.secret_key = "supersecretkey"  # Required for session management

# model_path = "model/skin_disease_model.h5"
model_path = "/home/ubuntu/skin-disease-backend/model/skin_disease_model.h5"

if not os.path.exists(model_path):
    print(f"❌ Model file missing: {model_path}")
else:
    # print("✅ Model found! Loading...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model Loaded Successfully!")


# Class Labels
class_labels = ['Celulitis bacteriana', 'Impétigo bacteriano', 'Pie de atleta fúngico', 'Hongos en las uñas fúngicos', 
                'Tiña fúngica', 'Larva migrans cutánea parasitaria', 'Varicela viral', 'Culebrilla viral',
                'Eczema', 'Melanoma', 'Dermatitis atópica', 'carcinoma de células basales', 'Nevos melanocíticos',
                'Lesiones benignas similares a la queratosis', 'Imágenes de psoriasis, liquen plano y enfermedades relacionadas',
                'Queratosis seborreicas y otros tumores benignos', 'Warts Molluscum and other Viral Infections']


# Disease Solutions Dictionary
disease_solutions = {
    'Celulitis bacteriana': {
        "medicamento": "Antibióticos como cefalexina o clindamicina. Los analgésicos de venta libre, como el paracetamol o el ibuprofeno, pueden ayudar a controlar el dolor.",
        "recurso": "mantener la zona limpia, elevar la extremidad afectada, aplicar compresas tibias y tomar analgésicos según las indicaciones de su médico."
    },
    'Impétigo bacteriano': {
        "medicamento": "Antibióticos tópicos como la mupirocina: una crema antibiótica común con receta que se aplica directamente sobre las llagas dos o tres veces al día durante 5 a 10 días. Ácido fusídico: lávese bien las manos antes y después de aplicar la crema.",
        "recurso": "Lave suavemente la zona afectada con agua tibia y jabón varias veces al día y evite rascarse."
    },
    'Pie de atleta fúngico': {
        "medicamento": "Cremas antimicóticas como el clotrimazol o la terbinafina, también polvos útiles como el miconazol.",
        "recurso": "Mantenga los pies secos, use calzado transpirable y cámbiese los calcetines al menos una vez al día. Aplique un polvo medicado como Tinactin, Gold Bond o Lotrimin AF."
    },
    'Hongos en las uñas fúngicos': {
        "medicamento": "Medicamentos antimicóticos orales como terbinafina e itraconazol",
        "recurso": "Aplicar aceite de árbol de té, aceite de coco o ajo en las uñas afectadas, o remojar los pies en una mezcla de vinagre de sidra de manzana y agua."
    },
    'Tiña fúngica': {
        "medicamento": "Cremas antimicóticas como miconazol y clotrimazol",
        "recurso": "Lo primero que debe hacer es consultar con un médico. Evite compartir objetos personales y mantenga la piel seca."
    },
    'Larva migrans cutánea parasitaria': {
        "medicamento": "Tome medicamentos como Albendazol o Ivermectina.",
        "recurso": "Evite caminar descalzo sobre suelo contaminado."
    },
    'Varicela viral': {
        "medicamento": "Medicamentos antivirales como el aciclovir (para casos graves)",
        "recurso": "Use compresas húmedas y frías o báñese con agua tibia cada 3 o 4 horas durante los primeros días. Aplique loción de calamina y evite rascarse."
    },
    'Culebrilla viral': {
        "medicamento": "Medicamentos antivirales como el valaciclovir",
        "recurso": "Aplicar compresas frías y tomar analgésicos."
    },
    'Eczema': {
        "medicamento": "Tome medicamentos como antihistamínicos y corticosteroides para aliviar la picazón.",
        "recurso": "Aplique las cremas de hidrocortisona indicadas en la etiqueta, generalmente de una a cuatro veces al día durante un máximo de siete días."
    },
    'Melanoma': {
        "medicamento": "Medicamentos como Aldesleukin, Amtagvi (Lifileucel) y Atezolizumab (Tecentriq).",
        "recurso": "Evite la exposición prolongada al sol, use protector solar todo el tiempo y use ropa protectora que cubra sus brazos, piernas y cara cuando salga al exterior."
    },
    'Dermatitis atópica': {
        "medicamento": "Estos medicamentos, como el tacrolimus (Protopic) y el pimecrolimus (Elidel), se utilizan para terapia a largo plazo o de mantenimiento.",
        "recurso": "La aplicación regular de emolientes (humectantes) y baños o duchas cortos y tibios pueden ayudar a calmar la piel, pero evite los baños muy calientes o prolongados, ya que pueden resecar la piel."
    },
    'carcinoma de células basales': {
        "medicamento": "Prefiero un medicamento como Aldara (imiquimod), Cemiplimab-rwlc, Ebudex (fluorouracilo tópico)",
        "recurso": "El aceite de mirra es uno de los remedios naturales más efectivos. Además de tratar el carcinoma basocelular, su uso cutáneo es muy seguro."
    },
    'Nevos melanocíticos': {
        "medicamento": "Prefiero un medicamento como la crema de imiquimod (Aldara), la crema de 5-fluorouracilo (5-FU)",
        "recurso": "Los remedios caseros, como el jugo de limón y el aloe vera, pueden hacer que los lunares se desvanezcan, reduzcan su tamaño o eliminen por completo."
    },
    'Lesiones benignas similares a la queratosis': {
        "medicamento": "medicamentos tópicos como 5-fluorouracilo, imiquimod, diclofenaco, tazaroteno o urea, y procedimientos como crioterapia, legrado, electrocauterización o terapia láser.",
        "recurso": "Concéntrese en el cuidado suave de la piel, que incluya baños tibios, exfoliación suave, hidratación y evitar ropa ajustada y jabones agresivos."
    },
    'Imágenes de psoriasis, liquen plano y enfermedades relacionadas': {
        "medicamento": "corticosteroides tópicos, medicamentos orales como retinoides o inmunosupresores y fototerapia.",
        "recurso": "Avena: la mejor manera de utilizar avena para tratar la erupción cutánea causada por liquen plano es molerla hasta convertirla en un polvo fino (avena coloidal) en una licuadora o procesador de alimentos."
    },
    'Queratosis seborreicas y otros tumores benignos': {
        "medicamento": "tratamientos tópicos como crema de tazaroteno o solución de peróxido de hidrógeno.",
        "recurso": "No existen remedios caseros comprobados para la eliminación completa ni para el tratamiento de otros tumores benignos, por lo que es mejor consultar a un médico profesional."
    },
    'Warts Molluscum and other Viral Infections': {
        "medicamento": "Considere tratamientos tópicos como ácido salicílico, cantaridina o tretinoína.",
        "recurso": "Considere tomar aceite de árbol de té, vinagre de sidra de manzana y baños de avena coloidal."
    }
}


# Simulated disease prediction function (Replace with ML model)
def predict_disease(image_path):
    predicted_disease = random.choice(list(disease_solutions.keys()))
    solution = disease_solutions.get(predicted_disease, {"medicamento": "Unknown", "recurso": "No remedy found"})
    # print(f"Predicted Disease: {predicted_disease}, Solution: {solution}")  # Debugging
    return predicted_disease, solution


@app.route("/")
def home():
    return "Bakcend is Running..."

# @app.route("/<path:path>")
# def serve_static(path):
#     return send_from_directory(app.static_folder, path)

IMG_URL = "http://13.232.98.98:5000/"

@app.route("/upload_base64", methods=["POST"])
def upload_base64():
    try:
        data = request.json  
        image_data = data.get("image")  
       

        if not image_data:
            return jsonify({"error": "No image data received"}), 400

        # Base64 string cleanup
        image_data = image_data.replace("data:image/jpeg;base64,", "").replace("data:image/png;base64,", "")
        
        # Decode Base64 image
        image_bytes = base64.b64decode(image_data)
        
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        
        # Generate the unique file name
        file_path = "static/uploads/"+"/"+random_str+".jpg"
        # img_path = "static/uploads/" + img_file.filename

        # Stored in file
        with open(file_path, "wb") as image_file:
            image_file.write(image_bytes)
    
        # **Detecting if the image is skin-related**
        if not is_skin_image(file_path):
            os.remove(file_path)  # Remove invalid image
            return jsonify({"error": "Invalid Image! Please upload a skin-related image."}), 400

        # **Disease Prediction**
        predicted_disease, solution = predict_disease(file_path)
        # random_number = random.randint(60, 90)
        # random_number1 = random.randint(20, 30)

        return jsonify({
            "message": "Image uploaded successfully",
            "image_url": IMG_URL+file_path,
            "prediction": predicted_disease,
            "medicamento": solution["medicamento"],
            "recurso": solution["recurso"],
            # "red":random_number,
            # "green":random_number1,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.7)
mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5)


# Here the OpenCV module is defining
def is_skin_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return False  # Invalid image
    
    # Convert to RGB for Mediapipe processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect face, hands, and pose
    face_results = mp_face_detection.process(img_rgb)
    hand_results = mp_hands.process(img_rgb)
    pose_results = mp_pose.process(img_rgb)

    # **Bounding Box Check for Face**
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            bbox_area = bboxC.width * bboxC.height
            
            # **Accept if face is very small (Captured from distance)**
            if bbox_area < 0.1:  
                return True  
        
        return False  # Reject if clear full face visible  

    # Accept if hands or pose detected (Human body part check)
    human_detected = hand_results.multi_hand_landmarks or pose_results.pose_landmarks
    if human_detected:
        return True  
    
    # ✅ **Accept if any human body part is detected**
    if hand_results.multi_hand_landmarks or pose_results.pose_landmarks:
        return True 
    
    # ✅ Deep Check: Pose Landmark Filtering for ALL BODY PARTS
    if pose_results.pose_landmarks:
        # List of essential body parts: Knees, Thighs, Shoulders, Elbows, Back
        required_parts = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        
        detected_parts = [
            pose_results.pose_landmarks.landmark[i] for i in required_parts
        ]
        
        if any(part.visibility > 0.1 for part in detected_parts):  
            return True  # Accept image if any major body part is visible

    # Skin Color Detection using HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 10, 60], dtype=np.uint8)
    upper_skin = np.array([30, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(img_hsv, lower_skin, upper_skin)

    # Skin ratio calculation
    skin_ratio = np.sum(skin_mask > 0) / (img.shape[0] * img.shape[1])
    
    if skin_ratio < 0.30:  # Minimum 30% skin required
        return False

    return True  # Valid skin image
    

@app.route("/upload", methods=["POST"])
def upload():
    
    if request.method == "POST":
        img_file = request.files.get("image")
        # print("imag_file",img_file)
        
        if img_file:
            img_path = "static/uploads/" + img_file.filename
            img_file.save(img_path)
            
            # Detecting the skin related image
            if not is_skin_image(img_path):
                os.remove(img_path)    # Remove the invalid image
                return jsonify({"error": "Invalid Image! Please upload a skin-related image."}), 400
            

            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            if model:
                preds = model.predict(img_array)
                session['prediction'] = class_labels[np.argmax(preds)]  # Store prediction in session
                prediction = class_labels[np.argmax(preds)]
                prediction_probability = float(np.max(preds)) * 100
            else:
                prediction = "Unknown (Model not loaded)"
                prediction_probability = 0.0
                
            solution = disease_solutions.get(prediction, {"medicamento": "Unknown", "recurso": "No remedy found"})
            # random_number = random.randint(60, 80)
            # random_number1 = random.randint(20, 30)
            
            print("prediction:", prediction)

            response_data = {
                "prediction": prediction,
                "prediction_probability": prediction_probability,
                "survival_probability": 100 - prediction_probability,
                "image_url": IMG_URL+img_path,
                "medicamento": solution["medicamento"],
                "recurso": solution["recurso"],
                # "red":random_number,
                # "green":random_number1,
            }
            return jsonify(response_data)
            
        else:
            session.pop('prediction', None)  # Remove prediction if no image uploaded

    return jsonify({"prediction": prediction, "image_url": img_path}) 
    # return render_template("index.html", prediction=session.get('prediction', None))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
