import os
import sys

# ===> 1. Ajout du chemin vers 'src' contenant utils.py et CNNModel
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ===> 2. Importation des fonctions et constantes depuis utils.py
try:
    from utils import preprocess_image_for_inference, CLASS_NAMES, CNNModel
    print("✅ utils.py importé avec succès.")
except ModuleNotFoundError as e:
    print("❌ ERREUR d'importation de utils.py:", e)
    raise

from flask import Flask, render_template, request
import torch
import tensorflow as tf
from torchvision import transforms
from PIL import Image
import numpy as np
from cnn_pytorch import CNNModel


# ===> 3. Initialisation Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===> 4. Détection du device PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===> 5. Chargement du modèle TensorFlow
try:
    tensorflow_model = tf.keras.models.load_model(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'emmanuel_model.tensorflow'))
    )
    print("✅ Modèle TensorFlow chargé avec succès.")
except Exception as e:
    print("❌ ERREUR chargement modèle TensorFlow:", e)
    raise

# ===> 6. Chargement du modèle PyTorch
try:
    pytorch_model = CNNModel(num_classes=len(CLASS_NAMES))
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'emmanuel_model.torch'))
    pytorch_model.load_state_dict(torch.load(model_path, map_location=device))
    pytorch_model.eval().to(device)
    print("✅ Modèle PyTorch chargé avec succès.")
except Exception as e:
    print("❌ ERREUR chargement modèle PyTorch:", e)
    raise

# ===> 7. Route principale
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        model_choice = request.form.get('model_choice')
        file = request.files.get('image')

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                if model_choice == 'tensorflow':
                    input_array = preprocess_image_for_inference(filepath, 'tensorflow')
                    preds = tensorflow_model.predict(input_array)
                    pred_idx = preds.argmax(axis=1)[0]
                    prediction = CLASS_NAMES[pred_idx]

                elif model_choice == 'pytorch':
                    input_tensor = preprocess_image_for_inference(filepath, 'pytorch')
                    with torch.no_grad():
                        input_tensor = input_tensor.to(device)
                        outputs = pytorch_model(input_tensor)
                        pred_idx = torch.argmax(outputs, dim=1).item()
                        prediction = CLASS_NAMES[pred_idx]
                else:
                    prediction = "Modèle non reconnu."
            except Exception as e:
                prediction = f"Erreur pendant la prédiction : {str(e)}"
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
    return render_template('index.html', prediction=prediction)

# ===> 8. Lancement de Flask
if __name__ == '__main__':
    app.run(debug=True, port=5001)
