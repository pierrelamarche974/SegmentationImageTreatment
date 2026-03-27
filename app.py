import os
import numpy as np
import streamlit as st
from PIL import Image
import cityscapes_utils

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

# Taille d'entrée attendue par le modèle (hauteur, largeur)
IMAGE_SIZE  = (256, 512)
# Chemin vers le modele TFLite
TFLITE_MODEL = os.path.join(os.path.dirname(__file__), 'fpn_full.tflite')
# Noms et palette couleurs des 8 classes Cityscapes
CLASS_NAMES = cityscapes_utils.CityscapeUtils.MAIN_CLASSES
PALETTE     = cityscapes_utils.CityscapeUtils.PALETTE_8

@st.cache_resource
def load_interpreter():
    # Charge une seule fois au demarrage
    if not os.path.exists(TFLITE_MODEL):
        raise FileNotFoundError(f"Missing TFLite model: {TFLITE_MODEL}")
    interpreter = Interpreter(model_path=TFLITE_MODEL)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def preprocess(img: np.ndarray) -> np.ndarray:
    # Reproduit exactement le preprocessing VGG16 utilisé à l'entraînement :
    # soustraction des moyennes ImageNet canal par canal, puis RGB → BGR
    x = img.astype(np.float32)
    x[..., 0] -= 103.939  # B
    x[..., 1] -= 116.779  # G
    x[..., 2] -= 123.68   # R
    return x[..., ::-1]   # RGB → BGR

def _prepare_input(x: np.ndarray, input_details) -> np.ndarray:
    dtype = input_details[0]["dtype"]
    if dtype == np.uint8:
        scale, zero_point = input_details[0]["quantization"]
        if scale:
            x = x / scale + zero_point
        return np.clip(x, 0, 255).astype(np.uint8)
    return x.astype(dtype)

def predict(interpreter, input_details, output_details, img_rgb: np.ndarray) -> np.ndarray:
    H, W = IMAGE_SIZE
    # Redimensionne l'image a la taille du modele
    resized = np.array(Image.fromarray(img_rgb).resize((W, H)))
    # Ajoute la dimension batch (1, H, W, 3) et applique le preprocessing
    x = preprocess(resized)[np.newaxis]
    x = _prepare_input(x, input_details)
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]["index"])
    # argmax sur les 8 classes -> masque d'entiers (H, W) entre 0 et 7
    return np.argmax(logits[0], axis=-1).astype(np.uint8)

def blend(img_rgb: np.ndarray, mask: np.ndarray, alpha=0.45) -> np.ndarray:
    H, W = IMAGE_SIZE
    base = np.array(Image.fromarray(img_rgb).resize((W, H))).astype(np.float32)
    # Mélange linéaire : (1-alpha) * image + alpha * masque colorisé
    return ((1 - alpha) * base + alpha * PALETTE[mask]).clip(0, 255).astype(np.uint8)

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='FPN Segmentation', layout='wide')
st.title('Segmentation sémantique — FPN VGG16')
st.caption('Cityscapes · 8 classes · 256×512')

interpreter, input_details, output_details = load_interpreter()
uploaded = st.file_uploader('Charger une image (JPG / PNG)', type=['jpg', 'jpeg', 'png'])

if uploaded:
    img_rgb = np.array(Image.open(uploaded).convert('RGB'))
    H, W    = IMAGE_SIZE

    with st.spinner('Inférence…'):
        mask = predict(interpreter, input_details, output_details, img_rgb)

    # Prépare les 3 visuels
    img_disp = np.array(Image.fromarray(img_rgb).resize((W, H)))
    mask_rgb = cityscapes_utils.CityscapeUtils.colorize_mask_8(mask)  # masque colorisé
    overlay  = blend(img_rgb, mask)                                    # image + masque fusionnés

    # Affichage côte à côte
    c1, c2, c3 = st.columns(3)
    c1.image(img_disp, caption='Image',        use_container_width=True)
    c2.image(mask_rgb, caption='Masque prédit', use_container_width=True)
    c3.image(overlay,  caption='Overlay',       use_container_width=True)


else:
    st.info('Chargez une image pour lancer la segmentation.')
