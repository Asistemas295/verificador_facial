from flask import Flask, render_template, request, jsonify
import dlib
import cv2
import base64
import numpy as np
import os
import glob

app = Flask(__name__)

# Cargar modelos de dlib
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_model_path = "dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_model_path)

# Cargar rostros registrados en carpeta /usuarios
def cargar_encodings_usuarios():
    usuarios = {}
    for archivo in glob.glob("usuarios/*.jpg"):
        nombre = os.path.splitext(os.path.basename(archivo))[0]
        img = cv2.imread(archivo)
        if img is None:
            print(f"⚠️ No se pudo leer {archivo}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rostros = detector(img_rgb)
        if rostros:
            shape = shape_predictor(img_rgb, rostros[0])
            encoding = np.array(face_rec_model.compute_face_descriptor(img_rgb, shape))
            usuarios[nombre] = encoding
        else:
            print(f"⚠️ No se detectó rostro en: {archivo}")
    return usuarios

encodings_usuarios = cargar_encodings_usuarios()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verificar', methods=['POST'])
def verificar():
    data = request.get_json()

    if not data or 'imagen' not in data:
        return jsonify({'resultado': '❌ No se recibió imagen'})

    imagen_base64 = data['imagen']
    if ',' in imagen_base64:
        imagen_base64 = imagen_base64.split(',')[1]

    try:
        img_bytes = base64.b64decode(imagen_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'resultado': '❌ Imagen inválida'})
    except Exception as e:
        return jsonify({'resultado': f'❌ Error al decodificar imagen: {e}'})

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rostros = detector(rgb_img)

    if not rostros:
        return jsonify({'resultado': '❌ No se detectó rostro'})

    shape = shape_predictor(rgb_img, rostros[0])
    encoding = np.array(face_rec_model.compute_face_descriptor(rgb_img, shape))

    # Comparar con todos los rostros registrados
    min_dist = 1.0
    nombre_match = "Desconocido"

    for nombre, enc_reg in encodings_usuarios.items():
        dist = np.linalg.norm(enc_reg - encoding)
        print(f"{nombre}: distancia = {dist:.3f}")
        if dist < min_dist and dist < 0.6:
            min_dist = dist
            nombre_match = nombre

    if nombre_match != "Desconocido":
        return jsonify({'resultado': f'✅ Rostro reconocido: {nombre_match}'})
    else:
        return jsonify({'resultado': '❌ Rostro no reconocido'})

@app.route('/registro')
def registro():
    return render_template('registrar.html')

@app.route('/registrar', methods=['POST'])
def registrar():
    data = request.get_json()
    nombre = data.get('nombre', '').strip()
    imagen_base64 = data.get('imagen', '')

    if not nombre or not imagen_base64:
        return jsonify({'resultado': '❌ Nombre o imagen faltante'})

    if ',' in imagen_base64:
        imagen_base64 = imagen_base64.split(',')[1]

    try:
        img_bytes = base64.b64decode(imagen_base64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'resultado': '❌ Error al decodificar imagen'})

        # Guardar la imagen con el nombre del usuario
        ruta = os.path.join("usuarios", f"{nombre}.jpg")
        cv2.imwrite(ruta, img)

        # Opción: recargar encodings sin reiniciar el servidor
        encodings_usuarios[nombre] = procesar_imagen(img)

        return jsonify({'resultado': f'✅ Usuario "{nombre}" registrado correctamente'})
    except Exception as e:
        return jsonify({'resultado': f'❌ Error: {e}'})

def procesar_imagen(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rostros = detector(rgb)
    if rostros:
        shape = shape_predictor(rgb, rostros[0])
        encoding = np.array(face_rec_model.compute_face_descriptor(rgb, shape))
        return encoding
    return None



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
