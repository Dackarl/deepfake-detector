
import streamlit as st
import cv2
import os
import tempfile
from deepface import DeepFace
import shutil
from PIL import Image

st.set_page_config(page_title="Detector de Deepfakes", layout="centered")
st.title("🧠 Detector de Deepfakes con IA")
st.write("Este prototipo analiza los primeros 100 frames de un video y detecta posibles manipulaciones faciales.")

uploaded_file = st.file_uploader("📤 Sube un video en formato .mp4", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.success("✅ Video cargado correctamente. Iniciando análisis...")

    cap = cv2.VideoCapture(video_path)
    os.makedirs("frames_sospechosos", exist_ok=True)

    frame_count = 0
    fake_signals = 0
    max_frames = 100
    sospechosos = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]

            if result['emotion']['neutral'] < 10:
                fake_signals += 1
                path = f"frames_sospechosos/frame_{frame_count}.jpg"
                cv2.imwrite(path, frame)
                sospechosos.append(path)

        except:
            fake_signals += 1

        frame_count += 1

    cap.release()

    st.subheader("📊 Resultados")
    st.write(f"Frames analizados: {frame_count}")
    st.write(f"Frames sospechosos: {fake_signals}")

    if frame_count == 0:
        st.warning("⚠️ No se pudieron analizar frames. Verifica el archivo.")
    else:
        porcentaje = fake_signals / frame_count
        if porcentaje > 0.3:
            st.error("⚠️ Posible deepfake detectado.")
        else:
            st.success("✅ El video parece auténtico.")

    if sospechosos:
        st.subheader("🖼️ Muestra de frames sospechosos")
        for ruta in sospechosos[:3]:
            img = Image.open(ruta)
            st.image(img, caption=ruta, use_column_width=True)

    shutil.rmtree("frames_sospechosos")
