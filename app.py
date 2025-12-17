import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import cv2
import os
import time

# Konfigurasi RTC untuk membantu koneksi webcam
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Load model custom
model = YOLO('models/bisindo.pt')

class YOLODetector(VideoProcessorBase):
    def __init__(self):
        self.conf_threshold = 0.4

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Deteksi YOLO
        results = model(img, conf=self.conf_threshold, imgsz=640)

        # Gambar bounding box dan label
        annotated_img = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

st.title("Deteksi Isyarat Bahasa Isyarat Indonesia (Bisindo) - Real-time Webcam")

# === Sidebar Pengaturan ===
st.sidebar.header("Pengaturan")

# Slider confidence threshold
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1, max_value=1.0, value=0.4, step=0.05,
    help="Nilai lebih rendah = lebih sensitif (bisa lebih banyak false positive)"
)

# Pemilihan sumber kamera
st.sidebar.subheader("Sumber Kamera")

# Deteksi kamera yang tersedia
available_cameras = []
for i in range(5):  # Cek sampai 5 device (biasanya cukup)
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW untuk Windows agar lebih stabil
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            available_cameras.append(i)
        cap.release()

if not available_cameras:
    st.error("Tidak ada kamera yang terdeteksi!")
    st.stop()

# Tampilkan pilihan kamera
camera_options = {f"Kamera {idx}": idx for idx in available_cameras}
selected_camera_name = st.sidebar.radio("Pilih kamera:", options=list(camera_options.keys()))

selected_camera_idx = camera_options[selected_camera_name]

st.sidebar.info(f"Menggunakan: **{selected_camera_name}** (device index: {selected_camera_idx})")

# Buat folder screenshots
os.makedirs("screenshots", exist_ok=True)

# === Webcam Stream dengan device index yang dipilih ===
ctx = webrtc_streamer(
    key=f"bisindo-detection-{selected_camera_idx}",  # Key unik agar reload saat ganti kamera
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=YOLODetector,
    media_stream_constraints={
        "video": {
            "deviceId": {"exact": selected_camera_idx} if selected_camera_idx is not None else True
        },
        "audio": False
    },
    async_processing=True,
)

# Update confidence threshold secara real-time
if ctx.video_processor:
    ctx.video_processor.conf_threshold = conf_threshold

# === Bagian Screenshot ===
if ctx and ctx.state.playing:
    st.info("✅ Webcam aktif. Arahkan tangan Anda ke kamera untuk deteksi isyarat Bisindo.")

    screenshot_placeholder = st.empty()

    if st.button("📸 Simpan Screenshot Hasil Deteksi"):
        if ctx.input_video_frame:
            raw_img = ctx.input_video_frame.to_ndarray(format="bgr24")
            results = model(raw_img, conf=conf_threshold, imgsz=640)
            annotated = results[0].plot()

            filename = f"screenshots/output_{int(time.time())}.png"
            cv2.imwrite(filename, annotated)
            st.success(f"Screenshot disimpan: `{filename}`")
            screenshot_placeholder.image(annotated, caption="Screenshot terbaru", use_column_width=True)
        else:
            st.warning("Frame belum tersedia. Tunggu sebentar setelah webcam aktif.")
else:
    st.warning("Klik tombol **START** pada komponen webcam di atas untuk mengaktifkan kamera.")

# === Catatan ===
st.markdown("""
### Catatan:
- Webcam diproses **di browser Anda (client-side)** untuk performa dan privasi.
- Ganti kamera melalui sidebar → aplikasi akan otomatis reload stream dengan kamera baru.
- Pastikan browser mengizinkan akses kamera.
- Jalankan dengan: `streamlit run app.py`
""")