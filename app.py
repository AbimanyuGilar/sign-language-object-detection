import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import cv2
import os
import time

# Konfigurasi STUN server
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Load model
model = YOLO('models/bisindo.pt')

class YOLODetector(VideoProcessorBase):
    def __init__(self):
        self.conf_threshold = 0.4

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=self.conf_threshold, imgsz=640)
        annotated_img = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# === UI ===
st.title("Deteksi Isyarat Bahasa Isyarat Indonesia (Bisindo) - Real-time Webcam")

st.sidebar.header("Pengaturan")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.4, 0.05,
    help="Nilai lebih rendah = lebih banyak deteksi (bisa false positive)"
)

# Info penting di sidebar
st.sidebar.markdown("### 📷 Ganti Kamera")
st.sidebar.info(
    "Setelah klik **START** dan izinkan akses kamera:\n\n"
    "→ Cari tombol **'🎥'** atau **'Select device'** di pojok kiri bawah video\n\n"
    "→ Klik tombol itu untuk pilih/ganti kamera (built-in, eksternal, front/back di HP)"
)

os.makedirs("screenshots", exist_ok=True)

# === Webcam dengan tombol ganti kamera built-in ===
ctx = webrtc_streamer(
    key="bisindo-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=YOLODetector,
    media_stream_constraints={"video": True, "audio": False},  # JANGAN pakai deviceId exact!
    async_processing=True,
)

# Update threshold real-time
if ctx.video_processor:
    ctx.video_processor.conf_threshold = conf_threshold

# === Status & Screenshot ===
if ctx and ctx.state.playing:
    st.success("✅ Webcam aktif! Gerakkan tangan untuk deteksi isyarat Bisindo.")
    st.info("💡 Untuk ganti kamera: klik ikon 🎥 di pojok kiri bawah video")

    placeholder = st.empty()
    if st.button("📸 Simpan Screenshot Hasil Deteksi"):
        if ctx.input_video_frame:
            img = ctx.input_video_frame.to_ndarray(format="bgr24")
            results = model(img, conf=conf_threshold, imgsz=640)
            annotated = results[0].plot()

            filename = f"screenshots/output_{int(time.time())}.png"
            cv2.imwrite(filename, annotated)
            st.success(f"Disimpan: {filename}")
            placeholder.image(annotated, caption="Screenshot terbaru", use_column_width=True)
else:
    st.warning("Klik tombol **START** di atas, lalu izinkan akses kamera di popup browser.")

st.caption("Catatan: Deteksi berjalan real-time di browser Anda (privat & cepat)")