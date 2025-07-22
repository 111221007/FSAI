import os
import cv2
import numpy as np
import threading
import time
from pathlib import Path
import warnings
import torch
import datetime
import pygame
from ttkbootstrap import Style, Label, Button
from ttkbootstrap.constants import *
from tkinter import Listbox, Canvas, SINGLE, END, messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
from ultralytics import YOLO
from accident_detection.scripts.email_sender import send_accident_email
from accident_detection.scripts.sms_sender import send_accident_sms
from fall_detection.scripts.email_sender import send_accident_email
from fall_detection.scripts.sms_sender import send_accident_sms

# ========= CONFIGURATION =========
MODEL_PATH = Path(r"C:\Users\cmpor\PycharmProjects\EdgeAI_Benchmark_Project\fall_detection\model\best.pt")
VIDEOS_DIR = Path(r"C:\Users\cmpor\PycharmProjects\EdgeAI_Benchmark_Project\fall_detection\data\input\test_videos")
SAVE_DIR = Path(r"C:\Users\cmpor\PycharmProjects\EdgeAI_Benchmark_Project\fall_detection\data\output\fall_frames")
SOUND_PATH = Path(r"C:/Users/cmpor/PycharmProjects/EdgeAI_Benchmark_Project/accident_detection/data/alarm.mp3")

CONFIDENCE_THRESHOLD = 0.5
CONSECUTIVE_FALL_FRAMES = 3
MAX_FALL_FRAMES = 5
VIDEO_CANVAS_WIDTH = 600
VIDEO_CANVAS_HEIGHT = 360

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ========= GLOBALS =========
stop_detection = False
is_muted = False
model = YOLO(str(MODEL_PATH)).to('cuda' if torch.cuda.is_available() else 'cpu')

# ========= SOUND AND ALERTS =========
def play_alarm_sound():
    if is_muted:
        return
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(SOUND_PATH))
        pygame.mixer.music.play()
    except Exception as e:
        print(f"🔔 Sound Error: {e}")

def toggle_mute(mute_button):
    global is_muted
    is_muted = not is_muted
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except:
        pass
    mute_button.config(text="Unmute 🔊" if is_muted else "Mute 🔇", bootstyle=WARNING)

def send_email_and_confirm(save_folder, on_success):
    def email_thread():
        send_accident_email(save_folder)  # Blocks until Gmail server confirms
        on_success()
    threading.Thread(target=email_thread, daemon=True).start()

def send_sms_async():
    threading.Thread(target=send_accident_sms, daemon=True).start()

# ========= UI HELPERS =========
def reset_ui(status_label, start_button, fall_label, progress_bar):
    status_label.config(text="Status: Waiting...")
    start_button.config(text="Start Detection", bootstyle=SUCCESS)
    fall_label.config(text="No Fall Detected")
    progress_bar['value'] = 0

def draw_text_box(frame, text, top_left, font_scale=0.5, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x1, y1 = top_left
    x2, y2 = x1 + tw, y1 + th
    cv2.rectangle(frame, (x1, y1), (x2, y2 + 5), (255, 255, 255), -1)
    cv2.putText(frame, text, (x1, y1 + th), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

def draw_detection_overlay(frame, label_text, timestamp_text, location_text):
    draw_text_box(frame, label_text, top_left=(20, 20))
    draw_text_box(frame, timestamp_text, top_left=(20, 45))
    draw_text_box(frame, location_text, top_left=(20, 70))

# ========= CORE DETECTION =========
def detect_falls(video_path, status_label, start_button, fall_label, progress_bar, video_canvas):
    global stop_detection
    try:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            messagebox.showerror("Error", "❌ Cannot open selected video.")
            reset_ui(status_label, start_button, fall_label, progress_bar)
            return

        video_filename = Path(video_path).stem
        save_path_folder = SAVE_DIR / "SelectedVideo" / video_filename
        save_path_folder.mkdir(parents=True, exist_ok=True)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = 1000 / fps

        status_label.config(text="Status: Running 🚀")
        fall_label.config(text="No Fall Detected")

        streak = 0
        frame_count = 0
        fall_frames = 0
        email_sent = False
        fall_detected_time = None
        frame_capture_time = None
        inference_latency_ms = None

        while cap.isOpened() and not stop_detection:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_capture_time = time.time()

            # Inference latency measurement
            start_infer = time.time()
            results = model.predict(frame, imgsz=640, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
            end_infer = time.time()
            inference_latency_ms = (end_infer - start_infer) * 1000
            print(f"Inference Latency: {inference_latency_ms:.2f} ms")

            boxes = results.boxes
            fall_detected = any(model.names[int(box.cls[0])] == "down" and float(box.conf[0]) > CONFIDENCE_THRESHOLD for box in boxes)

            if fall_detected and fall_detected_time is None:
                fall_detected_time = time.time()
                fall_detected_frame_capture_time = frame_capture_time  # For end-to-end latency

            if fall_detected:
                streak += 1
            else:
                streak = 0

            if streak >= CONSECUTIVE_FALL_FRAMES and fall_frames < MAX_FALL_FRAMES:
                fall_frames += 1
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                location = "Fall Detected @ lat:24.9451, long:121.3452"
                draw_detection_overlay(frame, "Fall Detected!", timestamp, location)

                save_path = save_path_folder / f"fall_frame_{fall_frames:04d}.jpg"
                cv2.imwrite(str(save_path), frame)
                fall_label.config(text="Fall Detected!")
                play_alarm_sound()

                if not email_sent:
                    def on_gmail_success():
                        alert_confirmed_time = time.time()
                        fall_to_alert_latency_ms = (alert_confirmed_time - fall_detected_time) * 1000
                        end_to_end_latency_ms = (alert_confirmed_time - fall_detected_frame_capture_time) * 1000
                        print(f"Latency from fall detection to alert confirmation: {fall_to_alert_latency_ms:.2f} ms")
                        print(f"End-to-end latency (frame capture to alert confirmation): {end_to_end_latency_ms:.2f} ms")
                        print(f"Inference latency (last fall frame): {inference_latency_ms:.2f} ms")
                        print("Fall alert email sent and confirmed by server!")
                    send_email_and_confirm(save_path_folder, on_gmail_success)
                    email_sent = True

                streak = 0

            color = (0, 0, 255) if fall_detected else (0, 255, 0)
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, h), color, 8)
            cv2.putText(frame, "Fall Detected!" if fall_detected else "No Fall",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb).resize((VIDEO_CANVAS_WIDTH, VIDEO_CANVAS_HEIGHT))
            imgtk = ImageTk.PhotoImage(image=img_pil)
            video_canvas.create_image(0, 0, anchor='nw', image=imgtk)
            video_canvas.imgtk = imgtk

            progress = (frame_count / total_frames) * 100
            progress_bar['value'] = progress
            video_canvas.update()

            sleep = delay - ((time.time() - start_time) * 1000)
            if sleep > 0:
                video_canvas.after(int(sleep))

        cap.release()
    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        reset_ui(status_label, start_button, fall_label, progress_bar)

# ========= GUI WRAPPERS =========
def start_detection(video, status, start, label, bar, canvas):
    global stop_detection
    stop_detection = False
    if video:
        threading.Thread(target=detect_falls, args=(video, status, start, label, bar, canvas), daemon=True).start()

def stop_detection_now():
    global stop_detection
    stop_detection = True

def select_video_and_start(listbox, status, start, label, bar, canvas):
    selected = listbox.curselection()
    if not selected:
        messagebox.showwarning("Warning", "Select a video first.")
        return
    video = VIDEOS_DIR / listbox.get(selected[0])
    if start['text'] == "Start Detection":
        start.config(text="Stop Detection", bootstyle=DANGER)
        status.config(text="Status: Starting 🚀")
        label.config(text="No Fall Detected")
        bar['value'] = 0
        start_detection(video, status, start, label, bar, canvas)
    else:
        stop_detection_now()

# ========= MAIN GUI =========
def main_gui():
    app = Style(theme='superhero').master
    app.title("FallSenseAI (Fall Detection System)")
    app.geometry("800x1000")

    Label(app, text="Select a Fall Video:", font=("Helvetica", 16)).pack(pady=10)
    listbox = Listbox(app, selectmode=SINGLE, font=("Helvetica", 14), width=70)
    listbox.pack(pady=10)
    for file in os.listdir(VIDEOS_DIR):
        if file.endswith((".mp4", ".avi", ".mov")):
            listbox.insert(END, file)

    start_button = Button(app, text="Start Detection", bootstyle=SUCCESS)
    mute_button = Button(app, text="Mute 🔇", bootstyle=WARNING)
    video_canvas = Canvas(app, width=VIDEO_CANVAS_WIDTH, height=VIDEO_CANVAS_HEIGHT, bg="black")
    progress_bar = Progressbar(app, length=500, mode='determinate')
    status_label = Label(app, text="Status: Waiting...", font=("Helvetica", 14))
    fall_label = Label(app, text="No Fall Detected", font=("Helvetica", 14))

    start_button.pack(pady=10)
    mute_button.pack(pady=5)
    video_canvas.pack(pady=10)
    progress_bar.pack(pady=10)
    status_label.pack(pady=10)
    fall_label.pack(pady=5)

    start_button.config(command=lambda: select_video_and_start(listbox, status_label, start_button, fall_label, progress_bar, video_canvas))
    mute_button.config(command=lambda: toggle_mute(mute_button))

    app.mainloop()

if __name__ == "__main__":
    main_gui()
