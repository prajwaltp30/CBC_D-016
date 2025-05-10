import cv2
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Important: Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
from scipy import stats

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Camera settings
realWidth = 640
realHeight = 480
videoWidth = 320
videoHeight = 240
videoChannels = 3
videoFrameRate = 15

# Processing parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 10

# Data buffers
bpm_data = []
time_data = []
rr_intervals = []
hrv_data = []
stress_data = []
spo2_data = []

# Global variables for latest values (for live graph)
latest_values = {
    "bpm": 0,
    "hrv": 0,
    "stress": 0,
    "spo2": 0,
    "avg_bpm": 0,
    "avg_hrv": 0,
    "avg_stress": 0,
    "avg_spo2": 0,
    "scan_completed": False
}

# Flags
graph_started = False
start_time = time.time()

# Initialize buffers
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))

def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

firstGauss = buildGauss(firstFrame, levels + 1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))
frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)
bpmBuffer = np.zeros((bpmBufferSize))

# 💥 New: Function to reset everything before every new scan
def reset_scan_data():
    global bpm_data, time_data, rr_intervals, hrv_data, stress_data, spo2_data
    global latest_values, bufferIndex, bpmBufferIndex, graph_started, start_time

    bpm_data.clear()
    time_data.clear()
    rr_intervals.clear()
    hrv_data.clear()
    stress_data.clear()
    spo2_data.clear()

    latest_values.update({
        "bpm": 0,
        "hrv": 0,
        "stress": 0,
        "spo2": 0,
        "avg_bpm": 0,
        "avg_hrv": 0,
        "avg_stress": 0,
        "avg_spo2": 0,
        "scan_completed": False
    })

    bufferIndex = 0
    bpmBufferIndex = 0
    graph_started = False
    start_time = time.time()

def generate_frames():
    global bufferIndex, bpmBufferIndex, graph_started, start_time

    reset_scan_data()  # Reset everything before starting new scan

    webcam = cv2.VideoCapture(0)
    webcam.set(3, realWidth)
    webcam.set(4, realHeight)

    loop_count = 0  # Add a counter to track the number of loops

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            continue

        detectionFrame = frame[videoHeight // 2:realHeight - videoHeight // 2, videoWidth // 2:realWidth - videoWidth // 2, :]

        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
        fourierTransform = np.fft.fft(videoGauss, axis=0)
        fourierTransform[mask == False] = 0

        if bufferIndex % bpmCalculationFrequency == 0:
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

            bpm_data.append(bpm)

            if bpm > 0:
                rr_interval = 60 * 1000 / bpm
                rr_intervals.append(rr_interval)
                avg_hrv = np.mean(rr_intervals[-10:]) if len(rr_intervals) >= 10 else np.mean(rr_intervals)
                hrv_data.append(avg_hrv)

                stress_score = (avg_hrv - 0.75) * 50 + (bpm - 75) * 0.1
                stress_data.append(stress_score)

                spo2 = 98 - (0.05 * (bpm - 75)) - (0.02 * (800 - avg_hrv))
                spo2 = max(min(spo2, 100), 90)
                spo2_data.append(spo2)

                latest_values["bpm"] = round(bpm, 2)
                latest_values["hrv"] = round(avg_hrv, 2)
                latest_values["stress"] = round(stress_score, 2)
                latest_values["spo2"] = round(spo2, 2)

        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha
        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)

        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        frame[videoHeight // 2:realHeight - videoHeight // 2, videoWidth // 2:realWidth - videoWidth // 2, :] = outputFrame

        bufferIndex = (bufferIndex + 1) % bufferSize

        current_time = time.time() - start_time
        time_data.append(current_time)

        if not graph_started and current_time >= 5:
            graph_started = True
            start_time = time.time()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Check if it's time to stop the scan
        loop_count += 1
        if loop_count > 500:  # Arbitrary loop limit to prevent infinite loop
            break

        if graph_started and current_time >= 20 and len(bpm_data) > 10:
            break

    webcam.release()

    latest_values["avg_bpm"] = round(np.mean(bpm_data), 2) if bpm_data else 0
    latest_values["avg_hrv"] = round(np.mean(hrv_data), 2)/10 if hrv_data else 0
    latest_values["avg_stress"] = round(np.mean(stress_data), 2)/10000 if stress_data else 0
    latest_values["avg_spo2"] = round(np.mean(spo2_data), 2) if spo2_data else 0
    latest_values["scan_completed"] = True

    save_static_final_graph()


def save_static_final_graph():
    try:
        plt.figure(figsize=(10, 9))

        time_points = list(range(len(bpm_data)))
        time_hrv = list(range(len(hrv_data)))
        time_stress = list(range(len(stress_data)))
        time_spo2 = list(range(len(spo2_data)))

        plt.subplot(221)
        plt.plot(time_points, bpm_data, marker='o', linestyle='-', color='blue')
        plt.title("Heart Rate (BPM)")
        plt.xlabel("Time")
        plt.ylabel("BPM")
        plt.grid(True)

        plt.subplot(222)

        # Ignore first 3 data points
        time_hrv_trimmed = time_hrv[3:]
        hrv_data_trimmed = hrv_data[3:]

        plt.plot(time_hrv_trimmed, hrv_data_trimmed, marker='o', linestyle='-', color='purple')
        plt.title("HRV (ms)")
        plt.xlabel("Time")
        plt.ylabel("HRV (ms)")
        plt.grid(True)


        plt.subplot(223)
        plt.plot(time_stress, stress_data, marker='o', linestyle='-', color='red')
        plt.title("Stress Level")
        plt.xlabel("Time")
        plt.ylabel("Stress")
        plt.grid(True)

        plt.subplot(224)
        plt.plot(time_spo2, spo2_data, marker='o', linestyle='-', color='green')
        plt.title("SpO₂ (%)")
        plt.xlabel("Time")
        plt.ylabel("SpO₂ %")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('static/final_graph.png')
        plt.close()
    except Exception as e:
        print(f"Error saving static graph: {e}")

# API for real-time data
def get_live_data():
    return latest_values
