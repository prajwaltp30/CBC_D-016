import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal

# --- Face detection setup (Simple Haarcascade) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

def detect_face():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:
            cap.release()
            cv2.destroyAllWindows()
            return True
        else:
            cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False

face_detected = detect_face()

if face_detected:
    print("[INFO] A face was detected. Starting heart rate monitoring...")
else:
    print("[INFO] No face detected. Exiting...")
    exit()

# --- Core EVM + HR Extraction Setup ---
def buildGauss(frame, levels):
    pyramid = [frame]
    for _ in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for _ in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

# --- Webcam and Video Parameters ---
webcam = cv2.VideoCapture(0)
realWidth = 640
realHeight = 480
videoWidth = 320
videoHeight = 240
videoChannels = 3
videoFrameRate = 15
webcam.set(3, realWidth)
webcam.set(4, realHeight)

levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (videoWidth // 2 + 5, 30)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels + 1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

bpm_data = []
time_data = []
start_time = time.time()
graph_started = False
average_bpm = None
warmup_done = False

i = 0
while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # ROI - Take Center Region (larger now)
    detectionFrame = frame[realHeight//4:realHeight*3//4, realWidth//4:realWidth*3//4, :]

    videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
    fourierTransform = np.fft.fft(videoGauss, axis=0)
    fourierTransform[mask == False] = 0

    if bufferIndex % bpmCalculationFrequency == 0:
        i += 1
        for buf in range(bufferSize):
            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
        hz = frequencies[np.argmax(fourierTransformAvg)]
        bpm = 60.0 * hz
        bpmBuffer[bpmBufferIndex] = bpm
        bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

        # Skip first 5 sec warmup
        if time.time() - start_time >= 5:
            if len(bpm_data) == 0 or bpmBuffer.mean() != bpm_data[-1]:
                bpm_data.append(bpmBuffer.mean())
                current_time = time.time() - start_time - 5  # shift after warmup
                time_data.append(current_time)
            warmup_done = True

    filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
    filtered = filtered * alpha

    filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
    outputFrame = detectionFrame + filteredFrame
    outputFrame = cv2.convertScaleAbs(outputFrame)

    bufferIndex = (bufferIndex + 1) % bufferSize

    frame[realHeight//4:realHeight*3//4, realWidth//4:realWidth*3//4, :] = outputFrame
    cv2.rectangle(frame, (realWidth//4, realHeight//4), (realWidth*3//4, realHeight*3//4), boxColor, boxWeight)

    if i > bpmBufferSize and warmup_done:
        cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, fontScale, fontColor, lineType)
    else:
        cv2.putText(frame, "Stabilizing...", loadingTextLocation, font, fontScale, fontColor, lineType)

    cv2.imshow("Webcam Heart Rate Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if warmup_done and (time.time() - start_time) >= 35 and bpm >= 40:
        average_bpm = np.mean(bpm_data)
        break

webcam.release()
cv2.destroyAllWindows()

# --- Smoothing BPM Data ---
def smooth_data(data, window_size=5):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

bpm_data_smooth = smooth_data(bpm_data, window_size=5)
time_data = time_data[-len(bpm_data_smooth):]  # adjust time_data length

# --- Metrics Calculation Functions ---
def calculate_rr_intervals(bpm_series):
    return [60.0 / bpm for bpm in bpm_series if bpm > 0]

def calculate_sdnn(rr_intervals):
    return np.std(rr_intervals) * 1000  # ms

def calculate_rmssd(rr_intervals):
    rr_diff = np.diff(rr_intervals)
    return np.sqrt(np.mean(rr_diff ** 2)) * 1000  # ms

def calculate_stress_score(rmssd, lf_hf_ratio=2.0, w1=0.5, w2=0.5):
    norm_rmssd = min(rmssd / 100, 1.0)
    norm_lfhf = min(lf_hf_ratio / 5, 1.0)
    stress = w1 * norm_lfhf + w2 * (1.0 - norm_rmssd)
    stress = max(stress, 0)  # No negative stress
    return stress * 100

def calculate_respiratory_rate(hr_series, fps=1):
    f, Pxx = signal.periodogram(hr_series, fs=fps)
    mask = (f >= 0.1) & (f <= 0.5)
    f = f[mask]
    Pxx = Pxx[mask]
    if len(f) == 0:
        return 0
    peak_freq = f[np.argmax(Pxx)]
    rr_bpm = peak_freq * 60
    return rr_bpm

# --- Calculate Metrics ---
rr_intervals = calculate_rr_intervals(bpm_data_smooth)
sdnn = calculate_sdnn(rr_intervals)
rmssd = calculate_rmssd(rr_intervals)
stress_score = calculate_stress_score(rmssd)
respiratory_rate = calculate_respiratory_rate(bpm_data_smooth)

# --- Print Results ---
print("\n--- Final Metrics ---")
print(f"Average BPM: {np.mean(bpm_data_smooth):.2f} bpm")
print(f"SDNN: {sdnn:.2f} ms")
print(f"RMSSD: {rmssd:.2f} ms")
print(f"Stress Score: {stress_score:.2f} / 100")
print(f"Respiration Rate: {respiratory_rate:.2f} bpm")

# --- Plotting with Matplotlib ---
plt.figure(figsize=(12, 10))

# Heart Rate
plt.subplot(2, 2, 1)
plt.plot(time_data, bpm_data_smooth, label="Smoothed Heart Rate (BPM)", color='blue')
plt.xlabel("Time (s)")
plt.ylabel("BPM")
plt.title("Heart Rate Over Time")
plt.grid(True)

# RR Intervals
plt.subplot(2, 2, 2)
rr_time = np.cumsum(rr_intervals)
plt.plot(rr_time, rr_intervals, label="RR Intervals", color='green')
plt.xlabel("Time (s)")
plt.ylabel("RR Interval (s)")
plt.title(f"HRV (SDNN: {sdnn:.2f} ms, RMSSD: {rmssd:.2f} ms)")
plt.grid(True)

# Stress Score
plt.subplot(2, 2, 3)
stress_series = [calculate_stress_score(calculate_rmssd(calculate_rr_intervals(bpm_data_smooth[max(0, i-10):i+1]))) for i in range(len(bpm_data_smooth))]
plt.plot(time_data, stress_series, label="Stress Score", color='red')
plt.xlabel("Time (s)")
plt.ylabel("Stress (0–100)")
plt.title(f"Stress Score Over Time")
plt.grid(True)

# Respiratory Rate
plt.subplot(2, 2, 4)
plt.plot(time_data, [respiratory_rate] * len(time_data), label="Respiration Rate", color='purple')
plt.xlabel("Time (s)")
plt.ylabel("Breaths per Minute (BPM)")
plt.title(f"Respiratory Rate: {respiratory_rate:.2f} BPM")
plt.grid(True)

plt.tight_layout()
plt.savefig("static/final_graph.png")
plt.show()
