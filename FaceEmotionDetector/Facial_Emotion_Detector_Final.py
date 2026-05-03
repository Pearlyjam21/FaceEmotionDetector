import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import mediapipe as mp
import cv2
import time
import torch.nn.functional as F
import os
import math
from collections import Counter
import sys # For graceful exit if webcam fails

# ==============================================================================
# Data Pre Processing and Constants 資料預處理與常數
# ==============================================================================

# DATA_DIR = "./data"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
MODEL_PATH = "./model4.2.2.pth" 
device = torch.device("cuda")

class_names = ["Angry", "Happy", "Neutral", "Sad"] 
num_class = len(class_names)
print(class_names, "number of classes:", num_class)

TARGET_DURATION = 5.0 # Seconds: How long the script should record data (0.0 for infinite loop)

class CNN(nn.Module):
  def __init__(self, num_classes=4): ########rmeove if want to trian diff dataset
    super().__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(), nn.MaxPool2d(2) #extra layer
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        # Assuming 4 convolutional blocks and a 64x64 effective input size
        nn.Linear(2048, 512), #128 x 16
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512,num_classes)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x
  
##Criterion
##re run this everytume chnage cnn code
model = CNN(num_classes=num_class).to(torch.device("cuda"))
criterion = nn.CrossEntropyLoss()
# 1. OPTIMIZER: Use Adam with Weight Decay (L2 Reg) to fight overfitting
optimizer = optim.Adam(
    model.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=1e-4  # L2 Regularization (Recommended value: 1e-4)
)

# 2. SCHEDULER: Reduce LR when Validation Loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',         # Monitor the validation loss
    factor=0.1,         # Reduce LR by 90%
    patience=10,        # Wait 10 epochs for val_loss to improve before stepping
    # verbose=True
)



# ==============================================================================
# Configuration and Global Constants
# ==============================================================================

# Webcam & Smoothing
CAM_WIDTH = 640
CAM_HEIGHT = 480
# EMA_ALPHA = 0.2 # Temporal Smoothing (0.0=none, 1.0=instant jump) <--- COMMENTED/REMOVED AS REQUESTED
ROTATION_ALPHA = 0.2 # Roll Smoothing (prevents jitter)

VALID_DETECTION_THRESHOLD = 0.50 # Min confidence for a frame to count towards the Majority Vote analysis

# --- COLOR CONSTANTS (BGR) ---
COLOR_GREEN = (0, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (50, 50, 50)


def rotate_image(img, angle):
    """Rotates an image around its center."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# ==============================================================================
# 4. Data Analysis Function
# ==============================================================================
def analyze_session_with_point_system(all_moods_data):
    """
    Performs robust post-session emotion analysis using a tiered Point System
    based on confidence levels (0-50%, 50-95%, >95%).
    """
    if not all_moods_data:
        print("\n⚠️ No emotions were processed. Analysis aborted.")
        return {"result": "N/A", "source": "No data processed."}

    # --- SETUP: ROBUSTLY EXTRACT CLASS NAMES ---
    class_names = []
    for item in all_moods_data:
        if isinstance(item, dict) and 'class_names' in item and item['class_names']:
            class_names = item['class_names']
            break
    
    if not class_names:
        class_names = ["Angry", "Happy", "Neutral", "Sad"]
        
    total_detections = len(all_moods_data)
    
    # Filter for valid data dictionaries
    valid_data_dicts = [d for d in all_moods_data if isinstance(d, dict) and 'confidence' in d and 'mood' in d]
    
    if not valid_data_dicts:
        print("\n⚠️ No valid emotion data found.")
        return {"result": "N/A", "source": "No valid data processed."}


    # --- 1. APPLY POINT SYSTEM SCORING ---
    emotion_scores = {name: 0 for name in class_names}
    emotion_counts = {name: 0 for name in class_names}
    emotion_confidences_sum = {name: 0.0 for name in class_names}

    for data in valid_data_dicts:
        mood = data['mood']
        confidence = data['confidence']
        
        # Apply the tiered scoring
        if 0.50 <= confidence < 0.95:
            points = 1
        elif confidence >= 0.95:
            points = 3
        else: # confidence < 0.50
            points = 0 
        
        emotion_scores[mood] += points
        
        # Also collect metrics for reporting
        if points > 0:
            emotion_counts[mood] += 1
            emotion_confidences_sum[mood] += confidence

    # --- 2. DETERMINE FINAL EMOTION AND SOURCE ---
    if sum(emotion_scores.values()) == 0:
        # Fallback if no frames scored above 50%
        final_emotion = "N/A"
        source = "No frame reached a 50%+ confidence threshold. Rerun with more stable lighting."
    else:
        # Find the emotion with the highest total score
        final_emotion = max(emotion_scores, key=emotion_scores.get)
        source = f"Tiered Confidence Point System: {emotion_scores[final_emotion]} total points."

    
    print("\n--- EMOTION ANALYSIS RESULTS (Tiered Point System) ---")
    print(f"Total Frames Processed: **{total_detections}**")
    print(f"\n### 🥇 Final Aggregated Emotion: **{final_emotion}**")
    print(f"Source: {source}")

    # --- 3. Distribution & Breakdown ---
    results = {'result': final_emotion, 'source': source, 'breakdown': {}}
    
    print("\n### 📊 Point & Count Distribution")
    print("| Emotion | Total Points | Frames Scored (>=50%) | Avg. Confidence |")
    print("|:--------|:-------------|:----------------------|:----------------|")
    
    sorted_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)

    for label, score in sorted_emotions:
        count = emotion_counts[label]
        avg_conf = (emotion_confidences_sum[label] / count) if count > 0 else 0.0
        
        print(f"| {label} | {score} | {count} | {avg_conf*100:.2f}% |")
        
        results['breakdown'][label] = {
            'total_points': score, 
            'scored_frames': count, 
            'avg_confidence': avg_conf
        }
        
    return results
    # ----------------------------------------------------------------------
    ## SCENARIO 2: FALLBACK TO CONFIDENCE-WEIGHTED AVERAGE (Uses All Data)
    # ----------------------------------------------------------------------
    # else:
    #     print(f"\n⚠️ No detections met the minimum confidence threshold ({threshold_pct:.0f}%).")
    #     print("💡 **FALLBACK TRIGGERED:** Using Confidence-Weighted Average from all data.")
        
    #     valid_prob_data = [
    #         d['probabilities'] 
    #         for d in valid_data_dicts 
    #         if 'probabilities' in d and len(d['probabilities']) == len(class_names)
    #     ]

    #     if not valid_prob_data:
    #         results['result'] = "N/A (No valid probability data)"
    #         results['source'] = "Analysis failed due to lack of valid probability data."
    #     else:
    #         all_probs_matrix = np.array(valid_prob_data)
    #         total_probs_sum = np.sum(all_probs_matrix, axis=0) 
    #         max_idx = np.argmax(total_probs_sum)
    #         final_emotion = class_names[max_idx] 
            
    #         total_valid_frames = len(valid_prob_data)
    #         average_support_for_winner = total_probs_sum[max_idx] / total_valid_frames
            
    #         results['result'] = final_emotion
    #         results['source'] = f"Confidence-Weighted Average across **{total_valid_frames}** frames (Avg. support: {average_support_for_winner*100:.2f}%)."
            
    #     print("---------------------------------")
    #     print(f"✅ Final Aggregated Emotion: **{results['result']}**")
    #     print(f"Source: {results['source']}")
    #     print("---------------------------------")
        
    # return results


# ==============================================================================
# 5. Main Inference Function (Returns Raw Data)
# ==============================================================================
def inference_face_based_auto_crop():

    global model, IMG_SIZE, MODEL_PATH, NUM_CLASSES, CAM_WIDTH, CAM_HEIGHT, ROTATION_ALPHA, TARGET_DURATION
    
    # Store detailed results for post-session analysis
    all_detected_frames_data = [] 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        sys.exit(1)

    # --- MODEL LOADING ---
    print(f"Loading model from: {MODEL_PATH} on device: {device}")
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device)
        state = ckpt.get("model_state", ckpt)
        class_names = ckpt.get("class_names", None)
        img_size = ckpt.get("img_size", IMG_SIZE)
    except Exception as e:
        print(f"❌ Error loading model checkpoint: {e}")
        sys.exit(1)


    if class_names is None or len(class_names) == 0:
        # Defaulting to common classes if not found in checkpoint
        class_names = ["Angry", "Happy", "Neutral", "Sad"]

    num_classes = len(class_names)
    
    # Initialize the CNN with the correct number of classes
    try:
        # NOTE: Assumes CNN class is defined elsewhere and is accessible here.
        if model is None or (hasattr(model, 'classifier') and model.classifier[4].out_features != num_classes):
             model = CNN(num_classes=num_classes)
            
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"⚠️ Warning: Model state keys mismatch. Attempting partial load. Error: {e}")
        # NOTE: Assumes CNN class is defined elsewhere and is accessible here.
        model = CNN(num_classes=num_classes) # Re-initialize if the mismatch is severe
        model.load_state_dict(state, strict=False)
        
    model.to(device)
    model.eval()

    # --- MEDIAPIPE INITIALIZATION (Face Mesh for 468 landmarks and PnP) ---
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # --- IMAGE PREPROCESSING ---
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    print(f"✅ Webcam starting... Press Q to quit. (Classes: {class_names})")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Webcam not detected. Check device index or permissions.")
        return []
    
    # Set video capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # Variables for rotational smoothing, EMA, and FPS
    last_angle = 0.0
    pTime = 0 # Previous time for FPS calculation
    loop_start_time = time.time() # For time-out logic
    
    # --- Head Pose Estimation (PnP) Setup ---
    # 3D model points (based on MediaPipe landmarks)
    model_points = np.array([
        (0.0, 0.0, 0.0),       # Nose tip (1)
        (0.0, -330.0, -65.0), # Chin (199)
        (-225.0, 170.0, -135.0), # Left eye inner corner (33)
        (225.0, 170.0, -135.0),  # Right eye inner corner (263)
        (-150.0, -150.0, -125.0), # Left mouth corner (61)
        (150.0, -150.0, -125.0)  # Right mouth corner (291)
    ], dtype=np.float64)
    pnp_landmarks = [1, 199, 33, 263, 61, 291]
    
    # Camera Intrinsic Matrix 
    focal_length = CAM_WIDTH * 1.2 
    center = (CAM_WIDTH/2, CAM_HEIGHT/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1)) 

    try:
        while True:
            
            elapsed_time = time.time() - loop_start_time
            
            # --- TIME-OUT LOGIC ---
            if TARGET_DURATION > 0 and elapsed_time >= TARGET_DURATION:
                print(f"✅ Inference stopped automatically after {TARGET_DURATION} seconds.")
                break
            
            ok, frame = cap.read()
            if not ok:
                break
            
            # Timer display
            timer_text = f"Recording: {elapsed_time:.1f}s / {TARGET_DURATION:.1f}s" if TARGET_DURATION > 0 else f"Recording: {elapsed_time:.1f}s"
            cv2.putText(frame, timer_text, (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ORANGE, 2)

            H, W, _ = frame.shape
            
            # --- MEDIAPIPE DETECTION ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            # Initialize label and confidence variables in case no face is detected
            label = "Detecting..."
            c = 0.0
            current_probs = np.zeros(num_classes)

            pose_text = "N/A"

            if not results.multi_face_landmarks:
                cv2.putText(frame, "No face", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_RED, 2)
            else:
                face_landmarks = results.multi_face_landmarks[0]
                
                # --- 1. Calculate Bounding Box and Keypoints from Face Mesh ---
                re_idx = 33 # Right eye inner corner (image left)
                le_idx = 263 # Left eye inner corner (image right)
                
                x_coords, y_coords, image_points = [], [], []
                re_x_frame, re_y_frame, le_x_frame, le_y_frame = 0, 0, 0, 0
                
                for idx, landmark in enumerate(face_landmarks.landmark):
                    px, py = int(landmark.x * W), int(landmark.y * H)
                    x_coords.append(px)
                    y_coords.append(py)

                    if idx == re_idx: re_x_frame, re_y_frame = px, py
                    if idx == le_idx: le_x_frame, le_y_frame = px, py
                    if idx in pnp_landmarks: image_points.append((px, py))

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                w, h = x_max - x_min, y_max - y_min
                
                if w <= 0 or h <= 0: continue
                
                # 2. Face Cropping with Padding
                pad = int(0.3 * max(w,h)) # <-- INCREASED PADDING TO 30%
                x1 = max(0, x_min - pad); y1 = max(0, y_min - pad)
                x2 = min(W, x_max + pad); y2 = min(H, y_max + pad)
                face = frame[y1:y2, x1:x2].copy()
                
                if face.size == 0: continue

                aligned = face

                # # 3. Roll Alignment using Eye Keypoints <--- UNCOMMENTED BLOCK
                c1_crop = (re_x_frame - x1, re_y_frame - y1) 
                c2_crop = (le_x_frame - x1, le_y_frame - y1) 
                
                dx = c2_crop[0] - c1_crop[0]
                dy = c2_crop[1] - c1_crop[1]

                face_w = face.shape[1]
                min_dx = max(10, 0.12 * face_w); max_angle = 30.0; min_angle = 2.0

                # Rotation smoothing logic
                if dx > min_dx:
                    raw_angle = math.degrees(math.atan2(dy, dx))
                    if abs(raw_angle) <= max_angle:
                        smooth_angle = last_angle * (1.0 - ROTATION_ALPHA) + raw_angle * ROTATION_ALPHA
                        if abs(smooth_angle) >= min_angle:
                            aligned = rotate_image(face, -smooth_angle) 
                            last_angle = smooth_angle
                        else:
                            last_angle = last_angle * (1.0 - ROTATION_ALPHA)
                    else:
                        last_angle = last_angle * (1.0 - ROTATION_ALPHA)
                else:
                    last_angle = last_angle * (1.0 - ROTATION_ALPHA)
                
                
                # # --- 4. Head Pose Estimation (PnP) --- <-- Still commented
                # image_points = np.array(image_points, dtype="double")
                
                # if len(image_points) == 6:
                #     (success, rotation_vector, translation_vector) = cv2.solvePnP(
                #         model_points, image_points, camera_matrix, dist_coeffs, 
                #         flags=cv2.SOLVEPNP_ITERATIVE
                #     )

                #     (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
                #     # Extract Yaw, Pitch, Roll in degrees
                #     pitch, yaw, roll = cv2.decomposeProjectionMatrix(cv2.hconcat((rotation_matrix, translation_vector)))[6]
                #     yaw, pitch, roll = np.degrees(yaw[0]), np.degrees(pitch[0]), np.degrees(roll[0])
                        
                #     pose_text = f"P: {pitch:.1f}° | Y: {yaw:.1f}° | R: {roll:.1f}°"
                #     cv2.putText(frame, pose_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)


                # Prepare model input
                try:
                    pil = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)).resize((img_size, img_size))
                    inp = preprocess(pil).unsqueeze(0).to(device)
                except Exception as e:
                    print(f"Error preprocessing face: {e}")
                    continue

                # Run inference (UNCOMMENTED)
                with torch.no_grad():
                    logits = model(inp)
                    # Get RAW probabilities
                    current_probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()

                # # --- 5. Temporal Smoothing (EMA) --- (REMOVED/COMMENTED AS REQUESTED)
                
                # Determine final prediction from RAW probabilities (Using current_probs instead of smoothed_probs)
                idx_val = np.argmax(current_probs)
                label = class_names[idx_val]
                c = current_probs[idx_val] # Use the RAW confidence
                
                # Store RAW prediction data for the robust final analysis (UNCOMMENTED)
                all_detected_frames_data.append({
                    'mood': label, 
                    'confidence': c, 
                    'probabilities': current_probs, 
                    'class_names': class_names
                })

                # Annotate original frame
                color = COLOR_GREEN if c > VALID_DETECTION_THRESHOLD else COLOR_ORANGE 
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                txt = f"{label} {c*100:5.1f}% (RAW)" # Updated label
                cv2.putText(frame, txt, (x1, max(20,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                

                # --- 6. Real-Time Visualization with Probability Bars (UNCOMMENTED) ---
                disp_size = max(180, img_size * 2)
                bar_height = 15
                total_bar_area_height = len(class_names) * (bar_height + 4)
                total_height = disp_size + total_bar_area_height + 4
                
                display_canvas = np.zeros((total_height, disp_size, 3), dtype=np.uint8)
                
                cropped_disp = cv2.resize(aligned, (disp_size, disp_size))
                display_canvas[0:disp_size, 0:disp_size] = cropped_disp
                
                cv2.rectangle(display_canvas, (0, disp_size-28), (disp_size, disp_size), (0,0,0), -1)
                cv2.putText(display_canvas, txt, (6, disp_size-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                start_y = disp_size + 4
                # Now sorting based on the RAW probabilities (current_probs)
                sorted_probs = sorted(zip(class_names, current_probs), key=lambda x: x[1], reverse=True)
                
                for i, (emotion, prob) in enumerate(sorted_probs):
                    bar_color = COLOR_GREEN if emotion == label else COLOR_WHITE
                    bar_w = int(prob * (disp_size - 100)) 
                        
                    cv2.rectangle(display_canvas, (95, start_y), (disp_size - 5, start_y + bar_height), COLOR_GRAY, -1)
                    cv2.rectangle(display_canvas, (95, start_y), (95 + bar_w, start_y + bar_height), bar_color, -1)
                        
                    cv2.putText(display_canvas, emotion, (5, start_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
                    cv2.putText(display_canvas, f"{prob*100:.1f}%", (100 + bar_w, start_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

                    start_y += (bar_height + 4) 
                    
                cv2.imshow("Detection and RAW Probabilities", display_canvas)

            # --- FPS Calculation and Display ---
            cTime = time.time()
            fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
            pTime = cTime
            
            cv2.putText(frame, f"FPS: {int(fps)}", (W - 120, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ORANGE, 2)

            cv2.imshow("Face-based AutoCrop Emotion (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close() 
        print("✅ Inference stopped. Preparing final analysis...")
        return all_detected_frames_data

# ==============================================================================
# Execution 
# ==============================================================================

# 1. Run the inference and collect the raw data
raw_session_data = inference_face_based_auto_crop() 

# 2. Pass the raw data to the NEW analysis function (assuming it's defined and named correctly)
if raw_session_data:
    final_analysis = analyze_session_with_point_system(raw_session_data)
    
    print("\n\n==================================================================")
    print("                 ✅ FINAL SESSION SUMMARY ✅                      ")
    print("==================================================================")
    
    # --- Overall Summary ---
    total_frames = len(raw_session_data)
    
    print(f"\n### 📝 Session Metrics")
    print(f"* **Total Frames Recorded:** {total_frames}")
    
    # Check if breakdown data exists (it should, based on the function)
    if 'breakdown' in final_analysis:
        scored_frames_count = sum(d.get('scored_frames', 0) for d in final_analysis['breakdown'].values())
        total_points_score = sum(d.get('total_points', 0) for d in final_analysis['breakdown'].values())
        
        print(f"* **Frames Scored (>50% Confidence):** {scored_frames_count} ({scored_frames_count / total_frames * 100:.2f}% of total)")
        print(f"* **Total Points Calculated:** {total_points_score}")
        print(f"* **Model Image Size:** {IMG_SIZE}x{IMG_SIZE}") # Assuming IMG_SIZE is accessible
        print(f"* **Roll Smoothing Alpha (ROTATION_ALPHA):** {ROTATION_ALPHA}")
    
    # --- Final Result ---
    print(f"\n### 🥇 Final Aggregated Emotion")
    print(f"* **Result:** **{final_analysis.get('result', 'N/A')}**")
    print(f"* **Analysis Method:** {final_analysis.get('source', 'N/A')}")
    
    # --- Detailed Breakdown Table ---
    if 'breakdown' in final_analysis and final_analysis['breakdown']:
        
        # Sort the breakdown by total points for a clean report
        sorted_breakdown = sorted(
            final_analysis['breakdown'].items(), 
            key=lambda item: item[1].get('total_points', 0), 
            reverse=True
        )

        print("\n### 📊 Detailed Confidence-Point Breakdown")
        print("| Emotion | Total Points | Scored Frames | Avg. Conf. (Scored) |")
        print("|:--------|:-------------|:--------------|:--------------------|")
        
        for label, metrics in sorted_breakdown:
            count = metrics.get('scored_frames', 0)
            avg_conf = metrics.get('avg_confidence', 0.0)
            
            print(f"| {label} | {metrics.get('total_points', 0)} | {count} | {avg_conf*100:.2f}% |")

    print("==================================================================")
    
else:
    print("\n❌ Could not perform final analysis as no data was recorded.")
    print("Please check webcam connection and model loading.")