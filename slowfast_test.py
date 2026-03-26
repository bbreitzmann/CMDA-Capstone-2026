#pip install torch torchvision pytorchvideo

import sys
from types import ModuleType
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as F_base
    fake_module = ModuleType("torchvision.transforms.functional_tensor")
    for attr in dir(F_base):
        setattr(fake_module, attr, getattr(F_base, attr))
    sys.modules["torchvision.transforms.functional_tensor"] = fake_module


import numpy as np
import pandas as pd
import json
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight


from pytorchvideo.models.hub import slowfast_r50
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, Normalize
from pytorchvideo.transforms import ShortSideScale, UniformTemporalSubsample


labels_dir = Path(r"C:\Users\brend\OneDrive\Capstone\CMDA-Capstone-2026\train")
telemetry_dir = Path(r"C:\Users\brend\OneDrive\Capstone\CMDA-Capstone-2026\train")
bdd_df = pd.read_csv("C:/Users/brend/Downloads/bdd_sce.csv")


json_dir_inf = r"C:\Users\brend\Downloads\kinematic_data"
video_dir_inf = r"C:\Users\brend\Downloads\extracted_videos"
meta_csv_path_inf = r"C:\Users\brend\Downloads\downloaded_videos_meta - downloaded_videos_meta.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CLASS_MAP_STR = {1: "Conflict", 2: "Bump", 3: "Hard Brake", 4: "Not an SCE"}


valid_ids = set(bdd_df['BDD_ID'].unique())
processed_data = []

print("Processing JSON files...")

for label_path in tqdm(list(labels_dir.glob("*.json")), desc="Extracting Features"):
    BDD_ID = label_path.stem 
    if BDD_ID not in valid_ids:
        continue
    telem_path = telemetry_dir / f"{BDD_ID}.json"
    if not telem_path.exists():
        continue

    event_features = {"BDD_ID": BDD_ID}

    with open(label_path, 'r') as f:
        lbl_data = json.load(f)
        attrs = lbl_data.get("attributes", {})
        event_features['weather'] = attrs.get("weather", "unknown")
        event_features['timeofday'] = attrs.get("timeofday", "unknown")
        
        frames = lbl_data.get("frames", [])
        if frames:
            objects = frames[0].get("objects", [])
            event_features['num_cars'] = sum(1 for obj in objects if obj.get("category") == "car")
            event_features['num_pedestrians'] = sum(1 for obj in objects if obj.get("category") == "pedestrian")
            event_features['num_trucks'] = sum(1 for obj in objects if obj.get("category") == "truck")
            event_features['total_objects'] = len(objects)
            
            event_features['has_vru'] = 1 if (event_features['num_pedestrians'] > 0 or 
                                              sum(1 for obj in objects if obj.get("category") in ["rider", "bicycle"]) > 0) else 0
            
            IMAGE_AREA = 1280 * 720
            max_area = 0
            drivable_area = 0
            
            for obj in objects:
                if "box2d" in obj:
                    box = obj["box2d"]
                    area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                    if area > max_area:
                        max_area = area
                elif obj.get("category") == "area/drivable" and "poly2d" in obj:
                    poly = obj["poly2d"]
                    x = np.array([p[0] for p in poly if isinstance(p[0], (int, float))])
                    y = np.array([p[1] for p in poly if isinstance(p[1], (int, float))])
                    if len(x) > 2 and len(x) == len(y):
                        drivable_area += 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            event_features['max_object_ratio'] = max_area / IMAGE_AREA
            event_features['drivable_area_ratio'] = drivable_area / IMAGE_AREA

    with open(telem_path, 'r') as f:
        tel_data = json.load(f) 
        label_ts = 0
        if frames and "timestamp" in frames[0]:
            label_ts = frames[0]["timestamp"] 
            
        start_time = tel_data.get("startTime", 0)
        event_time_unix = start_time + label_ts
        
        lookback_ms = 6000 
        lookforward_ms = 10000 
        min_time = event_time_unix - lookback_ms
        max_time = event_time_unix + lookforward_ms
        
        accel = tel_data.get("accelerometer", [])
        if accel:
            window_accel = [a for a in accel if min_time <= a['timestamp'] <= max_time]
            if window_accel:
                accel_x = np.array([a['x'] for a in window_accel])
                accel_y = np.array([a['y'] for a in window_accel])
                accel_z = np.array([a['z'] for a in window_accel])
                event_features['accel_z_min'] = np.percentile(accel_z, 1) 
                event_features['accel_z_max'] = np.percentile(accel_z, 99) 
                event_features['accel_x_std'] = np.std(accel_x) 
                accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
                event_features['max_g_force'] = np.percentile(accel_magnitude, 99)
                if len(accel_z) > 1:
                    jerk_z = np.diff(accel_z)
                    event_features['max_jerk_z'] = np.percentile(np.abs(jerk_z), 99)
                else:
                    event_features['max_jerk_z'] = 0
            
        gyro = tel_data.get("gyro", [])
        if gyro:
            window_gyro = [g for g in gyro if min_time <= g['timestamp'] <= max_time]
            if window_gyro:
                gyro_y = [g['y'] for g in window_gyro] 
                event_features['gyro_yaw_max_abs'] = np.percentile(np.abs(gyro_y), 99) 

        locations = tel_data.get("locations", [])
        if locations:
            global_speeds = np.array([loc.get("speed", 0) for loc in locations])
            if len(global_speeds) >= 3:
                event_features['global_speed_end'] = np.mean(global_speeds[-3:])
            else:
                event_features['global_speed_end'] = 0

            window_locs = [loc for loc in locations if min_time <= loc['timestamp'] <= max_time]
            if window_locs:
                speeds = np.array([loc.get("speed", 0) for loc in window_locs])
                event_features['speed_mean'] = np.mean(speeds)
                event_features['speed_std'] = np.std(speeds)
                if len(speeds) >= 3:
                    event_features['speed_start'] = np.mean(speeds[:3])
                    event_features['speed_end'] = np.mean(speeds[-3:])
                else:
                    event_features['speed_end'] = speeds[-1] if len(speeds) > 0 else 0
                    event_features['speed_start'] = speeds[0] if len(speeds) > 0 else 0
                event_features['total_speed_change'] = event_features['speed_start'] - event_features['speed_end']
                if len(speeds) > 1:
                    speed_changes = np.diff(speeds)
                    event_features['max_deceleration'] = np.percentile(speed_changes, 1)
                else:
                    event_features['max_deceleration'] = 0

    processed_data.append(event_features)

features_df = pd.DataFrame(processed_data).fillna(0)
final_model_df = features_df.merge(bdd_df[['BDD_ID', 'EVENT_TYPE', 'CONFLICT_TYPE']], on='BDD_ID', how='inner')


unique_ids = final_model_df['BDD_ID'].unique().tolist()
np.random.seed(1)
np.random.shuffle(unique_ids)
train_cutoff = int(0.70 * len(unique_ids))
val_cutoff = int(0.85 * len(unique_ids)) 

train_ids = unique_ids[:train_cutoff]
val_ids = unique_ids[train_cutoff:val_cutoff]
test_ids = unique_ids[val_cutoff:]

train_df = final_model_df[final_model_df['BDD_ID'].isin(train_ids)]
val_df = final_model_df[final_model_df['BDD_ID'].isin(val_ids)]
test_df = final_model_df[final_model_df['BDD_ID'].isin(test_ids)]

train_encoded = pd.get_dummies(train_df, columns=['weather', 'timeofday'])
val_encoded = pd.get_dummies(val_df, columns=['weather', 'timeofday']).reindex(columns=train_encoded.columns, fill_value=0)
test_encoded = pd.get_dummies(test_df, columns=['weather', 'timeofday']).reindex(columns=train_encoded.columns, fill_value=0)

drop_cols = ['BDD_ID', 'EVENT_TYPE', 'CONFLICT_TYPE'] 
X_train, y_train = train_encoded.drop(columns=drop_cols), train_encoded['EVENT_TYPE']
X_val, y_val = val_encoded.drop(columns=drop_cols), val_encoded['EVENT_TYPE']
X_test, y_test = test_encoded.drop(columns=drop_cols), test_encoded['EVENT_TYPE']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


print("\n" + "="*50)
print("TRAINING RANDOM FOREST CASCADE...")
y_train_binary = y_train.apply(lambda x: 0 if x == 4 else 1)
y_val_binary = y_val.apply(lambda x: 0 if x == 4 else 1)
rf_trigger = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled, y_train_binary)

event_mask_train = (y_train != 4)
rf_classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42).fit(X_train_scaled[event_mask_train], y_train[event_mask_train])


val_preds_binary = rf_trigger.predict(X_val_scaled)
final_predictions = np.full(len(y_val), 4)
event_indices = np.where(val_preds_binary == 1)[0]
if len(event_indices) > 0:
    final_predictions[event_indices] = rf_classifier.predict(X_val_scaled[event_indices])

print("\nPHASE 3: FULL CASCADED PIPELINE EVALUATION (RANDOM FOREST)")
print("Validation Accuracy:", accuracy_score(y_val, final_predictions))
print(classification_report(y_val, final_predictions))

print("\n" + "="*50)
print("TRAINING XGBOOST CASCADE...")
xgb_trigger = XGBClassifier(random_state=42, eval_metric='logloss').fit(X_train_scaled, y_train_binary)
le = LabelEncoder()
y_train_events_encoded = le.fit_transform(y_train[event_mask_train])
xgb_classifier = XGBClassifier(random_state=42, eval_metric='mlogloss').fit(X_train_scaled[event_mask_train], y_train_events_encoded, sample_weight=compute_sample_weight(class_weight='balanced', y=y_train_events_encoded))


val_preds_binary_xgb = xgb_trigger.predict(X_val_scaled)
final_predictions_xgb = np.full(len(y_val), 4)
event_indices_xgb = np.where(val_preds_binary_xgb == 1)[0]
if len(event_indices_xgb) > 0:
    final_predictions_xgb[event_indices_xgb] = le.inverse_transform(xgb_classifier.predict(X_val_scaled[event_indices_xgb]))

print("\nPHASE 3: FULL CASCADED PIPELINE EVALUATION (XGBOOST)")
print("Validation Accuracy:", accuracy_score(y_val, final_predictions_xgb))
print(classification_report(y_val, final_predictions_xgb))


class HybridSuperModel(nn.Module):
    def __init__(self, num_kin_features, num_classes=4):
        super().__init__()
        
        self.slowfast = slowfast_r50(pretrained=True)
        self.slowfast.blocks[6].proj = nn.Identity() 
        
        
        self.kin_net = nn.Sequential(
            nn.Linear(num_kin_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        
        self.classifier = nn.Sequential(
            nn.Linear(2304 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, video_pathway, kinematic_data):
        v_feats = self.slowfast(video_pathway)
        k_feats = self.kin_net(kinematic_data)
        return self.classifier(torch.cat((v_feats, k_feats), dim=1))


print("\n" + "="*50)
print("PHASE 4: HYBRID SLOWFAST INFERENCE (FOR EXTRACTED VIDEOS)")


model_hybrid = HybridSuperModel(num_kin_features=X_train.shape[1]).to(device).eval()

video_transform = Compose([
    Lambda(lambda x: x / 255.0),
    ShortSideScale(size=256),
    Lambda(lambda x: x.permute(1, 0, 2, 3)),
    Normalize([0.45]*3, [0.225]*3),
    Lambda(lambda x: x.permute(1, 0, 2, 3)),
    Lambda(lambda x: [UniformTemporalSubsample(8)(x), UniformTemporalSubsample(32)(x)])
])


meta_df_inf = pd.read_csv(meta_csv_path_inf)


print("\n" + "="*50)
print("PHASE 4.5: TRAINING THE HYBRID AI ON LOCAL VIDEOS")


for param in model_hybrid.slowfast.parameters():
    param.requires_grad = False
for param in model_hybrid.slowfast.blocks[6].parameters():
    param.requires_grad = True


class_weights = torch.tensor([4.0, 2.0, 2.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_hybrid.parameters()), lr=1e-4)


model_hybrid.train() 

epochs = 5
extracted_vids_train = [f for f in os.listdir(video_dir_inf) if f.endswith('.mov')]

for epoch in range(epochs):
    epoch_loss = 0
    
    for vid_file in tqdm(extracted_vids_train, desc=f"Training Epoch {epoch+1}/{epochs}", leave=False):
        bdd_id = vid_file.replace('.mov', '')

        
        if bdd_id not in final_model_df['BDD_ID'].values: continue
        row = final_model_df[final_model_df['BDD_ID'] == bdd_id].iloc[0]

        
        target_idx = int(row['EVENT_TYPE']) - 1
        target = torch.tensor([target_idx]).to(device)

        
        kin_row = final_model_df[final_model_df['BDD_ID'] == bdd_id].drop(columns=['BDD_ID', 'EVENT_TYPE', 'CONFLICT_TYPE'])
        kin_row_encoded = pd.get_dummies(kin_row, columns=['weather', 'timeofday']).reindex(columns=X_train.columns, fill_value=0)
        kin_scaled = scaler.transform(kin_row_encoded)
        kin_tensor = torch.tensor(kin_scaled, dtype=torch.float32).to(device)

        try:
            
            video = EncodedVideo.from_path(os.path.join(video_dir_inf, vid_file))
            clip = video.get_clip(start_sec=10, end_sec=18)
            frames = [f.unsqueeze(0).to(device) for f in video_transform(clip['video'])]

            
            optimizer.zero_grad()
            logits = model_hybrid(frames, kin_tensor)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        except Exception as e:
            continue
            
    print(f"Epoch {epoch+1} Completed | Total Error (Loss): {epoch_loss:.4f}")

# Save the trained weights
save_path = r"C:\Users\brend\OneDrive\Capstone\CMDA-Capstone-2026\hybrid_model_weights.pth"
torch.save(model_hybrid.state_dict(), save_path)
print(f"\nModel weights successfully saved to: {save_path}")


model_hybrid.eval()

print("\n" + "="*100)
print(f"{'VIDEO ID':<20} | {'XGBOOST PRED':<15} | {'HYBRID AI PRED':<18} | {'ACTUAL (GT)':<15} | {'WIN'}")
print("-" * 100)

extracted_vids = [f for f in os.listdir(video_dir_inf) if f.endswith('.mov')]
for vid_file in tqdm(extracted_vids, desc="Processing Individual Videos"):
    bdd_id = vid_file.replace('.mov', '')
    
    
    if bdd_id not in final_model_df['BDD_ID'].values: continue
    row = final_model_df[final_model_df['BDD_ID'] == bdd_id].iloc[0]
    
    
    kin_row = final_model_df[final_model_df['BDD_ID'] == bdd_id].drop(columns=['BDD_ID', 'EVENT_TYPE', 'CONFLICT_TYPE'])
    kin_row_encoded = pd.get_dummies(kin_row, columns=['weather', 'timeofday']).reindex(columns=X_train.columns, fill_value=0)
    kin_scaled = scaler.transform(kin_row_encoded)
    kin_tensor = torch.tensor(kin_scaled, dtype=torch.float32).to(device)
    
    
    xgb_bin = xgb_trigger.predict(kin_scaled)[0]
    if xgb_bin == 0:
        xgb_label = "Not an SCE"
    else:
        xgb_enc = xgb_classifier.predict(kin_scaled)
        xgb_label = CLASS_MAP_STR[le.inverse_transform(xgb_enc)[0]]

    
    try:
        video = EncodedVideo.from_path(os.path.join(video_dir_inf, vid_file))
        
        clip = video.get_clip(start_sec=10, end_sec=18)
        frames = [f.unsqueeze(0).to(device) for f in video_transform(clip['video'])]
        
        with torch.no_grad():
            logits = model_hybrid(frames, kin_tensor)
            ai_idx = torch.argmax(logits).item()
            
            ai_label = ["Conflict", "Bump", "Hard Brake", "Not an SCE"][ai_idx]
    except Exception as e:
        ai_label = "Error"

    
    gt_val = row['EVENT_TYPE']
    gt_label = CLASS_MAP_STR[gt_val]

    match_marker = "✓" if ai_label == gt_label else "✗"
    print(f"{bdd_id[:18]:<20} | {xgb_label:<15} | {ai_label:<18} | {gt_label:<15} | {match_marker}")

print("="*100)