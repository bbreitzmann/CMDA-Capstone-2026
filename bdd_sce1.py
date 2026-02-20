# Explore the csv: bdd_sce.csv which was provided by Dr. Liang Shi for initial exploration
# Initially used a random forest to evaluate situations (event type) from telemetry and label data
# Benchmark against xgboost

import numpy as np
import pandas as pd
import json
import glob
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Setup Paths
labels_dir = Path("data/100k/train")
telemetry_dir = Path("data/bddk100_info/100k/train")
bdd_df = pd.read_csv("data/bdd_sce.csv")

# Create a set of IDs from the CSV for fast lookup
valid_ids = set(bdd_df['BDD_ID'].unique())

processed_data = []

# Iterate and Extract Features
# We loop through the labels directory and look for the matching telemetry file
print("Processing JSON files...")
for label_path in tqdm(labels_dir.glob("*.json")):
    BDD_ID = label_path.stem # Gets the filename without '.json'
    
    # Only process if it exists in our target CSV
    if BDD_ID not in valid_ids:
        continue
        
    telem_path = telemetry_dir / f"{BDD_ID}.json"
    
    # Only process if BOTH JSONs exist
    if not telem_path.exists():
        continue

    # Initialize a dictionary for this specific driving event
    event_features = {"BDD_ID": BDD_ID}

    # Extract Spatial & Context Features (from Labels JSON)
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
            event_features['total_objects'] = len(objects)
            
            # ADVANCED SPATIAL: Find the largest object (closest to camera)
            max_area = 0
            for obj in objects:
                if "box2d" in obj:
                    box = obj["box2d"]
                    area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                    if area > max_area:
                        max_area = area
            event_features['closest_object_area'] = max_area

    # Extract Kinematic Features (from Telemetry JSON)
    with open(telem_path, 'r') as f:
        tel_data = json.load(f)
        
        accel = tel_data.get("accelerometer", [])
        if accel:
            accel_x = np.array([a['x'] for a in accel])
            accel_y = np.array([a['y'] for a in accel])
            accel_z = np.array([a['z'] for a in accel])
            
            # 1st and 99th percentiles to filter out 1-frame sensor noise
            event_features['accel_z_min'] = np.percentile(accel_z, 1) 
            event_features['accel_z_max'] = np.percentile(accel_z, 99) 
            event_features['accel_x_std'] = np.std(accel_x) 
            
            # Total G-Force (Magnitude)
            accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            event_features['max_g_force'] = np.percentile(accel_magnitude, 99)
            
            # 99th percentile of Jerk
            if len(accel_z) > 1:
                jerk_z = np.diff(accel_z)
                event_features['max_jerk_z'] = np.percentile(np.abs(jerk_z), 99)
            else:
                event_features['max_jerk_z'] = 0
            
        gyro = tel_data.get("gyro", [])
        if gyro:
            gyro_y = [g['y'] for g in gyro] 
            # 99th percentile of Yaw (ignoring a single phone rattle)
            event_features['gyro_yaw_max_abs'] = np.percentile(np.abs(gyro_y), 99) 

        locations = tel_data.get("locations", [])
        if locations:
            speeds = np.array([loc.get("speed", 0) for loc in locations])
            event_features['speed_mean'] = np.mean(speeds)
            
            # 1st percentile of deceleration (ignoring 1-frame GPS jumps)
            if len(speeds) > 1:
                speed_changes = np.diff(speeds)
                event_features['max_deceleration'] = np.percentile(speed_changes, 1)
            else:
                event_features['max_deceleration'] = 0

    # Append the fully enriched row to our master list
    processed_data.append(event_features)

# Build the Final DataFrame and fill any missing sensor gaps with 0
features_df = pd.DataFrame(processed_data).fillna(0)

# Merge the target variables (event type, conflict type) from the original CSV
final_model_df = features_df.merge(bdd_df[['BDD_ID', 'EVENT_TYPE', 'CONFLICT_TYPE']], on='BDD_ID', how='inner')

print(final_model_df.head())
print(f"Final usable dataset size: {final_model_df.shape}")

# Create train/val/test splits (70/15/15)
unique_ids = final_model_df['BDD_ID'].unique().tolist()
print(f"Total Unique IDs available for splitting: {len(unique_ids)}")

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

print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}")

# Define the target directory path
output_dir = Path("data/bdd_sce_features")

# Create the directory if it does not exist
# parents = True allows it to create the data folder too if that is somehow missing
# exist_ok = gTrue prevents it from crashing if the folder is already there
output_dir.mkdir(parents=True, exist_ok=True)

# Save the exact splits to disk inside the new folder
train_df.to_csv(output_dir / "train_features.csv", index=False)
val_df.to_csv(output_dir / "val_features.csv", index=False)
test_df.to_csv(output_dir / "test_features.csv", index=False)


# One-Hot Encode the categorical columns (weather, timeofday)
# This turns 'weather' into columns like 'weather_clear', 'weather_rainy' with 1s and 0s.
train_encoded = pd.get_dummies(train_df, columns=['weather', 'timeofday'])
val_encoded = pd.get_dummies(val_df, columns=['weather', 'timeofday'])
test_encoded = pd.get_dummies(test_df, columns=['weather', 'timeofday'])

# Ensure all dataframes have the exact same columns after encoding
# (Sometimes the val set might be missing a weather type that was in the train set)
val_encoded = val_encoded.reindex(columns=train_encoded.columns, fill_value=0)
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

# Define features (X) and targets (y)
# Let's focus on predicting 'EVENT_TYPE' first
drop_cols = ['BDD_ID', 'EVENT_TYPE', 'CONFLICT_TYPE'] 

X_train = train_encoded.drop(columns=drop_cols)
y_train = train_encoded['EVENT_TYPE']

X_val = val_encoded.drop(columns=drop_cols)
y_val = val_encoded['EVENT_TYPE']

X_test = test_encoded.drop(columns=drop_cols)
y_test = test_encoded['EVENT_TYPE']

# Scale the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Initialize the model
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Train the model on your training data
rf_model.fit(X_train_scaled, y_train)

print("Training Complete!")

# Make predictions on the validation set
val_predictions = rf_model.predict(X_val_scaled)

# Print out the scorecard
print("Validation Accuracy:", accuracy_score(y_val, val_predictions))
print("\nDetailed Classification Report:")
print(classification_report(y_val, val_predictions))


# MODEL 2: XGBOOST (BENCHMARK)
print("Training XGBoost...")

# XGBoost requires targets to be 0-indexed (0, 1, 2, 3). 
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Calculate sample weights to balance the classes exactly like RF's class_weight='balanced'
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_encoded)

# Initialize XGBoost
xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')

# Train the model
xgb_model.fit(X_train_scaled, y_train_encoded, sample_weight=sample_weights)

print("XGBoost Training Complete!")

# Predict on the validation set
xgb_preds_encoded = xgb_model.predict(X_val_scaled)

# Convert the 0-indexed predictions back to the original 1, 2, 3, 4 labels
val_predictions_xgb = le.inverse_transform(xgb_preds_encoded)

# Print out the scorecard
print("XGBoost Validation Accuracy:", accuracy_score(y_val, val_predictions_xgb))
print("\nDetailed XGBoost Classification Report:")
print(classification_report(y_val, val_predictions_xgb))