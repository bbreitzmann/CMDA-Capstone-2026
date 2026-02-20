# Explore the csv: bdd_sce.csv which was provided by Dr. Liang Shi for initial exploration
# Utilizing a cascade classifier to evaluate the safety critical events
# Big difference here outside of cascade architecture, is using 16 second windows to better understand a clip wholistically

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

# Setup local file paths for the static labels, the time-series telemetry, and the target dataset
labels_dir = Path("data/100k/train")
telemetry_dir = Path("data/bddk100_info/100k/train")
bdd_df = pd.read_csv("data/bdd_sce.csv")

# Create a fast lookup set of IDs so we only process files that exist in our target CSV
valid_ids = set(bdd_df['BDD_ID'].unique())

# This list will hold a dictionary of extracted features for every valid video
processed_data = []

print("Processing JSON files...")

# Loop through every static label file in the directory
for label_path in tqdm(labels_dir.glob("*.json")):
    
    # Extract the base filename to use as the unique ID
    BDD_ID = label_path.stem 
    
    # Skip this file if it is not part of our target dataset
    if BDD_ID not in valid_ids:
        continue
        
    # Construct the path to the matching telemetry file
    telem_path = telemetry_dir / f"{BDD_ID}.json"
    
    # Skip if we have a label file but no matching telemetry data
    if not telem_path.exists():
        continue

    # Initialize a dictionary to store all engineered features for this specific driving clip
    event_features = {"BDD_ID": BDD_ID}

    # Open the label JSON to extract environmental and spatial context
    with open(label_path, 'r') as f:
        lbl_data = json.load(f)
        
        # Extract basic environmental conditions
        attrs = lbl_data.get("attributes", {})
        event_features['weather'] = attrs.get("weather", "unknown")
        event_features['timeofday'] = attrs.get("timeofday", "unknown")
        
        # The BDD100K labels usually only contain annotations for a single keyframe
        frames = lbl_data.get("frames", [])
        if frames:
            objects = frames[0].get("objects", [])
            
            # Count the amount of traffic and actors in the scene
            event_features['num_cars'] = sum(1 for obj in objects if obj.get("category") == "car")
            event_features['num_pedestrians'] = sum(1 for obj in objects if obj.get("category") == "pedestrian")
            event_features['num_trucks'] = sum(1 for obj in objects if obj.get("category") == "truck")
            event_features['total_objects'] = len(objects)
            
            # Create a flag indicating if a Vulnerable Road User is present
            # Near misses with pedestrians or bikes usually cause sharper physical reactions than with cars
            event_features['has_vru'] = 1 if (event_features['num_pedestrians'] > 0 or 
                                              sum(1 for obj in objects if obj.get("category") in ["rider", "bicycle"]) > 0) else 0
            
            # Calculate the total pixel area of the standard BDD100K image resolution
            IMAGE_AREA = 1280 * 720
            max_area = 0
            drivable_area = 0
            
            # Iterate through the bounding boxes and polygons to understand the threat landscape
            for obj in objects:
                
                # Find the single largest object in the frame to determine if there is a looming threat
                if "box2d" in obj:
                    box = obj["box2d"]
                    area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                    if area > max_area:
                        max_area = area
                        
                # Calculate how much space the driver actually has to maneuver
                # This uses the Shoelace formula to find the area of the drivable polygon
                elif obj.get("category") == "area/drivable" and "poly2d" in obj:
                    poly = obj["poly2d"]
                    x = np.array([p[0] for p in poly if isinstance(p[0], (int, float))])
                    y = np.array([p[1] for p in poly if isinstance(p[1], (int, float))])
                    
                    if len(x) > 2 and len(x) == len(y):
                        drivable_area += 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            # Convert raw pixel areas into ratios for the machine learning models
            event_features['max_object_ratio'] = max_area / IMAGE_AREA
            event_features['drivable_area_ratio'] = drivable_area / IMAGE_AREA

    # Open the telemetry JSON to extract the physics of the vehicle
    with open(telem_path, 'r') as f:
        tel_data = json.load(f) 
        
        # Time Window Logic
        # Instead of summarizing 40 seconds of video which dilutes the signal
        # we isolate a specific window of time immediately surrounding the annotated event
        
        # Time Window Logic
        label_ts = 0
        if frames and "timestamp" in frames[0]:
            label_ts = frames[0]["timestamp"] 
            
        start_time = tel_data.get("startTime", 0)
        event_time_unix = start_time + label_ts
        
        # Look 6 seconds back to catch early evasive maneuvers (Near-Misses)
        # Look 10 seconds forward to ensure we capture the car coming to a full stop (Crashes)
        lookback_ms = 6000 
        lookforward_ms = 10000 
        min_time = event_time_unix - lookback_ms
        max_time = event_time_unix + lookforward_ms
        
        # Extract Accelerometer data filtered to our specific time window
        accel = tel_data.get("accelerometer", [])
        if accel:
            window_accel = [a for a in accel if min_time <= a['timestamp'] <= max_time]
            
            if window_accel:
                accel_x = np.array([a['x'] for a in window_accel])
                accel_y = np.array([a['y'] for a in window_accel])
                accel_z = np.array([a['z'] for a in window_accel])
                
                # Use percentiles instead of absolute max or min to filter out single frame sensor noise
                event_features['accel_z_min'] = np.percentile(accel_z, 1) 
                event_features['accel_z_max'] = np.percentile(accel_z, 99) 
                event_features['accel_x_std'] = np.std(accel_x) 
                
                # Calculate the total magnitude of force using the Pythagorean theorem in 3D space
                accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
                event_features['max_g_force'] = np.percentile(accel_magnitude, 99)
                
                # Calculate jerk which is the rate of change of acceleration over time
                if len(accel_z) > 1:
                    jerk_z = np.diff(accel_z)
                    event_features['max_jerk_z'] = np.percentile(np.abs(jerk_z), 99)
                else:
                    event_features['max_jerk_z'] = 0
            
        # Extract Gyroscope data filtered to our specific time window
        gyro = tel_data.get("gyro", [])
        if gyro:
            window_gyro = [g for g in gyro if min_time <= g['timestamp'] <= max_time]
            if window_gyro:
                gyro_y = [g['y'] for g in window_gyro] 
                event_features['gyro_yaw_max_abs'] = np.percentile(np.abs(gyro_y), 99) 

        # Extract Speed data filtered to our specific time window
        # Extract Speeds 
        locations = tel_data.get("locations", [])
        if locations:
            
            # We look at the entire unwindowed file to find the ultimate resting state of the car
            # This gives Phase 2 the critical context of whether the driver drove away or remained stopped
            global_speeds = np.array([loc.get("speed", 0) for loc in locations])
            if len(global_speeds) >= 3:
                event_features['global_speed_end'] = np.mean(global_speeds[-3:])
            else:
                event_features['global_speed_end'] = 0

            # THE WINDOWED DYNAMICS
            # We filter the speed data to only look at the exact moment of the event
            window_locs = [loc for loc in locations if min_time <= loc['timestamp'] <= max_time]
            if window_locs:
                speeds = np.array([loc.get("speed", 0) for loc in window_locs])
                
                # Analyze if the driver was driving smoothly or chaotically during the event
                event_features['speed_mean'] = np.mean(speeds)
                event_features['speed_std'] = np.std(speeds)
                
                # Reintroducing the smoothing average to protect against single frame GPS glitches
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

    # Add the fully populated dictionary of features to our main list
    processed_data.append(event_features)

# Convert the list of dictionaries into a Pandas DataFrame and replace missing sensor data with zero
features_df = pd.DataFrame(processed_data).fillna(0)

# Bring in the true target variables from the original dataset mapping by the unique ID
final_model_df = features_df.merge(bdd_df[['BDD_ID', 'EVENT_TYPE', 'CONFLICT_TYPE']], on='BDD_ID', how='inner')

print(final_model_df.head())
print(f"Final usable dataset size: {final_model_df.shape}")

# Create reproducible Train, Validation, and Test splits
unique_ids = final_model_df['BDD_ID'].unique().tolist()
print(f"Total Unique IDs available for splitting: {len(unique_ids)}")

np.random.seed(1)
np.random.shuffle(unique_ids)

# Split the data into seventy percent train, fifteen percent validation, and fifteen percent test sets
train_cutoff = int(0.70 * len(unique_ids))
val_cutoff = int(0.85 * len(unique_ids)) 

train_ids = unique_ids[:train_cutoff]
val_ids = unique_ids[train_cutoff:val_cutoff]
test_ids = unique_ids[val_cutoff:]

train_df = final_model_df[final_model_df['BDD_ID'].isin(train_ids)]
val_df = final_model_df[final_model_df['BDD_ID'].isin(val_ids)]
test_df = final_model_df[final_model_df['BDD_ID'].isin(test_ids)]

print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}")

# Save the exact splits to disk so experiments can be reproduced without rebuilding features
train_df.to_csv("train_features.csv", index=False)
val_df.to_csv("val_features.csv", index=False)
test_df.to_csv("test_features.csv", index=False)

# One Hot Encoding converts categorical text strings into binary columns so the models can read them
train_encoded = pd.get_dummies(train_df, columns=['weather', 'timeofday'])
val_encoded = pd.get_dummies(val_df, columns=['weather', 'timeofday'])
test_encoded = pd.get_dummies(test_df, columns=['weather', 'timeofday'])

# Reindex ensures that if the validation set happens to be missing a weather type present in training
# the column is still created but filled with zeros to prevent dimension errors
val_encoded = val_encoded.reindex(columns=train_encoded.columns, fill_value=0)
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

# Separate the predictive features from the target answers
drop_cols = ['BDD_ID', 'EVENT_TYPE', 'CONFLICT_TYPE'] 

X_train = train_encoded.drop(columns=drop_cols)
y_train = train_encoded['EVENT_TYPE']

X_val = val_encoded.drop(columns=drop_cols)
y_val = val_encoded['EVENT_TYPE']

X_test = test_encoded.drop(columns=drop_cols)
y_test = test_encoded['EVENT_TYPE']

# Standardize the numerical features so large values do not dominate smaller values in the algorithms
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Cascade Classifier Architecture
# We use a two step cascaded model to prevent the algorithms from getting overwhelmed
# The first model only decides if a driving clip is an anomaly or normal driving
# The second model only looks at confirmed anomalies and decides the severity of the event

print("Training Random Forest Cascade...")

# Translate the four classes into binary targets for the Trigger model
y_train_binary = y_train.apply(lambda x: 0 if x == 4 else 1)
y_val_binary = y_val.apply(lambda x: 0 if x == 4 else 1)

# Train the Anomaly Trigger
rf_trigger = RandomForestClassifier(n_estimators=100, random_state=42)
rf_trigger.fit(X_train_scaled, y_train_binary)

# Filter out normal driving data so the Classifier model only trains on chaotic events
event_mask_train = (y_train != 4)
X_train_events = X_train_scaled[event_mask_train]
y_train_events = y_train[event_mask_train]

# Train the Event Classifier using balanced class weights to account for rare near miss events
rf_classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_classifier.fit(X_train_events, y_train_events)

# Pass the validation set through the cascade architecture
# Set the default prediction for everything to normal driving
val_preds_binary = rf_trigger.predict(X_val_scaled)
final_predictions = np.full(len(y_val), 4)

# Locate the exact items the Trigger model flagged as anomalies
event_indices = np.where(val_preds_binary == 1)[0]

# Feed only the flagged anomalies into the second Classifier model to determine the final type
if len(event_indices) > 0:
    X_val_flagged = X_val_scaled[event_indices]
    event_specific_preds = rf_classifier.predict(X_val_flagged)
    final_predictions[event_indices] = event_specific_preds

print("\n PHASE 1: ANOMALY TRIGGER EVALUATION")
print("Accuracy:", accuracy_score(y_val_binary, val_preds_binary))
print(classification_report(y_val_binary, val_preds_binary, target_names=["Non-Event (4)", "Event (1,2,3)"]))

print("\n PHASE 2: EVENT CLASSIFIER EVALUATION")
# Evaluate the second model strictly on data it was meant to see to isolate its real performance
event_mask_val = (y_val != 4)
X_val_actual_events = X_val_scaled[event_mask_val]
y_val_actual_events = y_val[event_mask_val]
if len(y_val_actual_events) > 0:
    val_preds_events_only = rf_classifier.predict(X_val_actual_events)
    print("Accuracy:", accuracy_score(y_val_actual_events, val_preds_events_only))
    print(classification_report(y_val_actual_events, val_preds_events_only))
else:
    print("No events found in validation set to evaluate Model 2.")

print("\n PHASE 3: FULL CASCADED PIPELINE EVALUATION")
print("Validation Accuracy:", accuracy_score(y_val, final_predictions))
print(classification_report(y_val, final_predictions))


# XGBoost Cascade Implementation
print("\n" + "="*50)
print("Training XGBoost Cascade...")

# Train the XGBoost Anomaly Trigger
xgb_trigger = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_trigger.fit(X_train_scaled, y_train_binary)

# XGBoost requires target labels to be strictly zero indexed
# The Label Encoder translates the remaining classes down to zero indexing
le = LabelEncoder()
y_train_events_encoded = le.fit_transform(y_train_events)

# XGBoost does not have a built in balanced parameter for multi class problems
# We generate an array of sample weights manually to force the model to pay attention to rare classes
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_events_encoded)

# Train the XGBoost Event Classifier
xgb_classifier = XGBClassifier(random_state=42, eval_metric='mlogloss')
xgb_classifier.fit(X_train_events, y_train_events_encoded, sample_weight=sample_weights)

# Pass the validation set through the XGBoost cascade architecture
val_preds_binary_xgb = xgb_trigger.predict(X_val_scaled)
final_predictions_xgb = np.full(len(y_val), 4)
event_indices_xgb = np.where(val_preds_binary_xgb == 1)[0]

# Feed flagged anomalies into the second XGBoost model and decode the answers back to the original labels
if len(event_indices_xgb) > 0:
    X_val_flagged_xgb = X_val_scaled[event_indices_xgb]
    event_specific_preds_encoded_xgb = xgb_classifier.predict(X_val_flagged_xgb)
    event_specific_preds_xgb = le.inverse_transform(event_specific_preds_encoded_xgb)
    final_predictions_xgb[event_indices_xgb] = event_specific_preds_xgb

print("\n PHASE 1: ANOMALY TRIGGER (XGBOOST)")
print("Accuracy:", accuracy_score(y_val_binary, val_preds_binary_xgb))
print(classification_report(y_val_binary, val_preds_binary_xgb, target_names=["Non-Event (4)", "Event (1,2,3)"]))

print("\n PHASE 2: EVENT CLASSIFIER (XGBOOST)")
if len(y_val_actual_events) > 0:
    val_preds_events_only_encoded = xgb_classifier.predict(X_val_actual_events)
    val_preds_events_only = le.inverse_transform(val_preds_events_only_encoded)
    print("Accuracy:", accuracy_score(y_val_actual_events, val_preds_events_only))
    print(classification_report(y_val_actual_events, val_preds_events_only))

print("\n PHASE 3: FULL CASCADED PIPELINE EVALUATION (XGBOOST)")
print("Validation Accuracy:", accuracy_score(y_val, final_predictions_xgb))
print(classification_report(y_val, final_predictions_xgb))