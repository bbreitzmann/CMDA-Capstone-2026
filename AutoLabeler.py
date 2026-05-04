import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

INPUT_FILE = r"C:/Users/brend/Downloads/New folder (2)/CMDA4654/Capstone/joint_multitask_four_head_test_predictions.csv"
OUTPUT_DIR = r"C:/Users/brend/Downloads/New folder (2)/CMDA4654/Capstone"
MODEL_PATH = r"C:/Users/brend/Downloads/New folder (2)/CMDA4654/Capstone/rf_autolabeler.pkl"


TRAIN_MODE = True    
AUDIT_PERCENT = 0.20 


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

EVENT_MAP = {
    'Conflict': 1, 
    'Bump': 2, 
    'Hard Brake': 3, 
    'Not an SCE': 4, 
    'Not SCE': 4
}


CONFLICT_MAP = {
    'Q': 'Conflict with lead vehicle',
    'W': 'Single vehicle conflict',
    'E': 'Conflict with vehicle turning into another vehicle path (same direction)',
    'R': 'Conflict with parked vehicle',
    'T': 'Conflict with vehicle in adjacent lane',
    'Y': 'Conflict with vehicle turning across another vehicle path (opposite direction)',
    'U': 'Conflict with a following vehicle',
    'I': 'Conflict with vehicle turning into another vehicle path (opposite direction)',
    'O': 'Conflict with vehicle moving across another vehicle path (through intersection)',
    'P': 'Conflict with animal',
    'A': 'Conflict with vehicle turning across another vehicle path (same direction)',
    'S': 'Conflict with merging vehicle',
    'D': 'Conflict with pedalcyclist',
    'F': 'Conflict with pedestrian',
    'G': 'Conflict with obstacle/object in roadway',
    'H': 'Conflict with oncoming traffic',
    'J': 'Other'
}


df = pd.read_csv(INPUT_FILE)

event_features = ['Conflict', 'Bump', 'Hard Brake', 'Not SCE']
conf_cols = [f"conf17_{letter}" for letter in CONFLICT_MAP.keys()]


has_labels = 'event_true_label' in df.columns
if has_labels:
    df['true_event_code'] = df['event_true_label'].map(EVENT_MAP)
    df['orig_pred_code'] = df['event_pred_label'].map(EVENT_MAP)

if TRAIN_MODE:
    if not has_labels:
        raise ValueError("TRAIN_MODE is True but no true labels found in CSV.")
    
    print("Training new weights...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
    train_df = df.dropna(subset=['true_event_code'])
    rf.fit(train_df[event_features], train_df['true_event_code'])
    joblib.dump(rf, MODEL_PATH)
    print(f"Weights saved to: {MODEL_PATH}")
else:
    print(f"Loading existing weights from: {MODEL_PATH}")
    rf = joblib.load(MODEL_PATH)


df['final_event_type'] = rf.predict(df[event_features])
df['confidence_score'] = np.max(rf.predict_proba(df[event_features]), axis=1)

def get_best_conflict_letter(row):
    
    if row['final_event_type'] == 1:
    
        probs = {letter: row[f"conf17_{letter}"] for letter in CONFLICT_MAP.keys() if f"conf17_{letter}" in df.columns}
        if probs:
            
            return max(probs, key=probs.get).lower()
    return ""

df['final_conflict_type'] = df.apply(get_best_conflict_letter, axis=1)


if has_labels:
    acc_path = os.path.join(OUTPUT_DIR, "accuracy_diagnostics.csv")
    new_acc = accuracy_score(df['true_event_code'], df['final_event_type'])
    orig_acc = accuracy_score(df['true_event_code'], df['orig_pred_code'])
    cm = confusion_matrix(df['true_event_code'], df['final_event_type'])
    
    diag_df = pd.DataFrame({
        'Metric': ['Original Accuracy', 'Auto-Labeler Accuracy', 'Improvement'],
        'Value': [f"{orig_acc:.4f}", f"{new_acc:.4f}", f"{(new_acc - orig_acc):.4f}"]
    })
    diag_df.to_csv(acc_path, index=False)
    

    weights_df = pd.DataFrame({'Feature': event_features, 'Weight': rf.feature_importances_})
    with open(acc_path, 'a') as f:
        f.write("\nLEARNED FEATURE WEIGHTS\n")
        weights_df.to_csv(f, index=False)
        f.write("\nCONFUSION MATRIX (1=Conf, 2=Bump, 3=HB, 4=NotSCE)\n")
        pd.DataFrame(cm).to_csv(f)
    print(f"Diagnostics saved to: {acc_path}")

df = df.sort_values('confidence_score', ascending=False)
df['EVENT_ID'] = range(1, len(df) + 1)

formatted_master = pd.DataFrame()
formatted_master['EVENT_ID'] = df['EVENT_ID']
formatted_master['BDD_ID'] = df['BDD_ID']
formatted_master['EVENT_TYPE'] = df['final_event_type'].astype(int)
formatted_master['CONFLICT_TYPE'] = df['final_conflict_type']
formatted_master['BDD_START'] = df['start_pred'].round(2)


num_audit = int(len(df) * AUDIT_PERCENT)
audit_df = formatted_master.tail(num_audit)
clean_df = formatted_master.head(len(df) - num_audit)


clean_path = os.path.join(OUTPUT_DIR, "bdd_sce_output.csv")
audit_path = os.path.join(OUTPUT_DIR, "audit_list.csv")

clean_df.to_csv(clean_path, index=False)
audit_df.to_csv(audit_path, index=False)

print("Auto Label Complete!")