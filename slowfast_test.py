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


import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm 
from pytorchvideo.models.hub import slowfast_r50
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, Normalize
from pytorchvideo.transforms import ShortSideScale, UniformTemporalSubsample


CLASS_MAP = {
    0: "Not an SCE",
    1: "Conflict",
    2: "Bump",
    3: "Hard Brake"
}


def get_vtti_label(row):
    e_type = row['EVENT_TYPE']
    if e_type == 1: return "Conflict"
    if e_type == 2: return "Bump"
    if e_type == 3: return "Hard Brake"
    if e_type == 4: return "Not an SCE"
    return "Unknown"


def extract_kinematics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    a, g, l = data.get('accelerometer', []), data.get('gyro', []), data.get('locations', [])
    return {
        'BDD_ID': os.path.basename(json_path).replace('.json', ''),
        'accel_x_max': max([abs(i['x']) for i in a]) if a else 0,
        'accel_z_jerk': max(np.abs(np.diff([i['z'] for i in a]))) if len(a) > 1 else 0,
        'total_g_force': max([np.sqrt(i['x']**2 + i['y']**2 + i['z']**2) for i in a]) if a else 0,
        'gyro_yaw_max': max([abs(i['y']) for i in g]) if g else 0,
        'speed_max': max([i['speed'] for i in l]) if l else 0,
        'speed_mean': np.mean([i['speed'] for i in l]) if l else 0,
        'deceleration_max': abs(min(np.diff([i['speed'] for i in l]))) if len(l) > 1 else 0
    }

class HybridSlowFast(nn.Module):
    def __init__(self, num_kinematic_features=7, num_classes=4):
        super().__init__()
        self.slowfast = slowfast_r50(pretrained=True)
        self.slowfast.blocks[6].proj = nn.Identity() 
        self.kin_net = nn.Sequential(nn.Linear(num_kinematic_features, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(2304 + 32, 256), nn.ReLU(), nn.Linear(256, num_classes))

    def forward(self, video_pathway, kinematic_data):
        v_feats = self.slowfast(video_pathway)
        k_feats = self.kin_net(kinematic_data)
        return self.classifier(torch.cat((v_feats, k_feats), dim=1))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    json_dir = r"C:\Users\brend\Downloads\kinematic_data"
    video_dir = r"C:\Users\brend\Downloads\extracted_videos"
    meta_csv_path = r"C:\Users\brend\Downloads\downloaded_videos_meta - downloaded_videos_meta.csv" 
    
    print("Loading original VTTI metadata labels...")
    try:
        meta_df = pd.read_csv(meta_csv_path)
        meta_df['GT_Label'] = meta_df.apply(get_vtti_label, axis=1)
    except FileNotFoundError:
        print(f"ERROR: Could not find {meta_csv_path} in the current folder.")
        sys.exit()

    print(f"Initializing Model on {device}...")
    model = HybridSlowFast().to(device)
    model.eval()

    video_transform = Compose([
        Lambda(lambda x: x / 255.0),
        ShortSideScale(size=256),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),
        Normalize([0.45]*3, [0.225]*3),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),
        Lambda(lambda x: [UniformTemporalSubsample(8)(x), UniformTemporalSubsample(32)(x)])
    ])

    all_results = []
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    print(f"Starting analysis on {len(json_files)} videos...")

    for json_file in tqdm(json_files, desc="Processing Videos", unit="vid"):
        bdd_id = json_file.replace('.json', '')
        vid_path = os.path.join(video_dir, f"{bdd_id}.mov")
        
        if not os.path.exists(vid_path): 
            continue

        try:
            
            orig_row = meta_df[meta_df['BDD_ID'] == bdd_id]
            ground_truth = orig_row['GT_Label'].values[0] if not orig_row.empty else "Unknown"

            
            kin = extract_kinematics(os.path.join(json_dir, json_file))
            k_cols = ['accel_x_max', 'accel_z_jerk', 'total_g_force', 'gyro_yaw_max', 'speed_max', 'speed_mean', 'deceleration_max']
            kin_tensor = torch.tensor([kin[c] for c in k_cols], dtype=torch.float32).unsqueeze(0).to(device)

            
            video = EncodedVideo.from_path(vid_path)
            timeline_scores = []
            
            for start_t in range(0, 33, 10): 
                clip = video.get_clip(start_sec=start_t, end_sec=start_t + 8)
                frames = [f.unsqueeze(0).to(device) for f in video_transform(clip['video'])]
                with torch.no_grad():
                    logits = model(frames, kin_tensor)
                    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                    timeline_scores.append({
                        'score': float(np.sum(probs[1:])), 
                        'class': np.argmax(probs),
                        'timestamp': f"{start_t}s-{start_t+8}s"
                    })

            peak = max(timeline_scores, key=lambda x: x['score'])
            ai_prediction = CLASS_MAP[peak['class']]

            all_results.append({
                'BDD_ID': bdd_id,
                'Original_GT': ground_truth,
                'AI_Prediction': ai_prediction,
                'Timestamp': peak['timestamp'],
                'Match': "MATCH" if ai_prediction == ground_truth else "MISMATCH"
            })

        except Exception as e:
            print(f"Error on {bdd_id}: {e}")

    
    if all_results:
        print("\n" + "="*90)
        print(f"{'VIDEO IDENTIFIER':<20} | {'PREDICTED':<15} | {'ACTUAL':<15} | {'TIME WINDOW'}")
        print("-" * 90)
        
        for res in all_results:
            status = "✓" if res['Match'] == "MATCH" else "✗"
            print(f"{res['BDD_ID'][:18]:<20} | "
                  f"{res['AI_Prediction']:<15} | "
                  f"{res['Original_GT']:<15} | "
                  f"{res['Timestamp']} {status}")
        
        print("="*90)
        matches = sum(1 for r in all_results if r['Match'] == "MATCH")
        print(f"Final Count: {matches}/{len(all_results)} matches agreed with original metadata.")