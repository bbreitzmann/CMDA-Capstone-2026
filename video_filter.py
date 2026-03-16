import os
import shutil
import pandas as pd

# Load the IDs we actually care about (the 2,500 annotated clips)
csv_path = "meta.csv" 
df = pd.read_csv(csv_path)

# Convert to a set for instant lookup speeds
target_ids = set(df['BDD_ID'].dropna().astype(str).tolist())

# unzipped the current batch
batch_folder = "data/bdd100k/videos/train" 

# master folder where you want to keep the final 2,500 videos
saved_folder = "data/annotated_videos_only" 
os.makedirs(saved_folder, exist_ok=True)

# Process the batch
found_count = 0
deleted_count = 0

print(f"Scanning {batch_folder} for annotated videos...")

for filename in os.listdir(batch_folder):
    # Ignore hidden system files like .DS_Store
    if filename.startswith("."):
        continue

    # Extract the ID by removing the file extension
    video_id, ext = os.path.splitext(filename)
    source_path = os.path.join(batch_folder, filename)

    # Check if the video ID is in our annotated list
    if video_id in target_ids:
        destination_path = os.path.join(saved_folder, filename)
        # Move the file to our permanent folder
        shutil.move(source_path, destination_path)
        found_count += 1
    else:
        # Delete the file permanently to free up disk space
        os.remove(source_path)
        deleted_count += 1

print(f"Batch processing complete!")
print(f"Saved {found_count} videos to {saved_folder}.")
print(f"Deleted {deleted_count} useless videos to save space.")