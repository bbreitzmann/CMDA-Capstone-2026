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
duplicate_count = 0

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
        
        # --- THE FIX: Check if we already have this video ---
        if os.path.exists(destination_path):
            # Duplicate found! Delete the extra copy from the batch folder
            os.remove(source_path)
            duplicate_count += 1
        else:
            # Move the new file to our permanent folder
            shutil.move(source_path, destination_path)
            found_count += 1
    else:
        # Delete the file permanently to free up disk space
        os.remove(source_path)
        deleted_count += 1

print(f"Batch processing complete!")
print(f"Saved {found_count} new videos to {saved_folder}.")
print(f"Skipped {duplicate_count} repeated duplicate videos.")
print(f"Deleted {deleted_count} useless videos to save space.")


## Create dataframe for downloaded videos only

# Get all video IDs currently sitting in the saved folder
downloaded_files = os.listdir(saved_folder)
downloaded_ids = [os.path.splitext(f)[0] for f in downloaded_files if not f.startswith(".")]

# Filter the original meta dataframe to only include rows we have the video for
df_downloaded = df[df['BDD_ID'].astype(str).isin(downloaded_ids)]

# Merge w/ original annotations for conflict type
# Load the original bdd_sce.csv file (Make sure the path matches where it is on your computer)
original_annotations = pd.read_csv("data/bdd_sce.csv")

# We merge on BOTH 'EVENT_ID' and 'BDD_ID'. 
# This is crucial because a few videos actually have multiple events in them!
df_downloaded = df_downloaded.merge(
    original_annotations[['EVENT_ID', 'BDD_ID', 'CONFLICT_TYPE']], 
    on=['EVENT_ID', 'BDD_ID'], 
    how='left'
)
# Save to a new CSV file
out_csv_path = "downloaded_videos_meta.csv"
df_downloaded.to_csv(out_csv_path, index=False)

print(f"Tracking Update: You now have {len(df_downloaded)} out of {len(df)} total annotated videos.")
print(f"Saved current progress to: {out_csv_path}")