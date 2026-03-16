import os
import json
import glob
import numpy as np
import pandas as pd

# Path to event-level CSV (one row per event)
EVENT_CSV = "data/bdd_sce.csv"

# Folder containing the single-frame context/annotation JSONs
LABELS_JSON_DIR = "data/100k/train"

# Folder containing the time-series kinematics JSONs
INFO_JSON_DIR = "data/bddk100_info/100k/train"

# Output files
OUT_TS = "X_ts.npy" # CNN tensor: (N, C, T)
OUT_CTX = "X_ctx.csv" # Static context features
OUT_META = "meta.csv" # Traceability / labels

# Time-series window settings
PRE_SEC = 3.0 # seconds BEFORE event time
POST_SEC = 3.0 # seconds AFTER event time
T = 256 # fixed number of time steps after resampling

# Channel settings for time-series
USE_GYRO = True # include gyro x/y/z from info JSON
USE_SPEED = True # include speed from locations
STANDARDIZE_PER_CHANNEL = False # set True for per-sample z-scoring

# Label mapping
# "4way" keeps EVENT_TYPE 1,2,3,4
# "3way" drops class 4 ("Not an SCE")
LABEL_MODE = "4way" 


# HELPER: BUILD FILE INDEXES
def build_json_index(folder, id_keys):
    """
    Build a dictionary mapping possible IDs -> json file path.

    This is helpful because sometimes the JSON filename matches the ID,
    and sometimes the ID is stored inside the JSON body.

    Parameters
    folder : str
        Directory containing JSON files.
    id_keys : list[str]
        Keys to look for inside the JSON, e.g. ["name"] or ["rideID"].

    Returns
    dict
        Mapping from discovered ID strings to file path.
    """
    index = {}

    for fp in glob.glob(os.path.join(folder, "*.json")):
        # Index by filename stem first
        stem = os.path.splitext(os.path.basename(fp))[0]
        index[stem] = fp

        # Also index by IDs stored inside JSON body
        try:
            with open(fp, "r") as f:
                d = json.load(f)

            for key in id_keys:
                if key in d and isinstance(d[key], str):
                    index[d[key]] = fp
        except Exception:
            # If file is malformed, skip silently for now
            continue

    return index


# HELPER: LABEL MAPPING
def map_event_type(event_type, mode="4way"):
    """
    Convert original EVENT_TYPE values into zero-based class labels.

    Original meaning:
      1 = Conflict
      2 = Bump
      3 = Hard Brake
      4 = Not an SCE

    Returns
    int or None
        Zero-based class index, or None if the row should be dropped.
    """
    event_type = int(event_type)

    if mode == "4way":
        return event_type - 1   # 1->0, 2->1, 3->2, 4->3

    if mode == "3way":
        if event_type == 4:
            return None
        return event_type - 1   # 1->0, 2->1, 3->2

    raise ValueError("LABEL_MODE must be '4way' or '3way'")



# PARSE LABELS JSON (STATIC CONTEXT)
def box_area(obj):
    """
    Compute box area for objects with a box2d field.
    Returns None if no box2d exists (e.g. poly2d lanes/areas).
    """
    box = obj.get("box2d")
    if box is None:
        return None

    width = max(0.0, float(box["x2"]) - float(box["x1"]))
    height = max(0.0, float(box["y2"]) - float(box["y1"]))
    return width * height


def extract_labels_context(labels_json):
    """
    Extract static context features from the labels JSON.

    This JSON is the single-timestamp "scene snapshot" stream.
    It contains:
      - clip-level attributes (weather, scene, timeofday)
      - one annotated frame with objects at a single timestamp

    Returns
    dict
        One row of static features keyed by BDD_ID.
    """
    bdd_id = labels_json.get("name")

    # Clip-level attributes
    clip_attrs = labels_json.get("attributes", {})
    weather = clip_attrs.get("weather", "undefined")
    scene = clip_attrs.get("scene", "undefined")
    timeofday = clip_attrs.get("timeofday", "undefined")

    # Single annotated frame
    frames = labels_json.get("frames", [])
    if len(frames) > 0:
        frame = frames[0]
        label_ts_ms = frame.get("timestamp") # e.g. 10000 ms
        objects = frame.get("objects", [])
    else:
        label_ts_ms = np.nan
        objects = []

    # Count object categories with box2d (things like car, traffic light, etc.)
    categories = [obj.get("category", "unknown") for obj in objects]

    # Object counts
    n_car = sum(cat == "car" for cat in categories)
    n_pedestrian = sum(cat == "pedestrian" for cat in categories)
    n_truck = sum(cat == "truck" for cat in categories)
    n_bus = sum(cat == "bus" for cat in categories)
    n_traffic_light = sum(cat == "traffic light" for cat in categories)
    n_traffic_sign = sum(cat == "traffic sign" for cat in categories)

    # Count only objects that have boxes (not drivable area / lane poly2d)
    boxed_objects = [obj for obj in objects if "box2d" in obj]
    total_boxed_objects = len(boxed_objects)

    # Bounding-box area summaries
    areas = [box_area(obj) for obj in boxed_objects]
    areas = [a for a in areas if a is not None]

    if len(areas) > 0:
        min_box_area = float(np.min(areas))
        max_box_area = float(np.max(areas))
        mean_box_area = float(np.mean(areas))
    else:
        min_box_area = np.nan
        max_box_area = np.nan
        mean_box_area = np.nan

    # Traffic light colors
    tl_colors = []
    for obj in objects:
        if obj.get("category") == "traffic light":
            color = obj.get("attributes", {}).get("trafficLightColor", "none")
            tl_colors.append(color)

    n_tl_red = sum(c == "red" for c in tl_colors)
    n_tl_yellow = sum(c == "yellow" for c in tl_colors)
    n_tl_green = sum(c == "green" for c in tl_colors)

    # Occlusion / truncation info
    occluded_flags = []
    truncated_flags = []
    for obj in boxed_objects:
        attrs = obj.get("attributes", {})
        occluded_flags.append(bool(attrs.get("occluded", False)))
        truncated_flags.append(bool(attrs.get("truncated", False)))

    pct_occluded = float(np.mean(occluded_flags)) if len(occluded_flags) > 0 else np.nan
    pct_truncated = float(np.mean(truncated_flags)) if len(truncated_flags) > 0 else np.nan

    return {
        "BDD_ID": bdd_id,
        "label_ts_ms": label_ts_ms,
        "weather": weather,
        "scene": scene,
        "timeofday": timeofday,
        "n_car": n_car,
        "n_pedestrian": n_pedestrian,
        "n_truck": n_truck,
        "n_bus": n_bus,
        "n_traffic_light": n_traffic_light,
        "n_traffic_sign": n_traffic_sign,
        "n_tl_red": n_tl_red,
        "n_tl_yellow": n_tl_yellow,
        "n_tl_green": n_tl_green,
        "total_boxed_objects": total_boxed_objects,
        "min_box_area": min_box_area,
        "max_box_area": max_box_area,
        "mean_box_area": mean_box_area,
        "pct_occluded": pct_occluded,
        "pct_truncated": pct_truncated,
    }



# PARSE INFO JSON (TIME-SERIES)
def list_of_dicts_to_df(records, value_cols):
    """
    Convert a sensor stream (list of dicts) into a sorted DataFrame.

    Example input:
      [{"timestamp": ..., "x": ..., "y": ..., "z": ...}, ...]

    Parameters
    records : list[dict]
    value_cols : list[str]
        Value columns to keep, e.g. ["x","y","z"] or ["speed"]

    Returns
    pd.DataFrame or None
        DataFrame with columns: timestamp + requested value columns
    """
    if not isinstance(records, list) or len(records) == 0:
        return None

    rows = []
    for r in records:
        if "timestamp" not in r:
            continue

        row = {"timestamp": float(r["timestamp"])}
        for col in value_cols:
            row[col] = float(r.get(col, np.nan))
        rows.append(row)

    if len(rows) == 0:
        return None

    df = pd.DataFrame(rows).sort_values("timestamp").drop_duplicates("timestamp")
    return df


def resample_stream(df, start_ms, end_ms, target_grid, value_cols):
    """
    Slice a stream to [start_ms, end_ms] and interpolate onto target_grid.

    Parameters
    df : pd.DataFrame
        Must include 'timestamp' and the value columns.
    start_ms, end_ms : float
        Window bounds.
    target_grid : np.ndarray
        New timestamps to interpolate onto.
    value_cols : list[str]
        Columns to interpolate.

    Returns
    np.ndarray or None
        Shape = (len(value_cols), len(target_grid))
    """
    if df is None:
        return None

    window = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)].copy()

    # Need at least 2 points to interpolate
    if len(window) < 2:
        return None

    t = window["timestamp"].to_numpy()

    out = []
    for col in value_cols:
        y = window[col].to_numpy()

        # Fill missing values if any
        y = pd.Series(y).interpolate(limit_direction="both").to_numpy()

        # Interpolate onto fixed grid
        y_interp = np.interp(target_grid, t, y)
        out.append(y_interp)

    return np.stack(out, axis=0)   # (channels, T)


def build_timeseries_tensor(info_json, event_time_ms, pre_sec=3.0, post_sec=3.0, T=256,
                            use_gyro=True, use_speed=True):
    """
    Build a fixed-shape tensor (C, T) from the info JSON around a chosen event time.

    The info JSON contains dense time-series streams like:
      - gyro: x,y,z
      - possibly accelerometer: x,y,z
      - locations: speed, course, lat/lon
      - rideID
      - locations
      - gyro
      - endTime
    so this function is written to handle that structure directly.

    Returns
    np.ndarray or None
        Tensor shape (C, T), or None if data is insufficient.
    """
    start_ms = event_time_ms - pre_sec * 1000.0
    end_ms = event_time_ms + post_sec * 1000.0
    target_grid = np.linspace(start_ms, end_ms, T)

    channel_blocks = []
    channel_names = []

    # Accelerometer if present
    if "accelerometer" in info_json:
        acc_df = list_of_dicts_to_df(info_json["accelerometer"], ["x", "y", "z"])
        acc_rs = resample_stream(acc_df, start_ms, end_ms, target_grid, ["x", "y", "z"])
        if acc_rs is not None:
            channel_blocks.append(acc_rs)
            channel_names.extend(["accel_x", "accel_y", "accel_z"])

    
    # Gyro if present
    if use_gyro and "gyro" in info_json:
        gyro_df = list_of_dicts_to_df(info_json["gyro"], ["x", "y", "z"])
        gyro_rs = resample_stream(gyro_df, start_ms, end_ms, target_grid, ["x", "y", "z"])
        if gyro_rs is not None:
            channel_blocks.append(gyro_rs)
            channel_names.extend(["gyro_x", "gyro_y", "gyro_z"])

    
    # Speed from locations if present
    if use_speed and "locations" in info_json:
        loc_df = list_of_dicts_to_df(info_json["locations"], ["speed"])
        speed_rs = resample_stream(loc_df, start_ms, end_ms, target_grid, ["speed"])
        if speed_rs is not None:
            channel_blocks.append(speed_rs)
            channel_names.extend(["speed"])

    # If nothing usable was found, return None
    if len(channel_blocks) == 0:
        return None, None

    X = np.concatenate(channel_blocks, axis=0).astype(np.float32)  # (C, T)

    # Optional per-sample standardization
    if STANDARDIZE_PER_CHANNEL:
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        X = (X - means) / stds

    return X, channel_names



# MAIN PIPELINE
def main():
    
    # Build file indexes
    labels_index = build_json_index(LABELS_JSON_DIR, id_keys=["name"])
    info_index = build_json_index(INFO_JSON_DIR, id_keys=["rideID", "name"])

    
    # Read event CSV
    events = pd.read_csv(EVENT_CSV)

    # Build labels-context lookup table once
    context_lookup = {}

    for bdd_id, fp in labels_index.items():
        try:
            with open(fp, "r") as f:
                d = json.load(f)
            if "name" in d:
                row = extract_labels_context(d)
                context_lookup[row["BDD_ID"]] = row
        except Exception:
            continue

    
    # Accumulators
    X_ts_list = []
    ctx_rows = []
    meta_rows = []

    missing_labels_json = 0
    missing_info_json = 0
    missing_time_series = 0
    dropped_label_mode = 0

    channel_names_master = None

    
    # Loop through event rows
    for row in events.itertuples(index=False):
        event_id = getattr(row, "EVENT_ID")
        bdd_id = getattr(row, "BDD_ID")
        event_type = getattr(row, "EVENT_TYPE")

        y = map_event_type(event_type, LABEL_MODE)
        if y is None:
            dropped_label_mode += 1
            continue

        
        # Find labels JSON context
        ctx = context_lookup.get(bdd_id)

        if ctx is None:
            missing_labels_json += 1
            # Still continue if info JSON exists, but context will be partial
            ctx = {
                "BDD_ID": bdd_id,
                "label_ts_ms": np.nan,
                "weather": "undefined",
                "scene": "undefined",
                "timeofday": "undefined",
                "n_car": np.nan,
                "n_pedestrian": np.nan,
                "n_truck": np.nan,
                "n_bus": np.nan,
                "n_traffic_light": np.nan,
                "n_traffic_sign": np.nan,
                "n_tl_red": np.nan,
                "n_tl_yellow": np.nan,
                "n_tl_green": np.nan,
                "total_boxed_objects": np.nan,
                "min_box_area": np.nan,
                "max_box_area": np.nan,
                "mean_box_area": np.nan,
                "pct_occluded": np.nan,
                "pct_truncated": np.nan,
            }

        
        # Find info JSON
        info_fp = info_index.get(bdd_id)
        if info_fp is None:
            missing_info_json += 1
            continue

        with open(info_fp, "r") as f:
            info_json = json.load(f)


        # Choose event anchor time
        # Priority:
        # 1) BDD_START from CSV if present
        # 2) labels JSON frame timestamp + ride start time
        # 3) fallback to midpoint of ride
        ride_start_time = info_json.get("startTime", np.nan)
        ride_end_time = info_json.get("endTime", np.nan)

        if hasattr(row, "BDD_START") and not pd.isna(getattr(row, "BDD_START")) and not pd.isna(ride_start_time):
            # BDD_START is assumed to be in seconds from start of ride/clip
            event_time_ms = float(ride_start_time) + 1000.0 * float(getattr(row, "BDD_START"))
            anchor_source = "BDD_START"
        elif not pd.isna(ctx["label_ts_ms"]) and not pd.isna(ride_start_time):
            # Single annotation timestamp from labels JSON, relative to clip start
            event_time_ms = float(ride_start_time) + float(ctx["label_ts_ms"])
            anchor_source = "label_ts_ms"
        elif not pd.isna(ride_start_time) and not pd.isna(ride_end_time):
            # Last-resort fallback: midpoint of ride
            event_time_ms = 0.5 * (float(ride_start_time) + float(ride_end_time))
            anchor_source = "midpoint"
        else:
            missing_time_series += 1
            continue

        
        # Build CNN tensor
        X_ts, channel_names = build_timeseries_tensor(
            info_json=info_json,
            event_time_ms=event_time_ms,
            pre_sec=PRE_SEC,
            post_sec=POST_SEC,
            T=T,
            use_gyro=USE_GYRO,
            use_speed=USE_SPEED,
        )

        if X_ts is None:
            missing_time_series += 1
            continue

        # Check consistency of channel layout
        if channel_names_master is None:
            channel_names_master = channel_names
        else:
            if channel_names != channel_names_master:
                # If different files produce different channels, skip for consistency
                missing_time_series += 1
                continue

        
        # Save outputs
        X_ts_list.append(X_ts)

        # Static context row
        ctx_rows.append({
            "BDD_ID": bdd_id,
            "EVENT_ID": event_id,
            "EVENT_TYPE": int(event_type),
            "y": int(y),
            **ctx
        })

        # Metadata row
        meta_rows.append({
            "BDD_ID": bdd_id,
            "EVENT_ID": event_id,
            "EVENT_TYPE": int(event_type),
            "y": int(y),
            "event_time_ms": event_time_ms,
            "anchor_source": anchor_source,
            "info_json_path": info_fp,
            "label_ts_ms": ctx["label_ts_ms"],
        })

    
    # Finalize outputs
    if len(X_ts_list) == 0:
        raise RuntimeError("No usable time-series examples were built.")

    X_ts = np.stack(X_ts_list, axis=0)  # (N, C, T)
    X_ctx = pd.DataFrame(ctx_rows)
    meta = pd.DataFrame(meta_rows)

    np.save(OUT_TS, X_ts)
    X_ctx.to_csv(OUT_CTX, index=False)
    meta.to_csv(OUT_META, index=False)

    print("Done.")
    print(f"X_ts shape: {X_ts.shape}")   # (N, C, T)
    print(f"Channel names: {channel_names_master}")
    print(f"Saved: {OUT_TS}, {OUT_CTX}, {OUT_META}")
    print()
    print("Diagnostics:")
    print(f"  missing_labels_json: {missing_labels_json}")
    print(f"  missing_info_json:   {missing_info_json}")
    print(f"  missing_time_series: {missing_time_series}")
    print(f"  dropped_label_mode:  {dropped_label_mode}")



# RUN
if __name__ == "__main__":
    main()