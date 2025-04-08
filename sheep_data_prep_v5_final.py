import os
import subprocess
import random
import shutil
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

# Set up paths
BASE_DIR = Path("C:/Users/Jace/Datasets/Sheep_Videos")
ZIP_FILES = [
    BASE_DIR / "Standing_Walking_Misc_Sheep_Classes.zip",
    BASE_DIR / "Grazing_Running_Sitting_Sheep_Classes.zip",
]
EXTRACT_DIR = BASE_DIR / "raw_data"
SEVEN_ZIP = Path("C:/Program Files/7-Zip/7z.exe")

# Map real folder names to standard class labels
CLASS_FOLDER_MAP = {
    "Grazing": "Grazing",
    "Running": "Running",
    "Sitting": "Sitting",
    "Standing": "Standing",
    "Walking": "Walking",
    "Extra Activities (Noise & Misc.)": "Extra Activities"
}
CLASSES = list(CLASS_FOLDER_MAP.values())
SPLITS = ["train", "val", "test"]

# Output directories for different resolutions
PREP_DIRS = {
    224: BASE_DIR / "Data_Preparation_224",
    112: BASE_DIR / "Data_Preparation_112"
}

def extract_archives(skip=False):
    if skip:
        print("[INFO] Skipping extraction.")
        return
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    for zip_file in ZIP_FILES:
        print(f"[INFO] Extracting {zip_file.name}")
        subprocess.run([str(SEVEN_ZIP), "x", str(zip_file), f"-o{str(EXTRACT_DIR)}", "-y"])
    print("[INFO] Extraction complete.")

def compress_videos(skip=False):
    if skip:
        print("[INFO] Skipping compression.")
        return

    video_files = list(EXTRACT_DIR.rglob("*.MOV")) + list(EXTRACT_DIR.rglob("*.mp4"))
    print(f"[INFO] Found {len(video_files)} video files to process.")
    if len(video_files) != 416:
        print(f"[WARN] Expected 416 videos but found {len(video_files)}")

    for size in PREP_DIRS:
        for cls in CLASSES:
            (PREP_DIRS[size] / "compressed_videos" / cls).mkdir(parents=True, exist_ok=True)

    for idx, vid in enumerate(video_files, start=1):
        raw_class_folder = vid.parent.name
        cls = CLASS_FOLDER_MAP.get(raw_class_folder, "Extra Activities")

        for size, base_dir in PREP_DIRS.items():
            output_class_dir = base_dir / "compressed_videos" / cls
            output_file = output_class_dir / f"{vid.stem}.mp4"
            if output_file.exists():
                output_file.unlink()
            print(f"[INFO] ({idx}/{len(video_files)}) Compressing {vid.name} to {size}px")
            subprocess.run([
                "ffmpeg", "-i", str(vid),
                "-vf", f"scale='if(gt(iw,ih),{size},-2)':'if(gt(iw,ih),-2,{size})'",
                "-c:v", "libx264", "-preset", "faster", "-crf", "23"
            ] + ([] if args.keep_audio else ["-an"]) + [str(output_file)])
    total_original_size = sum(f.stat().st_size for f in video_files)
    for size, base_dir in PREP_DIRS.items():
        compressed_files = list((base_dir / "compressed_videos").rglob("*.mp4"))
        total_compressed_size = sum(f.stat().st_size for f in compressed_files)
        print(f"[INFO] Compressed size at {size}px: {total_compressed_size / 1e9:.2f} GB")
    print(f"[INFO] Original size: {total_original_size / 1e9:.2f} GB")
    print("[INFO] Compression complete for all sizes.")

def split_data(test_ratio=0.2, val_ratio=0.1):
    print("[INFO] Splitting unique videos into train/val/test...")
    for prep_dir in PREP_DIRS.values():
        COMPRESSED_DIR = prep_dir / "compressed_videos"
        OUTPUT_DIR = prep_dir / "data"
        for split in SPLITS:
            for cls in CLASSES:
                (OUTPUT_DIR / "videos" / split / cls).mkdir(parents=True, exist_ok=True)

        all_videos = list(COMPRESSED_DIR.rglob("*.mp4"))
        video_map = {}
        skipped_duplicates = []
        for video in all_videos:
            key = f"{video.parent.name}_{video.name}"
            if key not in video_map:
                video_map[key] = video
            else:
                skipped_duplicates.append(video)

        unique_videos = list(video_map.values())
        random.shuffle(unique_videos)
        total = len(unique_videos)
        n_test = int(total * test_ratio)
        n_val = int(total * val_ratio)
        test_videos = unique_videos[:n_test]
        val_videos = unique_videos[n_test:n_test + n_val]
        train_videos = unique_videos[n_test + n_val:]

        split_assignments = {
            "train": train_videos,
            "val": val_videos,
            "test": test_videos
        }

        for split, videos in split_assignments.items():
            for video in videos:
                raw_class = video.parent.name
                cls = CLASS_FOLDER_MAP.get(raw_class, "Extra Activities")
                dest = OUTPUT_DIR / "videos" / split / cls / video.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(video, dest)

        print("[INFO] Split complete: {} train, {} val, {} test".format(len(train_videos), len(val_videos), len(test_videos)))
        print(f"[INFO] Total unique videos: {len(unique_videos)}")
        print(f"[INFO] Skipped due to duplication: {len(skipped_duplicates)}")
        if skipped_duplicates:
            print("[INFO] Example skipped duplicates:")
            for dup in skipped_duplicates[:5]:
                print(f"  - {dup}")

def extract_frames(frames_per_video=5):
    print(f"[INFO] Extracting {frames_per_video} random frame(s) per video...")
    for prep_dir in PREP_DIRS.values():
        OUTPUT_DIR = prep_dir / "data"
        FRAMES_DIR = prep_dir / "frames"
        for split in SPLITS:
            for cls in CLASSES:
                input_dir = OUTPUT_DIR / "videos" / split / cls
                output_dir = FRAMES_DIR / split / cls
                output_dir.mkdir(parents=True, exist_ok=True)
                for video_file in tqdm(list(input_dir.glob("*.mp4")), desc=f"{split}/{cls}"):
                    cap = cv2.VideoCapture(str(video_file))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames <= 0:
                        print(f"[ERROR] Cannot read frames from {video_file}")
                        continue
                    chosen_frames = sorted(random.sample(range(total_frames), min(frames_per_video, total_frames)))
                    count = 0
                    for i in range(total_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if i in chosen_frames:
                            frame_path = output_dir / f"{video_file.stem}_frame{count}.jpg"
                            target_size = 112 if '112' in str(prep_dir) else 224
                            resized = cv2.resize(frame, (target_size, target_size))
                            cv2.imwrite(str(frame_path), resized)
                            count += 1
                    cap.release()
    print("[INFO] Frame extraction complete.")

def run_tests():
    print("[TEST] Running data integrity checks...")
    for size, prep_dir in PREP_DIRS.items():
        print(f"[CHECK] Resolution: {size}px")
        compressed = list((prep_dir / "compressed_videos").rglob("*.mp4"))
        print(f"  - Compressed videos: {len(compressed)}")
        if len(compressed) != 416:
            print("  [WARNING] Compressed video count is not 416!")

        output_dir = prep_dir / "data" / "videos"
        for split in SPLITS:
            for cls in CLASSES:
                folder = output_dir / split / cls
                count = len(list(folder.glob("*.mp4")))
                print(f"  - {split}/{cls}: {count} videos")

        frame_dir = prep_dir / "frames"
        for split in SPLITS:
            for cls in CLASSES:
                img_folder = frame_dir / split / cls
                count = len(list(img_folder.glob("*.jpg")))
                print(f"  - {split}/{cls}: {count} frames")
    print("[TEST] Folder count checks complete.")

def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-audio", action="store_true", help="Preserve audio tracks during compression")
    parser.add_argument("--skip-extract", action="store_true", help="Skip zip extraction")
    parser.add_argument("--skip-compress", action="store_true", help="Skip compression")
    parser.add_argument("--skip-split", action="store_true", help="Skip train/val/test split")
    parser.add_argument("--frames-per-video", type=int, default=5, help="Number of frames to extract per video")
    args = parser.parse_args()

    extract_archives(skip=args.skip_extract)
    compress_videos(skip=args.skip_compress)
    if not args.skip_split:
        split_data()
    extract_frames(frames_per_video=args.frames_per_video)
    run_tests()

if __name__ == "__main__":
    main()
