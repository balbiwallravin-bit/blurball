import cv2
import os
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

SSIM_THRESHOLD = 0.99


def process_video(video_path, filter=False):
    video_path = os.path.abspath(video_path)

    # --------- Parse video meta ---------
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    parent_dir = os.path.dirname(video_path)
    hour = os.path.basename(parent_dir)
    date = os.path.basename(os.path.dirname(parent_dir))

    segment_name = f"{date}__{hour}__{video_name}"

    # --------- Unified output root ---------
    base_output = "/home/lht/blurtrack/video_maked"
    segment_dir = os.path.join(base_output, segment_name)
    frames_dir = os.path.join(segment_dir, "frames_roi")

    done_flag = Path(frames_dir) / ".done"

    # --------- Reuse if finished before ---------
    if done_flag.exists():
        existing = len(list(Path(frames_dir).glob("*.jpg")))
        print(f"[ROI] Reuse existing frames: {frames_dir} ({existing})")
        return frames_dir

    os.makedirs(frames_dir, exist_ok=True)

    # --------- ROI (16:9 fixed) ---------
    ROI_X0 = 367
    ROI_Y0 = 100
    ROI_X1 = 1760
    ROI_Y1 = 884

    # --------- Open video ---------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None

    prev_frame_gray = None
    unique_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --------- Crop ROI ---------
        frame = frame[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        save_frame = True
        if filter and prev_frame_gray is not None:
            score = ssim(prev_frame_gray, frame_gray)
            save_frame = score < SSIM_THRESHOLD

        if save_frame:
            frame_filename = f"{unique_index:05d}.jpg"
            cv2.imwrite(
                os.path.join(frames_dir, frame_filename),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            unique_index += 1

        prev_frame_gray = frame_gray

    cap.release()

    # --------- Write done flag ---------
    done_flag.write_text("ok\n")

    print(f"[ROI] Saved {unique_index} frames to {frames_dir}")

    return frames_dir