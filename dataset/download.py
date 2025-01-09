import numpy as np
import cv2
import os
from numpy.lib.scimath import sqrt as csqrtv
from huggingface_hub import list_repo_files, snapshot_download
import subprocess
import json

def extract_frame(video_path, output_path, frame_number=500):
    
    # FFmpeg command to extract specific frame
    # -frames:v 1 tells FFmpeg to extract just one frame
    # -vf select="eq(n\,499)" selects the 500th frame (0-based indexing)
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'select=eq(n\,{frame_number-1})',
        '-frames:v', '1',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Saved frame {frame_number} to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frame: {e.stderr.decode()}")
        return False
    
def get_video_info(video_path):
    """Get video duration and fps using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,nb_frames',
        '-of', 'json',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        # Get FPS (might be in fraction form like "30000/1001")
        fps_str = info['streams'][0]['r_frame_rate']
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den
        else:
            fps = float(fps_str)
            
        # Get total frames
        total_frames = int(info['streams'][0]['nb_frames'])
        
        return fps, total_frames
    except subprocess.CalledProcessError:
        return None, None
    

# Define the repository ID and folder path
repo_id = "quchenyuan/360x_dataset_HR"
panoramic_folder = "panoramic/"

# Step 1: List all files in the repository and filter panoramic videos
all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
video_files = [f for f in all_files if f.startswith(panoramic_folder) and f.endswith(".mp4")]

# Print the list of video files
print("Panoramic video files:")
for video in video_files:
    print(video)

# Step 2: Download each video file individually
# Set the custom download path
custom_download_path = "/root/jh/DeepCalib/dataset/360x_dataset_HR"
download_dir = os.path.join(custom_download_path, "datasets--quchenyuan--360x_dataset_HR/snapshots/e8da2eeb18e86784b3913b22ca947e27bda41187")
output_frame_dir = "dump"

done_list = ["panoramic/019cc67f-512f-4b8a-96ef-81f806c86ce1.mp4", "panoramic/02594cd1-c13c-4fbe-a804-f8d8c72ca409.mp4"]

for video_file in video_files:
    if video_file in done_list:
        print(f"Skipping {video_file} (already downloaded)")
        continue  # Skip to the next video file in the loop
    
    print(f"Downloading {video_file}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[video_file],  # Download only the current video file
        cache_dir=custom_download_path,
        # token=""
    )
    print(f"Downloaded {video_file}.")
    
    # Optionally extract the first frame from the video (if needed)
    video_path = os.path.join(download_dir, video_file)
    print(f"video path: {video_path}")
    if os.path.exists(video_path):
        
        start_frame_number = 500
        fps, total_frames = get_video_info(video_path)
        if fps is None :
            fps = 25
            total_frames = 100000
        
        # Calculate frames per minute
        frames_per_minute = int(fps * 60)
        
        # Generate frame numbers: start at 500, then every minute
        frame_nums = list(range(500, total_frames, frames_per_minute))
        
        for frame_num in frame_nums:
            base_name = os.path.basename(video_file)  # Gets e57ae4db-3235-40b2-a563-a3c0639a74d7.mp4
            uuid = os.path.splitext(base_name)[0]     # Removes .mp4
            frame_path = os.path.join(output_frame_dir, f"{uuid}_frame{frame_num}.jpg")
            if extract_frame(video_path, frame_path, frame_num):
                print(f"Saved! {uuid} at {frame_num}th as {frame_path}")
            else:
                print(f"failed! {video_path} at {frame_num}th")
    
    # Remove the real downloaded video to save space
    real_video_path = os.path.realpath(video_path)
    print(f"Removing real file: {real_video_path}")
    os.remove(real_video_path)  # Remove the actual blob file

    # Remove the downloaded video to save space
    print(f"Removing: {video_path}")
    os.remove(video_path)

print("All files processed!")
