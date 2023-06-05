import os
import random
from moviepy.editor import VideoFileClip

# change this for your own path
input_directory = '/path/to/sourcefootage'
output_directory = 'path/to/dataset'
clip_length = 15  # in seconds
quality = 1000  # Quality for the output video
random_seed = 42  # set a specific seed for reproducibility

def split_into_clips(video_path, clip_length, output_directory):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    start_time = 0
    end_time = clip_length
    clip_index = 1
    clips = []
    while start_time < duration:
        clip_name = f'clip_{clip_index}.mp4'
        if end_time > duration:
            end_time = duration
        subclip = clip.subclip(start_time, end_time)
        # Resize the subclip to 480p
        subclip_resized = subclip.resize(height=480)
        clips.append((subclip_resized, clip_name))
        clip_index += 1
        start_time = end_time
        end_time += clip_length
    return clips

# Prepare dataset directories
dataset_splits = ['train', 'test', 'validation']
for split in dataset_splits:
    dir_path = os.path.join(output_directory, split)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Generate clips and split them
all_clips = []
for video_file in os.listdir(input_directory):
    video_path = os.path.join(input_directory, video_file)
    all_clips.extend(split_into_clips(video_path, clip_length, output_directory))

# Set random seed for reproducibility
random.seed(random_seed)
# Shuffle all clips
random.shuffle(all_clips)

# Split the clips into train, test, and validation sets
train_split = int(len(all_clips) * 0.8)
test_split = train_split + int(len(all_clips) * 0.1)
train_clips = all_clips[:train_split]
test_clips = all_clips[train_split:test_split]
validation_clips = all_clips[test_split:]

# Write clips to their respective directories
for split, clips in zip(dataset_splits, [train_clips, test_clips, validation_clips]):
    for clip, clip_name in clips:
        output_path = os.path.join(output_directory, split, clip_name)
        clip.write_videofile(output_path, bitrate=str(quality)+'k', codec='mpeg4')
