import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

# change this for your own path
input_directory = '/path/to/sourcefootage'
output_directory = 'path/to/dataset'
clip_length = 15  # in seconds
quality = 1000  # Quality for the output video

# Create the output directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def split_into_clips(video_path, clip_length, output_directory):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    start_time = 0
    end_time = clip_length
    clip_index = 1
    while start_time < duration:
        clip_name = os.path.join(output_directory, f'clip_{clip_index}.mp4')
        if end_time > duration:
            end_time = duration
        subclip = clip.subclip(start_time, end_time)
        # Resize the subclip to 480p
        subclip_resized = subclip.resize(height=480)
        subclip_resized.write_videofile(clip_name, bitrate=str(quality)+'k', codec='mpeg4')
        clip_index += 1
        start_time = end_time
        end_time += clip_length

for video_file in os.listdir(input_directory):
    video_path = os.path.join(input_directory, video_file)
    split_into_clips(video_path, clip_length, output_directory)

