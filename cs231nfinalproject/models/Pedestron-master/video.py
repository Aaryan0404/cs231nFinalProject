import cv2
import os
import argparse
import glob

def generate_video(image_folder, video_name, fps):
    """
    Generate a video out of images located in a directory.

    Parameters:
    - image_folder: str, path to directory with images
    - video_name: str, name of video file to be created
    - fps: int, frames per second
    """

    images = glob.glob(os.path.join(image_folder, '*.jpg'))
    
    # Sort images by index in filename
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create video out of images in a folder.')
    parser.add_argument('--folder', type=str, help='Path to the folder containing images')
    parser.add_argument('--video', type=str, help='Name of the video file to be created')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')

    args = parser.parse_args()
    
    generate_video(args.folder, args.video, args.fps)
