import cv2
import os

def create_video_from_images(image_folder, video_file, fps):
    # Get all the image file names from the directory
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort the images by filename
    images.sort()
    
    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID' for .avi files
    video = cv2.VideoWriter(video_file, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        
        # Check if the frame is read correctly
        if frame is None:
            print(f"Warning: {image_path} could not be read and is skipped.")
            continue
        
        video.write(frame)

    # Release everything when job is finished
    video.release()
    print(f"Video saved as {video_file}")

# Example usage
image_folder = '/home/ayesha.ishaq/Desktop/3DMOTFormer/open_eval/vis/3363f396bb43405fbdd17d65dc123d4e/pred'
video_file = '3363f396bb43405fbdd17d65dc123d4e_555_pred.mp4'
fps = 2  # frames per second

create_video_from_images(image_folder, video_file, fps)
