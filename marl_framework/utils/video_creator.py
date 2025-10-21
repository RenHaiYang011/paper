import cv2
import os

def create_video_from_images(image_folder: str, output_video_path: str, fps: int = 30):
    """
    Reads images from a folder and compiles them into a video.

    Args:
        image_folder (str): Path to the folder containing images.
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the video.

    Returns:
        None
    """
    # Get list of image files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # Ensure images are in the correct order

    if not images:
        raise ValueError(f"No images found in folder: {image_folder}")

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        raise ValueError(f"Failed to read the first image: {first_image_path}")

    height, width, layers = frame.shape
    video_writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Skipping unreadable image: {image_path}")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    # Example usage
    image_folder = "res/plots"  # Update this path to your image folder
    output_video_path = "res/trajectory_video.mp4"  # Update this path to your desired output video
    create_video_from_images(image_folder, output_video_path)