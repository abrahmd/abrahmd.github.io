from PIL import Image
import os

def images_to_gif(image_folder, output_gif, duration=500, loop=0):
    """
    Convert a sequence of images into a GIF.
    
    Args:
        image_folder (str): Path to folder containing images.
        output_gif (str): Output GIF filename (e.g., 'output.gif').
        duration (int): Duration (ms) for each frame.
        loop (int): Number of loops (0 means infinite).
    """
    # Collect images
    images = []
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, file_name)
            img = Image.open(img_path).convert("RGB")
            images.append(img)

    if not images:
        raise ValueError("No images found in folder.")

    # Save as GIF
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )
    print(f"GIF saved as {output_gif}")

images_to_gif('dolly_images', 'dolly_images/dolly.gif')