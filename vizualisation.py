import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image_with_pixels_over_threshold(image, threshold=20):
    """
    Display a 2D image with pixels above a threshold marked in red.

    Parameters
    ----------
    image : np.ndarray
        2D image (height, width)
    threshold : float
        Threshold used to detect pixels
    """
    # Find the coordinates of pixels > threshold
    coords = np.where(image > threshold)  # (rows, cols)

    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.scatter(coords[1], coords[0], color='red', s=5, marker='x', label=f'>{threshold}')
    plt.title(f'Pixels > {threshold}')
    plt.axis('off')
    plt.legend()
    plt.show()

    # Print the number of pixels above the threshold
    print(f"Number of pixels > {threshold}: {len(coords[0])}")


def plot_6images(data, indices, threshold=None, show_threshold=True, figsize=(12, 6), cmap='gray'):
    """
    Display images from a 3D array in a 2x3 grid.
    Optionally highlight pixels above a given threshold in red.

    Parameters
    ----------
    data : np.ndarray
        3D array of shape (n_images, width, height).
    indices : list
        List of 6 image indices to display.
    threshold : float, optional
        Threshold above which pixels are marked (None = no threshold).
    show_threshold : bool, optional
        True to show pixels above the threshold, False to ignore (default True).
    figsize : tuple, optional
        Size of the matplotlib figure (default (12, 6)).
    cmap : str, optional
        Colormap to use for displaying the images (default 'gray').
    """
    if len(indices) != 6:
        raise ValueError("La liste indices doit contenir exactement 6 éléments pour un affichage 2x3.")

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    for ax, idx in zip(axes, indices):
        image = data[idx, :, :]
        ax.imshow(image, cmap=cmap)

        if show_threshold and threshold is not None:
            coords = np.where(image > threshold)
            ax.scatter(coords[1], coords[0], color='red', s=3, marker='.')

        ax.set_title(f"Image {idx}", fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def create_video_from_images(images, output_path='output_video.mp4', fps=5):
    """
    Create a video file from the image sequence
    
    Parameters:
    images: numpy array of shape (n_images, height, width)
    output_path: path to save the video file
    fps: frames per second (speed of the video)
    """
    # Get image dimensions
    n_images, height, width = images.shape
    
    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    print(f"Creating video with {n_images} frames at {fps} FPS...")
    
    for i in range(n_images):
        # Get current image and convert to float32
        current_image = images[i, :, :].astype(np.float32)
        
        # Normalize image to 0-255 range
        img_normalized = cv2.normalize(current_image, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_normalized.astype(np.uint8)
        
        # Write frame to video file
        out.write(img_uint8)
        
        # Show progress
        if (i + 1) % 5 == 0 or i == 0 or i == n_images - 1:
            print(f"Processed frame {i+1}/{n_images}")
    
    # Release the video writer
    out.release()
    print(f"Video successfully saved as: {output_path}")
    print(f"Video dimensions: {width}x{height}, Duration: {n_images/fps:.1f} seconds")

def plot_image_histogram(data, image_index=0, bins=100):
    """
    Plot the histogram of pixel values for a single 2D image within a 3D array.

    Parameters
    ----------
    data : np.ndarray
        3D array of images (N, H, W)
    image_index : int, optional
        Index of the image to plot (default: 0)
    bins : int, optional
        Number of bins for the histogram (default: 100)
    """
    img = data[image_index, :, :]  # select 2D image

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(img.ravel(), bins=bins, color='blue', alpha=0.7)
    ax.set_xlabel('Pixel value')
    ax.set_ylabel('Number of pixels')
    ax.set_title(f'Histogram of pixels - image {image_index}')
    plt.tight_layout()
    plt.show()

def print_image_stats(data, image_index=0):
    """
    Print min, max, mean, median of a 2D image within a 3D array in a clean format.

    Parameters
    ----------
    data : np.ndarray
        3D array of images (N, H, W)
    image_index : int, optional
        Index of the image to analyze (default: 0)
    """
    img = data[image_index, :, :]
    print(f"{'Statistic':<15} {'Value':>10}")
    print("-" * 26)
    print(f"{'Min':<15} {np.min(img):>10.4f}")
    print(f"{'Max':<15} {np.max(img):>10.4f}")
    print(f"{'Mean':<15} {np.mean(img):>10.4f}")
    print(f"{'Median':<15} {np.median(img):>10.4f}")
    print(50*'*')