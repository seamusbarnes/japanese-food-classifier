import os
import subprocess
from fastbook import search_images_ddg, download_url
from fastai.vision.utils import download_images, verify_images, get_image_files, resize_images
from fastai.vision.all import Image
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

def delete_images(extensions, depth=1, verbose=True):
    """
    Deletes files with the given extensions up to a certain max depth from the current folder.
    
    Args:
        extensions (list): List of file extensions to delete (e.g. ['jpg', 'png', 'gif']).
        depth (int): Maximum recursion depth for searching files.
    """
    for ext in extensions:
        command = ['find', '.', '-maxdepth', str(depth), '-type', 'f', '-name', f"*.{ext}", '-delete']
        try:
            subprocess.run(command, check=True)
            if verbose:
                print(f"Deleted all .{ext} files in depth {depth}")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

def download_single_image(term, attempts=10, view=True):
    """
    Searches DuckDuckGo for images matching 'term' and downloads the first image that works.
    
    Args:
        term (str): Search term (e.g. 'cat').
        attempts (int): Number of download attempts before giving up.
        view (bool): If True, displays the downloaded image as a thumbnail.
    
    Returns:
        dest (str): Path to the downloaded image.
    """
    dest = f'{term}.jpg'
    urls = search_images_ddg(term, max_images=10)
    for i in range(attempts):
        try:
            download_url(urls[i], dest, show_progress=False)
            if view:
                image = Image.open(dest)
                # image.show()
                display(image.to_thumb(256, 256))
            break
        except Exception as e:
            print(f'Error on attempt {i+1}: {e}')
    return dest

def download_dataset(terms, subfolder, n_images=200, force=False, verbose=True):
    """
    Creates a dataset by searching DuckDuckGo for a list of terms and downloading 
    images for each term. Images are resized, and any corrupted images are removed.

    Args:
        terms (list): List of search terms (e.g. ['sushi', 'ramen']).
        subfolder (str): Name of subfolder where images will be saved.
        n_images (int): Number of images to attempt to download for each search term.
        force (bool): If False, does not download again if the path already exists.
        verbose (bool): Print progress if True.

    Returns:
        path (Path): Path to the folder containing the downloaded images.
    """
    path = Path(os.path.join('datasets', subfolder))

    if path.exists() and not force:
        return path
    
    for term in terms:
        t0 = datetime.now()
        dest = path/term
        dest.mkdir(exist_ok=True, parents=True)
        urls = search_images_ddg(term, max_images=n_images)
        download_images(dest, urls=urls[:])
        resize_images(dest, max_size=400, dest=dest)
        t1 = datetime.now()
        if verbose:
            print(f'{term} images downloaded in {(t1 - t0).total_seconds():.2f} s')
    
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)

    return path

def inference_new_image(term, learner, display_result=True):
    """
    Downloads a new image by searching DuckDuckGo for 'term' and uses the given 
    learner to make a prediction.

    Args:
        term (str): Search term to find an image for inference.
        learner (Learner): A trained fastai learner.
        display_result (bool): If True, displays the image and prints the prediction.

    Returns:
        cat (str): Predicted category.
        cat_idx (int): Index of the predicted category in the probability tensor.
        probs (Tensor): Probability tensor for all categories.
    """
    image_dest = download_single_image(term, view=False)
    cat, cat_idx, probs = learner.predict(image_dest)
    
    if display_result:
        img = Image.open(image_dest)
        img.show()
        display(img.to_thumb(256, 256))
        print(f'Category: {cat}; Prob: {probs[cat_idx]*100:.2f}%')

    return cat, cat_idx, probs

def plot_cat_probabilities(probs, learner):
    """
    Plots a bar chart of all categories vs. the probability predicted by the model.
    
    Args:
        probs (Tensor): Probability tensor of shape (n_categories,).
        learner (Learner): Trained fastai learner, used to fetch category names.
    """
    categories = learner.dls.vocab

    sorted_indices = sorted(range(len(probs)), key=lambda x: probs[x], reverse=True)
    sorted_categories = [categories[idx] for idx in sorted_indices]
    sorted_probs = [probs[idx].item() for idx in sorted_indices]

    plt.figure(figsize=(10, 8))
    plt.bar(sorted_categories, sorted_probs, color='skyblue')
    plt.xlabel('Categories')
    plt.ylabel('Probability')
    plt.title('Predicted Category Probabilities')
    plt.xticks(rotation=45, ha="right")
    plt.show()
