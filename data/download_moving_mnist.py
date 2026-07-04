import os
import requests
from tqdm import tqdm

def download_moving_mnist(save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
    filepath = os.path.join(save_dir, "mnist_test_seq.npy")
    
    if os.path.exists(filepath):
        print(f"{filepath} already exists. Skipping download.")
        return filepath
        
    print(f"Downloading Moving MNIST from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1 MB
    
    with open(filepath, 'wb') as file, tqdm(
        total=total_size, unit='iB', unit_scale=True
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)
            
    print(f"Successfully downloaded to {filepath}")
    return filepath

if __name__ == "__main__":
    # Ensure we run relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    download_moving_mnist(data_dir)
