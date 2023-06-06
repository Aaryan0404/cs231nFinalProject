import os
import requests
from bs4 import BeautifulSoup

base_url = 'https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/'
output_dir = '/path/to/output/'  # replace this with the actual path

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

def scrape_directory(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href.endswith('/') and href != '../':
            # This is a subdirectory, scrape it recursively
            scrape_directory(url + href)
        elif any(href.endswith(ext) for ext in ['.mp4', '.avi', '.mkv']):
            # This is a video file, download it
            download_file(url + href, os.path.join(output_dir, href))

scrape_directory(base_url)
