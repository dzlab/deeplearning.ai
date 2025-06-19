import os

from dotenv import find_dotenv, load_dotenv


def load_env():
    _ = load_dotenv(find_dotenv())

def get_llama_api_key():
    load_env()
    llama_api_key = os.getenv("LLAMA_API_KEY")
    return llama_api_key

def get_llama_base_url():
    load_env()
    llama_base_url = os.getenv("LLAMA_BASE_URL")
    return llama_base_url

def get_together_api_key():
    load_env()
    together_api_key = os.getenv("TOGETHER_API_KEY")
    return together_api_key

def get_hf_access_token():
    load_env()
    github_access_token = os.getenv("HF_ACCESS_TOKEN")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def plot_tiled_image(width, height, tile_size, patch_size):
    # Create a new image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Divide the image into tiles
    for i in range(0, width, tile_size):
        for j in range(0, height, tile_size):
            # Draw a rectangle for each tile
            draw.rectangle((i, j, i + tile_size, j + tile_size),
                           outline='black')

            # Divide each tile into patches
            for x in range(i, i + tile_size, patch_size):
                for y in range(j, j + tile_size, patch_size):
                    # Draw a rectangle for each patch
                    draw.rectangle((x, y, x + patch_size, y + patch_size),
                                   outline='gray')

    # Add separator lines with text
    font = ImageFont.load_default()
    font = font.font_variant(size=28)

    for i in range(tile_size, width, tile_size):
        draw.line((i, 0, i, height), fill='black')
        draw.text((i - 150, height // 5), '<tile_x_separator|>', font=font,
                  fill='blue')
        draw.text((i - 150, height // 1.5), '<tile_x_separator|>', font=font,
                  fill='blue')
    for j in range(tile_size, height, tile_size):
        draw.line((0, j, width, j), fill='black')
        draw.text((width // 2, j - 20), '<tile_y_separator|>', font=font,
                  fill='blue')
    draw.line((0, height - 10, width, height - 10), fill='black')
    draw.text((width // 2, height - 40), '<tile_y_separator|>', font=font,
              fill='blue')


    # Add additional texts
    draw.text((10, 10), 'Image Size: 768x768', font=font, fill='black')
    draw.text((10, 40), 'Tile Size: 336x336; # of Tiles: 9', font=font,
              fill='black')
    draw.text((10, 70),
              'Patch Size: 28x28; # of Patches per Tile: 144 (12x12)',
              font=font, fill='black')

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Display the image using matplotlib
    plt.imshow(img_array)
    plt.axis('off')  # Turn off the axis
    plt.show()