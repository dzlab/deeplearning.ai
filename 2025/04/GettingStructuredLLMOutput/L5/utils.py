from transformers import AutoTokenizer
from PIL import Image

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# This function applies a simple chat template to the prompt
# Move this to utils
def template(prompt: str, 
             system_prompt: str = "You are a helpful assistant.") -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, 
         {"role": "user", "content": prompt}],
        tokenize=False,
        add_bos=True,
        add_generation_prompt=True,
    )

def load_and_resize_image(image_path, max_size=1024):
    """
    Load and resize an image while maintaining aspect ratio

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) of the output image

    Returns:
        PIL Image: Resized image
    """
    image = Image.open(image_path)

    # Get current dimensions
    width, height = image.size

    # Calculate scaling factor
    scale = min(max_size / width, max_size / height)

    # Only resize if image is larger than max_size
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image

DEFAULT_BASE_PROMPT="Is this a hotdog or not a hotdog"
def get_messages(image, base_prompt=DEFAULT_BASE_PROMPT):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    # The image is provided as a PIL Image object
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": base_prompt
                },
            ],
        }
    ]
    return messages
