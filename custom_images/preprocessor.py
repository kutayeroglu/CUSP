from PIL import Image
import argparse

def convert_jpg_to_rgb(img_name: str):
    try:
        img = Image.open(f"{img_name}.jpg").convert("RGB")
    except FileNotFoundError:
        img = Image.open(f"{img_name}.jpeg").convert("RGB")
    img.save(f"{img_name}.png")

def reduce_channels(img_name: str):
    img = Image.open(f"{img_name}.png")
    if img.mode == "RGBA":
        img = img.convert("RGB")
        img.save(f"{img_name}.png")
    elif img.mode == "RGB":
        print(f"{img_name} is already in RGB format.")
    else:
        print(f"Unsupported image mode: {img.mode}. Only RGB and RGBA are supported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing tool")
    parser.add_argument("--func", required=True, choices=["convert", "reduce"], 
                      help="Function to execute: 'convert' or 'reduce'")
    parser.add_argument("--name", required=True, 
                      help="Base name of the image file (without extension)")
    
    args = parser.parse_args()
    
    function_mapping = {
        "convert": convert_jpg_to_rgb,
        "reduce": reduce_channels
    }
    
    # Call the selected function with the provided name
    function_mapping[args.func](args.name)