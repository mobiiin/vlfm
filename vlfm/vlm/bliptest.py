import numpy as np
from PIL import Image
import base64
import json
import requests
from io import BytesIO

class BLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        print(f"BLIP2ITMClient.cosine: {image.shape}, {txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])

def send_request(url: str, image: np.ndarray, txt: str) -> dict:
    # Convert the image to a base64-encoded string
    pil_img = Image.fromarray(image)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Prepare the JSON payload
    payload = {
        "image": img_base64,
        "txt": txt
    }

    # Set the headers to indicate JSON content
    headers = {
        "Content-Type": "application/json"
    }

    # Send the request to the server
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    # Print the raw response for debugging
    print("Raw response:", response.text)
    
    # Return the JSON response
    return response.json()

def main():
    # Load an image (replace with your image path)
    image_path = "/home/mhabibp/Pictures/Screenshots/habitat1.png"
    image = np.array(Image.open(image_path))

    # Define the text prompt
    text_prompt = "this is a living room"

    # Create the client and send the request
    client = BLIP2ITMClient()
    similarity_score = client.cosine(image, text_prompt)

    print(f"Cosine Similarity: {similarity_score}")

if __name__ == "__main__":
    main()