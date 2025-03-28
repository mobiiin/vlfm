from typing import Any, Optional
import numpy as np
import torch
from PIL import Image
import base64
from io import BytesIO
import re
import requests
import json
import cv2

from server_wrapper import ServerMixin, host_model, str_to_image

# from vlfm.utils.frame_saver import get_last_frames

try:
    from transformers import AutoModel, AutoTokenizer
except ModuleNotFoundError:
    print("Could not import transformers. This is OK if you are only using the client.")

class VLMModel:
    """Vision-Language Model for indoor navigation."""

    def __init__(
        self,
        model_name: str = "Efficient-Large-Model/VILA1.5-3b",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        self.model.to(device)
        self.device = device

    def process_input(self, image1: np.ndarray, image2: np.ndarray, prompt: str, replace_word: str = "chair") -> tuple:
        """
        Process the images and text prompt using the model.

        Args:
            image1 (numpy.ndarray): The first input image as a numpy array.
            image2 (numpy.ndarray): The second input image as a numpy array.
            prompt (str): The text prompt for the model.
            replace_word (str): The word to replace "couch" with. Defaults to "chair".

        Returns:
            tuple: A tuple containing the model's response and a dictionary of action scores.
        """
        # Replace "couch" (case-insensitive) with the specified word in the prompt
        updated_prompt = re.sub(r"couch", replace_word, prompt, flags=re.IGNORECASE)

        pil_img1 = Image.fromarray(image1)
        pil_img2 = Image.fromarray(image2)

        # Preprocess images (if required by the model)
        # Add your preprocessing steps here

        # Combine the prompt and images
        inputs = self.tokenizer(updated_prompt, return_tensors="pt").to(self.device)
        # Add image embeddings to the inputs if required by the model

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=300, temperature=1)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract action scores using regex
        pattern = r"(Go forward|Go backward|Turn right|Turn left): (\d+(\.\d+)?)"
        matches = re.findall(pattern, response)
        action_scores = {action: float(score) for action, score, _ in matches}

        # Clear CUDA cache
        torch.cuda.empty_cache()

        return response, action_scores


class VLMModelClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/vlm"

    def process_input(self, image1: np.ndarray, image2: np.ndarray, prompt: str, replace_word: str = "chair") -> tuple:
        """
        Send the images and text prompt to the server and get the model's response.

        Args:
            image1 (numpy.ndarray): The first input image as a numpy array.
            image2 (numpy.ndarray): The second input image as a numpy array.
            prompt (str): The text prompt for the model.
            replace_word (str): The word to replace "couch" with. Defaults to "chair".

        Returns:
            tuple: A tuple containing the model's response and a dictionary of action scores.
        """
        try:
            response = self.send_request(self.url, image1=image1, image2=image2, prompt=prompt, replace_word=replace_word)
            return response["response"], response["action_scores"]
        except Exception as e:
            print(f"Error processing input: {e}")
            return "", {}  # Return empty response and action scores

    def send_request(self, url: str, **kwargs) -> dict:
        """
        Send a request to the server with the images, prompt, and replace_word.

        Args:
            url (str): The server URL.
            **kwargs: Keyword arguments including 'image1', 'image2', 'prompt', and 'replace_word'.

        Returns:
            dict: The server's response containing the model's response and action scores.
        """
        # Convert the images to base64-encoded strings
        image1 = kwargs.get("image1")
        image2 = kwargs.get("image2")
        prompt = kwargs.get("prompt")
        replace_word = kwargs.get("replace_word")

        pil_img1 = Image.fromarray(image1)
        pil_img2 = Image.fromarray(image2)

        buffered1 = BytesIO()
        buffered2 = BytesIO()

        pil_img1.save(buffered1, format="PNG")
        pil_img2.save(buffered2, format="PNG")

        img_base64_1 = base64.b64encode(buffered1.getvalue()).decode("utf-8")
        img_base64_2 = base64.b64encode(buffered2.getvalue()).decode("utf-8")

        # Prepare the JSON payload
        payload = {
            "image1": img_base64_1,
            "image2": img_base64_2,
            "txt": prompt,
            "replace_word": replace_word,  # Add replace_word to the payload
        }

        # Set the headers to indicate JSON content
        headers = {
            "Content-Type": "application/json"
        }

        # Send the request to the server
        response = requests.post(url, data=json.dumps(payload), headers=headers)

        # Debugging: Print the raw response and status code
        print("Status Code:", response.status_code)
        print("Raw Response:", response.text)

        # Check if the response is valid JSON
        try:
            return response.json()
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print("Response is not valid JSON.")
            return {"response": "", "action_scores": {}}


if __name__ == "__main__":
    # Initialize the model
    print("Loading model...")
    try:
        vlm = VLMModel(model_name="Efficient-Large-Model/VILA1.5-3b")
        print("Model loaded successfully!")
        print(f"Model is using device: {vlm.device}")
    except Exception as e:
        print(f"Error loading model: {e}")