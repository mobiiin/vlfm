from typing import Any, Optional
import numpy as np
import torch
from PIL import Image
import base64
from io import BytesIO
import re
import requests
import json

from .server_wrapper import ServerMixin, host_model, str_to_image

try:
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
except ModuleNotFoundError:
    print("Could not import transformers. This is OK if you are only using the client.")




class VLMModel:
    """Vision-Language Model for indoor navigation."""

    def __init__(
        self,
        model_name: str = "google/paligemma-3b-mix-224",
        device: Optional[Any] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            revision="bfloat16",
        ).eval()
        self.device = device
        self.dtype = dtype

    def process_input(self, image: np.ndarray, prompt: str) -> str:
        """
        Process the image and text prompt using the model.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            prompt (str): The text prompt for the model.

        Returns:
            str: The model's response.
        """
        pil_img = Image.fromarray(image)

        model_inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt").to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            response = self.processor.decode(generation, skip_special_tokens=True)

        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        return response


class VLMModelClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/vlm"

    def process_input(self, image: np.ndarray, prompt: str) -> str:
        """
        Send the image and text prompt to the server and get the model's response.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            prompt (str): The text prompt for the model.

        Returns:
            str: The model's response.
        """
        try:
            response = self.send_request(self.url, image=image, prompt=prompt)
            return response["response"]
        except Exception as e:
            print(f"Error processing input: {e}")
            return ""  # Return empty response

    def send_request(self, url: str, **kwargs) -> dict:
        """
        Send a request to the server with the image and prompt.

        Args:
            url (str): The server URL.
            **kwargs: Keyword arguments including 'image' and 'prompt'.

        Returns:
            dict: The server's response containing the model's response.
        """
        # Convert the image to a base64-encoded string
        image = kwargs.get("image")
        prompt = kwargs.get("prompt")

        pil_img = Image.fromarray(image)
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Prepare the JSON payload
        payload = {
            "image": img_base64,
            "txt": prompt,
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
            return {"response": ""}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    args = parser.parse_args()

    print("Loading model...")

    class VLMModelServer(ServerMixin, VLMModel):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            prompt = payload["txt"]
            response = self.process_input(image, prompt)
            return {"response": response}

    vlm = VLMModelServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(vlm, name="vlm", port=args.port)