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

    def process_input(self, image1: np.ndarray, prompt: str, replace_word: str = "chair", image2: Optional[np.ndarray] = None) -> tuple:
        """
        Process the images and text prompt using the model.

        Args:
            image1 (numpy.ndarray): The first input image as a numpy array.
            image2 (Optional[numpy.ndarray]): The second input image as a numpy array. Defaults to None.
            prompt (str): The text prompt for the model.
            replace_word (str): The word to replace "couch" with. Defaults to "chair".

        Returns:
            tuple: A tuple containing the model's response and a dictionary of action scores.
        """
        # Replace "couch" (case-insensitive) with the specified word in the prompt
        
        updated_prompt = re.sub(r"couch", replace_word, prompt, flags=re.IGNORECASE)

        pil_img1 = Image.fromarray(image1)
        images = [pil_img1]

        if image2 is not None:
            pil_img2 = Image.fromarray(image2)
            images.append(pil_img2)

        # Process the images and prompt
        model_inputs = self.processor(text=updated_prompt, images=images, return_tensors="pt").to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            response = self.processor.decode(generation, skip_special_tokens=True)

        # Extract action scores using regex (if applicable)
        pattern = r"(Go forward|Go backward|Turn right|Turn left): (\d+(\.\d+)?)"
        matches = re.findall(pattern, response)
        action_scores = {action: float(score) for action, score, _ in matches}

        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        return response, action_scores

class VLMModelClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/vlm"

    def process_input(self, image1: np.ndarray, prompt: str, replace_word: str = "chair", image2: Optional[np.ndarray] = None) -> tuple:
        """
        Send the images and text prompt to the server and get the model's response.

        Args:
            image1 (numpy.ndarray): The first input image as a numpy array.
            image2 (Optional[numpy.ndarray]): The second input image as a numpy array. Defaults to None.
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
        images = [pil_img1]

        if image2 is not None:
            pil_img2 = Image.fromarray(image2)
            images.append(pil_img2)

        buffered_images = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            buffered_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

        # Prepare the JSON payload
        payload = {
            "image1": buffered_images[0],
            "txt": prompt,
            "replace_word": replace_word,  # Add replace_word to the payload
        }

        if len(buffered_images) > 1:
            payload["image2"] = buffered_images[1]

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    args = parser.parse_args()

    print("Loading model...")

    class VLMModelServer(ServerMixin, VLMModel):
        def process_payload(self, payload: dict) -> dict:
            try:
                print("Received payload:", payload)  # Debugging
                # Decode base64-encoded images
                image1 = str_to_image(payload["image1"])
                image2 = str_to_image(payload["image2"]) if "image2" in payload else None
                prompt = payload["txt"]
                replace_word = payload.get("replace_word", "chair")  # Default to "chair" if not provided

                print("Images decoded successfully")  # Debugging

                # Process the input
                response, action_scores = self.process_input(image1=image1, image2=image2, prompt=prompt, replace_word=replace_word)
                return {"response": response, "action_scores": action_scores}
            except Exception as e:
                print(f"Error in process_payload: {e}")
                return {"response": f"Error: {str(e)}", "action_scores": {}}

    vlm = VLMModelServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(vlm, name="vlm", port=args.port)