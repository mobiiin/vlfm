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

from .server_wrapper import ServerMixin, host_model, str_to_image

from vlfm.utils.frame_saver import get_last_frames

try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
except ModuleNotFoundError:
    print("Could not import transformers. This is OK if you are only using the client.")


class VLMModel:
    """Vision-Language Model for indoor navigation."""

    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        # Load model and processor
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        self.model.to(device)
        self.device = device

    def process_input(self, image: np.ndarray, prompt: str, replace_word: str = "chair") -> tuple:
        """
        Process the image and text prompt using the model.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            prompt (str): The text prompt for the model.
            replace_word (str): The word to replace "couch" with. Defaults to "chair".

        Returns:
            tuple: A tuple containing the model's response and a dictionary of action scores.
        """
        # Replace "couch" (case-insensitive) with the specified word in the prompt
        updated_prompt = re.sub(r"couch", replace_word, prompt, flags=re.IGNORECASE)

        pil_img = Image.fromarray(image)
         
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": updated_prompt},
                    {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(pil_img, text=prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=300, temperature=1)
            response = self.processor.decode(output[0], skip_special_tokens=True)

        # Extract action scores using regex
        pattern = r"(Go forward|Go backward|Turn right|Turn left): (\d+(\.\d+)?)"
        matches = re.findall(pattern, response)
        action_scores = {action: float(score) for action, score, _ in matches}

        return response, action_scores


class VLMModelClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/vlm"

    def process_input(self, image: np.ndarray, prompt: str, replace_word: str = "chair") -> tuple:
        """
        Send the image and text prompt to the server and get the model's response.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            prompt (str): The text prompt for the model.
            replace_word (str): The word to replace "couch" with. Defaults to "chair".

        Returns:
            tuple: A tuple containing the model's response and a dictionary of action scores.
        """
        # getting obstacle maps
        # topdown_obstacle_frames = get_last_frames()
        # if frames is not None: 
        #     cv2.imwrite("_topdownmappp.png", frames[0])  
        # print(f"VLMModelClient.process_input: {image.shape}, {prompt}, replace_word={replace_word}")
        response = send_request(self.url, image=image, prompt=prompt, replace_word=replace_word)
        print("VLM Model Response:", response)
        return response["response"], response["action_scores"]


def send_request(url: str, **kwargs) -> dict:
    """
    Send a request to the server with the image, prompt, and replace_word.

    Args:
        url (str): The server URL.
        **kwargs: Keyword arguments including 'image', 'prompt', and 'replace_word'.

    Returns:
        dict: The server's response containing the model's response and action scores.
    """
    # Convert the image to a base64-encoded string
    image = kwargs.get("image")
    prompt = kwargs.get("prompt")
    replace_word = kwargs.get("replace_word")

    pil_img = Image.fromarray(image)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Prepare the JSON payload
    payload = {
        "image": img_base64,
        "txt": prompt,
        "replace_word": replace_word,  # Add replace_word to the payload
    }

    # Set the headers to indicate JSON content
    headers = {
        "Content-Type": "application/json"
    }

    # Send the request to the server
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    # Print the raw response for debugging
    # print("Raw response:", response.text)
    
    # Return the JSON response
    return response.json()


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
            replace_word = payload.get("replace_word", "chair")  # Default to "chair" if not provided
            response, action_scores = self.process_input(image, prompt, replace_word=replace_word)
            return {"response": response, "action_scores": action_scores}

    vlm = VLMModelServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(vlm, name="vlm", port=args.port)