import numpy as np
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from typing import Optional, Any, List

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

    def process_input(self, images: List[np.ndarray], prompt: str) -> tuple:
        """
        Process the images and text prompt using the model.

        Args:
            images (List[numpy.ndarray]): A list of input images as numpy arrays.
            prompt (str): The text prompt for the model.

        Returns:
            tuple: A tuple containing the model's response and a dictionary of action scores.
        """
        pil_images = [Image.fromarray(image) for image in images]
         
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                    {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(pil_images, text=prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=300, temperature=1)
            response = self.processor.decode(output[0], skip_special_tokens=True)

        return response

# Load your images
image_path1 = "../../_topdownmappp.png" 
image_path2 = "../../_currentview.png"  
image1 = np.array(Image.open(image_path1))
image2 = np.array(Image.open(image_path2))

# Define your prompt
prompt = '''
explain what you understand from these two images. hint: the second image is the topdown view map of the house.
'''

# Initialize the VLMModel
vlm_model = VLMModel()

# Process the input
response = vlm_model.process_input([image1, image2], prompt)

# Print the results
print("Model Response:", response)