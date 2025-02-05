import numpy as np
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from typing import Optional, Any  

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

    def process_input(self, image: np.ndarray, prompt: str) -> tuple:
        """
        Process the image and text prompt using the model.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            prompt (str): The text prompt for the model.
            replace_word (str): The word to replace "couch" with. Defaults to "chair".

        Returns:
            tuple: A tuple containing the model's response and a dictionary of action scores.
        """
        pil_img = Image.fromarray(image)
         
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(pil_img, text=prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=100, temperature=1)
            response = self.processor.decode(output[0], skip_special_tokens=True)

        return response

# Load your image
image_path = "../../_topdownmappp_num.png" 
image = np.array(Image.open(image_path))

# Define your prompt
prompt = '''
this is a top down map. the grey parts are obstacles. 
the yellow circle is a robot exploring the environment. 
what number is closest to the robots location? 
which direction is the robot headed? 
'''

# Initialize the VLMModel
vlm_model = VLMModel()

# Process the input
response = vlm_model.process_input(image, prompt)

# Print the results
print("Model Response:", response)