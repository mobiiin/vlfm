from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import json
import torch
import re

# Import your VLMModel implementation
from your_module import VLMModel  # Replace 'your_module' with the actual module name

class PromptEngineer:
    """
    A class to handle prompt engineering with the VLMModel.
    It maintains conversation history, manages token limits, and generates context-aware prompts.
    """

    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf", device: Optional[Any] = None, max_tokens: int = 4096):
        """
        Initialize the PromptEngineer with the VLMModel.

        Args:
            model_name (str): Name of the model to load.
            device (Optional[Any]): Device to run the model on (e.g., "cuda" or "cpu").
            max_tokens (int): Maximum token limit for the model's input.
        """
        self.model = VLMModel(model_name=model_name, device=device)
        self.conversation_history: List[Dict[str, str]] = []  # Stores the conversation history
        self.max_tokens = max_tokens  # Maximum token limit for the model
        self.initial_prompt: Optional[str] = None  # Stores the initial prompt (main objective)

    def add_to_history(self, prompt: str, response: str) -> None:
        """
        Add a prompt and its corresponding response to the conversation history.
        If the token limit is exceeded, remove the second prompt and response (but keep the initial prompt).

        Args:
            prompt (str): The user's prompt.
            response (str): The model's response.
        """
        self.conversation_history.append({"prompt": prompt, "response": response})

        # Check if the token limit is exceeded
        while self._calculate_token_usage() > self.max_tokens and len(self.conversation_history) > 1:
            # Remove the second prompt and response (keep the initial prompt)
            self.conversation_history.pop(1)

    def _calculate_token_usage(self) -> int:
        """
        Calculate the total token usage of the conversation history.

        Returns:
            int: Estimated token count.
        """
        # Use the model's processor to tokenize the conversation history
        total_tokens = 0
        for entry in self.conversation_history:
            # Tokenize the prompt and response
            prompt_tokens = self.model.processor.tokenizer(entry["prompt"], return_tensors="pt").input_ids
            response_tokens = self.model.processor.tokenizer(entry["response"], return_tensors="pt").input_ids
            total_tokens += prompt_tokens.shape[1] + response_tokens.shape[1]
        return total_tokens

    def generate_prompt(self) -> str:
        """
        Generate a new prompt based on the model's previous response and the robot's current location.

        Returns:
            str: A new prompt to send to the model.
        """
        if not self.conversation_history:
            # Initial prompt if no history exists
            self.initial_prompt = "Describe the scene in front of you and identify your current location."
            return self.initial_prompt

        # Parse the model's last response to determine the robot's location
        last_response = self.conversation_history[-1]["response"]
        if "corridor" in last_response.lower():
            return "You are in a corridor. What do you see ahead? Try to exit the corridor and describe where it leads."
        elif "room" in last_response.lower():
            return "You are in a room. Describe the room and look for exits."
        else:
            return "Continue exploring and describe what you see."

    def process_image_and_prompt(self, image: np.ndarray, prompt: str) -> Tuple[str, Dict[str, float]]:
        """
        Process an image and prompt using the VLMModel.

        Args:
            image (np.ndarray): The input image as a numpy array.
            prompt (str): The text prompt for the model.

        Returns:
            Tuple[str, Dict[str, float]]: The model's response and action scores.
        """
        response, action_scores = self.model.process_input(image, prompt)
        self.add_to_history(prompt, response)
        return response, action_scores

    def save_conversation_history(self, file_path: str) -> None:
        """
        Save the conversation history to a JSON file.

        Args:
            file_path (str): Path to the output JSON file.
        """
        with open(file_path, "w") as f:
            json.dump(self.conversation_history, f, indent=4)


# Example Usage
if __name__ == "__main__":
    # Initialize the PromptEngineer
    prompt_engineer = PromptEngineer()

    # Load an example image (replace with your actual image)
    image = np.zeros((224, 224, 3), dtype=np.uint8)  # Placeholder image

    # Start the conversation
    initial_prompt = prompt_engineer.generate_prompt()
    print(f"Initial Prompt: {initial_prompt}")

    # Simulate a multi-turn conversation
    for turn in range(5):  # Adjust the number of turns as needed
        response, action_scores = prompt_engineer.process_image_and_prompt(image, initial_prompt)
        print(f"Turn {turn + 1} - Model Response: {response}")

        # Generate the next prompt based on the model's response
        initial_prompt = prompt_engineer.generate_prompt()
        print(f"Next Prompt: {initial_prompt}")

    # Save the conversation history
    prompt_engineer.save_conversation_history("conversation_history.json")