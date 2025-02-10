import re
from typing import List, Dict, Tuple, Optional, Deque, Any
from collections import deque
import numpy as np
from PIL import Image
import json
import os

# Import the VLMModelClient from the provided script
from vlfm.vlm.llava_ask import VLMModelClient

class PromptEngineer:
    """
    A class to handle prompt engineering with the VLMModel.
    It maintains conversation history, manages token limits, and generates context-aware prompts.
    """

    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf", device: Optional[Any] = None, max_tokens: int = 4096):
        """
        Initialize the PromptEngineer with the VLMModelClient.

        Args:
            model_name (str): Name of the model to load.
            device (Optional[Any]): Device to run the model on (e.g., "cuda" or "cpu").
            max_tokens (int): Maximum token limit for the model's input.
        """
        self._vlm_client = VLMModelClient(port=int(os.environ.get("LLAVA_PORT", "12182")))
        self.conversation_history: List[Dict[str, str]] = []  # Stores the conversation history
        self.action_history: Deque[str] = deque(maxlen=10)  # Stores the last 10 actions
        self.max_tokens = max_tokens  # Maximum token limit for the model
        self.initial_prompt: Optional[str] = None  # Stores the initial prompt (main objective)

    def add_to_history(self, prompt: str, response: str, action: str) -> None:
        """
        Add a prompt, response, and action to the conversation and action history.

        Args:
            prompt (str): The user's prompt.
            response (str): The model's response.
            action (str): The action taken by the robot.
        """
        self.conversation_history.append({"prompt": prompt, "response": response})
        self.action_history.append(action)

        # Check if the token limit is exceeded
        while self._calculate_token_usage() > self.max_tokens and len(self.conversation_history) > 1:
            # Remove the second prompt and response (keep the initial prompt)
            self.conversation_history.pop(1)

    def _format_history(self) -> str:
        """
        Format the conversation history and action history into a readable string.

        Returns:
            str: A formatted string containing the conversation and action history.
        """
        history_str = "### Conversation History:\n"
        for i, entry in enumerate(self.conversation_history):
            history_str += f"Turn {i + 1}:\n"
            history_str += f"  - Prompt: {entry['prompt']}\n"
            history_str += f"  - Response: {entry['response']}\n"
        
        history_str += "\n### Action History:\n"
        for i, action in enumerate(self.action_history):
            history_str += f"Turn {i + 1}: {action}\n"
        
        return history_str

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
            prompt_tokens = len(entry["prompt"].split())  # Simple word count as a proxy for tokens
            response_tokens = len(entry["response"].split())  # Simple word count as a proxy for tokens
            total_tokens += prompt_tokens + response_tokens
        return total_tokens

    def detect_loop(self) -> bool:
        """
        Detect if the robot is stuck in a loop based on the action history.

        Returns:
            bool: True if a loop is detected, False otherwise.
        """

        if len(self.action_history) < 4:
            return False  # Not enough actions to detect a loop

        # Check for repeating patterns in the action history
        last_four_actions = list(self.action_history)[-4:]
        if last_four_actions[0] == last_four_actions[2] and last_four_actions[1] == last_four_actions[3]:
            return True  # Loop detected (e.g., "Turn left → Turn right → Turn left → Turn right")

        return False

    def parse_response(self, response: str) -> Dict[str, str]:
        """
        Parse the model's structured response to extract answers to the first three questions.

        Args:
            response (str): The model's response.

        Returns:
            Dict[str, str]: A dictionary containing the parsed answers.
        """
        # Define regex patterns to extract the answers
        part_of_house_pattern = re.compile(r"1\. \*\*Part of the House\*\*: (.+?)\n")
        target_object_found_pattern = re.compile(r"2\. \*\*Can a Target Object Be Found Here\?\*\*: (.+?)\n")
        recommended_action_pattern = re.compile(r"3\. \*\*Recommended Action\*\*: (.+?)\n")

        # Extract the answers using regex
        part_of_house_match = part_of_house_pattern.search(response)
        target_object_found_match = target_object_found_pattern.search(response)
        recommended_action_match = recommended_action_pattern.search(response)

        # Store the parsed answers in a dictionary
        parsed_response = {
            "part_of_house": part_of_house_match.group(1).strip() if part_of_house_match else None,
            "target_object_found": target_object_found_match.group(1).strip() if target_object_found_match else None,
            "recommended_action": recommended_action_match.group(1).strip() if recommended_action_match else None,
        }

        return parsed_response

    def generate_prompt(self, parsed_response: Dict[str, str]) -> str:
        """
        Generate a new prompt based on the parsed answers from the model's response,
        including the conversation and action history.

        Args:
            parsed_response (Dict[str, str]): The parsed answers from the model's response.

        Returns:
            str: A new prompt to send to the model.
        """
        if not self.conversation_history:
            # Initial prompt if no history exists
            self.initial_prompt = ''' 
                You are a robot navigating an indoor environment in search of a couch. 
                The image on the left is your current observation
                You must think step by step and ensure that all parts of your response are consistent. 

                Here are the tasks:
                1. Identify what part of the house we are about to enter (choose from: [bedroom, living room, kitchen, corridor, bathroom]).
                2. Assess whether a couch can realistically be found in this area, based on common sense and the current observation. 
                3. Determine the most logical next action for the robot (choose from: [go forward, go backward, turn right, turn left]). 
                - The chosen action must prioritize exploring areas likely to contain a couch. 
                - Avoid suggesting actions that contradict previous observations (e.g., don't explore a bathroom if couches aren't found there). 
                - If you are in a corridor, continue your path and Try to exit the corridor and describe where it leads. 
                - Make Sure the robot isn't stuck in an action loop
                4. Provide a probability score for each possible action in the following format:
                - Go forward: [Score]
                - Go backward: [Score]
                - Turn right: [Score]
                - Turn left: [Score]

                Each probability score should be a number between 0 and 1, with two decimal places of precision. 
                - A score of 1 means full confidence that the action will lead to finding the couch. 
                - A score of 0 means no confidence. 

                When providing your response, use this structure:
                1. **Part of the House**: [Your answer]
                - Reasoning: [Explain why you think this is the correct part of the house based on the observation and map.]
                2. **Can a Couch Be Found Here?**: [Yes/No]
                - Reasoning: [Explain why or why not.]
                3. **Recommended Action**: [Your action]
                - Reasoning: [Explain why this action is the most logical based on steps 1 and 2.]
                4. **Probability Scores for Each Action**:
                - Go forward: [Score]
                - Go backward: [Score]
                - Turn right: [Score]
                - Turn left: [Score]

                Important: Ensure that the recommended action aligns with the reasoning from steps 1 and 2. If a couch cannot be found in the current area, prioritize moving to areas more likely to contain a couch. 
                '''
            return self.initial_prompt

        # Generate follow-up prompts based on the parsed answers
        # part_of_house = parsed_response["part_of_house"]
        # target_object_found = parsed_response["target_object_found"]
        # recommended_action = parsed_response["recommended_action"]

        # Include the conversation and action history in the prompt
        history_str = self._format_history()

        # if part_of_house and "corridor" in part_of_house.lower():
        #     return f"{history_str}\n You are in a corridor. What do you see ahead? Try to exit the corridor and describe where it leads."
        # elif part_of_house and "room" in part_of_house.lower():
        #     return f"{history_str}\nYou are in a room. Describe the room and look for exits."
        # elif target_object_found and "yes" in target_object_found.lower():
        #     return f"{history_str}\nThe target object is found here. Describe the surroundings of the target object."
        # else:
        return f"{history_str}\nContinue exploring."

    def process_image_and_prompt(self, image: np.ndarray, prompt: str, target_object: str = "chair") -> Tuple[str, Dict[str, float]]:
        """
        Process an image and prompt using the VLMModelClient.

        Args:
            image (np.ndarray): The input image as a numpy array.
            prompt (str): The text prompt for the model.

        Returns:
            Tuple[str, Dict[str, float]]: The model's response and action scores.
        """
        response, action_scores = self._vlm_client.process_input(image, prompt, target_object)

        # Determine the recommended action
        parsed_response = self.parse_response(response)
        recommended_action = parsed_response["recommended_action"]

        # Check for looping behavior
        if self.detect_loop():
            if self.action_history:  # If there is action history, revert to the last action
                last_action = self.action_history[-1]  # Get the last action from the history
                print(f"Loop detected! Reverting to the last action: {last_action}.")
                recommended_action = last_action
            else:  # If no action history exists, default to "Go forward"
                print("Loop detected! No previous actions found. Defaulting to 'Go forward'.")
                recommended_action = "Go forward"

        # Add the prompt, response, and action to the history
        self.add_to_history(prompt, response, recommended_action)

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
    initial_prompt = prompt_engineer.generate_prompt({})
    print(f"Initial Prompt: {initial_prompt}")

    # Simulate a multi-turn conversation
    for turn in range(20):  # Adjust the number of turns as needed
        response, action_scores = prompt_engineer.process_image_and_prompt(image, initial_prompt)
        print(f"Turn {turn + 1} - Model Response: {response}")

        # Parse the model's response
        parsed_response = prompt_engineer.parse_response(response)
        print(f"Parsed Response: {parsed_response}")

        # Generate the next prompt based on the parsed response
        initial_prompt = prompt_engineer.generate_prompt(parsed_response)
        print(f"Next Prompt: {initial_prompt}")

    # Save the conversation history
    prompt_engineer.save_conversation_history("conversation_history.json")