import requests

import numpy as np
from PIL import Image

from vlm_ask import VLMModelClient
import pdb

test_image_path = "/home/mhabibp/Pictures/Screenshots/habitat1.png"
test_prompt = '''

          You are a robot navigating an indoor environment in search of a couch. 
The image on the left is your current observation
You must think step by step and ensure that all parts of your response are consistent. 

Here are the tasks:
1. Identify what part of the house we are about to enter (choose from: [bedroom, living room, kitchen, corridor, bathroom]). 
2. Assess whether a couch can realistically be found in this area, based on common sense and the current observation. 
3. Determine the most logical next action for the robot (choose from: [go forward, go backward, turn right, turn left]). 
   - The chosen action must prioritize exploring areas likely to contain a couch. 
   - Avoid suggesting actions that contradict previous observations (e.g., don't explore a bathroom if 
   couches aren't found there). 
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

Important: Ensure that the recommended action aligns with the reasoning from steps 1 and 2. 
If a couch cannot be found in the current area, prioritize moving to areas more likely to contain a couch.
          '''
    
# Ensure the server is running and reachable
server_vlm = VLMModelClient()
try:
    print("Sending request to server...")
    image = Image.open("/home/mhabibp/Pictures/Screenshots/habitat1.png").convert("RGB")
    response = server_vlm.process_input(np.array(image), test_prompt)
    print("Server response:", response)
except requests.exceptions.Timeout:
    print("Error: The server took too long to respond. Try increasing the timeout.")
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the server. Ensure it is running on the correct port.")
except Exception as e:
    print(f"Unexpected error: {e}")


# import requests

# url = "http://localhost:8002/process"

# image = Image.open("/home/mhabibp/Pictures/Screenshots/habitat1.png")

    
# data = {"prompt": test_prompt}
# response = requests.post(url, files=np.array(image), data=data)

# # Print the result
# print(response)