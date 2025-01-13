from fastapi import FastAPI, UploadFile, Form
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch
import io
import re

# Initialize FastAPI app
app = FastAPI()

# Load the model and processor
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True
)
model.to("cuda:0")

@app.post("/process")
async def process_image(file: UploadFile, prompt: str = Form(...)):
    """
    Process the uploaded image and the prompt, then return the model's response.
    """
    # Read the uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Prepare the conversation with the prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]

    # Apply the prompt template
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Process the input
    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=300)
    response = processor.decode(output[0], skip_special_tokens=True)

    # Extract action probabilities
    pattern = r"(Go forward|Go backward|Turn right|Turn left): (\d+(\.\d+)?)"
    matches = re.findall(pattern, response)
    action_scores = {action: float(score) for action, score, _ in matches}

    return {
        "response": response,
        "action_scores": action_scores,
    }

