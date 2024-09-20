from PIL import Image
import requests
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

model_id = "microsoft/Phi-3.5-vision-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="eager"
)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Load the local image
image = Image.open("../test_rgb_out/1.png")

# Prepare the input
messages = [
    {"role": "user", "content": "<|image_1|> Describe this image.",}
]

prompt = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

# Generate the response
generation_args = {
    "max_new_tokens": 1000,
    "temperature": 0.0,
    "do_sample": False,
}

generate_ids = model.generate(**inputs,
                              eos_token_id=processor.tokenizer.eos_token_id,
                              **generation_args)

# Remove input tokens
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids,
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]

print(response)