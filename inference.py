import torch
from unsloth import FastModel


model, tokenizer = FastModel.from_pretrained(
    model_name="adapters/qwen3-14b",
    max_seq_length=2048,
    load_in_4bit=True,
)

FastModel.for_inference(model)


messages = [
    {
        "role": "user",
        "content": "Write a safe Python divide function."
    }
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)


inputs = tokenizer(
    text,
    return_tensors="pt",
).to("cuda")


outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    do_sample=True,
)


generated = outputs[0][inputs.input_ids.shape[1]:]

print(
    tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )
)
