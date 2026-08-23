# Unsloth must be imported before trl / transformers / peft.
from unsloth import FastModel

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "unsloth/Qwen3-14B-bnb-4bit"
MAX_LENGTH = 2048

# Change this to False once the pipeline works.
SMOKE_TEST = True

print("Loading model...")

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_LENGTH,
    # QLoRA:
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)


print("Adding LoRA adapters...")

model = FastModel.get_peft_model(
    model,
    # LoRA rank
    r=32,
    # Train attention + MLP projections
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    # Unsloth's memory-efficient gradient checkpointing
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train",
)


def split_turn(example):
    messages = example["messages"]
    return {
        "prompt": messages[:-1],
        "completion": messages[-1:],
    }


dataset = dataset.map(split_turn, remove_columns=["messages"])

print(dataset)
print(dataset[0])


bf16 = torch.cuda.is_bf16_supported()

training_args = SFTConfig(
    output_dir="outputs/qwen3-14b",
    # Context
    max_length=MAX_LENGTH,
    eos_token=tokenizer.eos_token,
    # Batch size of 2 × accumulation 4 = effective batch 8
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    # For our 3-example test, don't run epochs forever.
    max_steps=10 if SMOKE_TEST else -1,
    num_train_epochs=1 if SMOKE_TEST else 2,
    # Standard LoRA starting point
    learning_rate=2e-4,
    warmup_ratio=0.05,
    # Memory-efficient optimizer
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    logging_steps=1 if SMOKE_TEST else 10,
    save_strategy="steps",
    save_steps=10 if SMOKE_TEST else 100,
    save_total_limit=2,
    bf16=bf16,
    fp16=not bf16,
    completion_only_loss=True,
    packing=False,
    seed=3407,
    report_to="none",
)


trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)


print("Starting training 🚀")

stats = trainer.train()

print(stats)

print("Saving LoRA adapter...")

model.save_pretrained("adapters/qwen3-14b")
tokenizer.save_pretrained("adapters/qwen3-14b")

print("Done.")
