from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os

# 🧹 Disable bnb welcome + HF warnings
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# ✅ Load dataset
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# ✅ Model ID (CodeLlama 7B Instruct)
model_id = "codellama/CodeLlama-7b-Instruct-hf"

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# ✅ BitsAndBytes config for 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ✅ Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# ✅ Resize token embeddings
model.resize_token_embeddings(len(tokenizer))

# ✅ Apply LoRA FIRST (before enabling gradient checkpointing)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # all safe for LLaMA
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)
model.enable_input_require_grads()


# ✅ THEN enable gradient checkpointing
model.gradient_checkpointing_enable()

# ✅ Print trainable parameters
model.print_trainable_parameters()

# ✅ Optional: sanity check
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"✅ {name} is trainable.")

# ✅ Tokenize function
def tokenize(batch):
    prompts = [
        f"### Instruction:\n{p}\n\n### Response:\n{c}"
        for p, c in zip(batch["prompt"], batch["completion"])
    ]
    tokens = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# ✅ Tokenize dataset
tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# ✅ Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned-codellama-bash",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    logging_dir="./logs",
    report_to="none"
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

# ✅ Train the model
trainer.train()
