import os
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
import torch

hf_token = 'hf_token' #허깅페이스 토큰
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype, load_in_4bit = load_in_4bit,
    token=hf_token
)
model = FastLanguageModel.get_peft_model(
    model,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r = 16, lora_alpha = 16, lora_dropout = 0,
    bias = "none", use_gradient_checkpointing = "unsloth", 
    random_state = 3407, use_rslora = False, loftq_config = None
)

prompt = """아래는 작업을 설명하는 지시사항입니다. 입력된 내용을 바탕으로 적절한 응답을 작성하세요.
### 지시사항: 아래 입력에 대한 적절한 응답을 제공하세요.
### 입력: {input}
### 응답: {response}
"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    inputs = examples["input"]
    responses = examples["response"]
    texts = []
    for input, response in zip(inputs, responses):
        text = prompt.format(input=input, response=response) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("DatasetPath", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

trainer = SFTTrainer(
    model = model, tokenizer = tokenizer,
    train_dataset = dataset, dataset_text_field = "text",
    max_seq_length = max_seq_length, dataset_num_proc = 2,
    packing = False, args = TrainingArguments(
        per_device_train_batch_size=128,
        gradient_accumulation_steps=4,
        # warmup_steps=5, max_steps=60,
        learning_rate=2e-4, lr_scheduler_type="linear",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps = 1, output_dir="outputs",
        optim="adamw_8bit", weight_decay=0.01,
        num_train_epochs=3, seed=3407
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved()/1024/1024/1024, 3)
max_memory = round(gpu_stats.total_memory/1024/1024/1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

print('Saving...')
model.push_to_hub_gguf(
    "ModelPath",
    tokenizer,
    quantization_method="q8_0",
    token=hf_token
)
print('Success Save')
# os.system('pm2 stop 0')
