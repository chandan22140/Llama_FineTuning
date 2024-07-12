
# !pip install -q accelerate transformers peft==0.4.0 bitsandbytes==0.40.2 trl==0.4.7 optuna requests==2.31.0 pyarrow==14.0.1

import os
import optuna
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
new_model = "llama-2-7b-codewello"

# Load dataset (you can process it here)
dataset = load_dataset(dataset_name, split="train")

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Set training parameters
def objective(trial):
    lora_r = trial.suggest_int("lora_r", 16, 128)
    lora_alpha = trial.suggest_int("lora_alpha", 16, 64)
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.3)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", 2, 8)
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 4)

    training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=25,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard"
    )

    lora_config = LoraConfig(
        r=lora_r,
        # alpha=lora_alpha,
        # dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        quantization_config=bnb_config,
    )

    model = PeftModel(base_model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    trainer.train()

    eval_result = trainer.evaluate()
    return eval_result["eval_loss"]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_trial = study.best_trial
print(f"Best trial: {best_trial.params}")

# Using the best hyperparameters for final training
lora_r = best_trial.params["lora_r"]
lora_alpha = best_trial.params["lora_alpha"]
lora_dropout = best_trial.params["lora_dropout"]
learning_rate = best_trial.params["learning_rate"]
per_device_train_batch_size = best_trial.params["per_device_train_batch_size"]
gradient_accumulation_steps = best_trial.params["gradient_accumulation_steps"]

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=learning_rate,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard"
)

lora_config = LoraConfig(
    r=lora_r,
    alpha=lora_alpha,
    dropout=lora_dropout,
    target_modules=["q_proj", "v_proj"]
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
    quantization_config=bnb_config,
)

model = PeftModel(base_model, lora_config)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)

# Commented out IPython magic to ensure Python compatibility.
#  %load_ext tensorboard
#  %tensorboard --logdir results/runs

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

# Empty VRAM
del model
del pipe
del trainer
import gc
gc.collect()
gc.collect()

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Upload
# !huggingface-cli login

# model.push_to_hub(new_model, use_temp_dir=False)
# tokenizer.push_to_hub(new_model, use_temp_dir=False)
