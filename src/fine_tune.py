import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,   
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from trl import SFTTrainer

import matplotlib.pyplot as plt    
import numpy as np                 

BASE_MODEL = "./qwen-1.5b" 
OUTPUT_DIR = "qwen_medical_lora"
MAX_SEQ_LENGTH = 2048

PER_DEVICE_TRAIN_BATCH_SIZE = 8   
GRADIENT_ACCUMULATION_STEPS = 4    

NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 1e-4   
METRICS_FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

MEDICAL_PROMPT = """你是一名专业的儿科医生，正在为家长提供线上健康咨询。请根据给出的信息，给出耐心、通俗、条理清晰的解答。

指令
{instruction}

详细提问
{input}

医生回复
{output}"""

def formatting_func(examples):
    instructions = examples.get("instruction", [])
    inputs = examples.get("input", [])
    outputs = examples.get("output", [])
    
    texts = []
    for instruction, input_, output in zip(instructions, inputs, outputs):
        text = MEDICAL_PROMPT.format(
            instruction=instruction,
            input=input_,
            output=output,
        )
        texts.append(text)
    return {"text": texts}

def process_dataset():
    if not os.path.exists("data/train.jsonl"):
        print("Warning: 找不到 data/train.jsonl,请检查路径。")

    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/train.jsonl",
            "test": "data/test.jsonl",
        },
    )

    print(">>> Processing dataset...")
    dataset = dataset.map(formatting_func, batched=True)
    print(f">>> Train set size: {len(dataset['train'])}")
    print(f">>> Test set size: {len(dataset['test'])}")
    
    return dataset

def load_qwen_optimized():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,        
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa" 
    )

    model.gradient_checkpointing_enable() 
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=64,               
        lora_alpha=128,     
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

def ensure_dir(path: str):
    dirname = path if os.path.isdir(path) else os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

def plot_steps(xs_steps, ys, ylabel: str, out_png: str, title: str = "", marker: str = ""):
    ensure_dir(out_png)
    plt.figure()
    if xs_steps and ys:
        if marker:
            plt.plot(xs_steps, ys, marker) 
        else:
            plt.plot(xs_steps, ys)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

class MetricsRecorderCallback(TrainerCallback):
    def __init__(self):
        self.steps = []
        self.history = {
            "loss": [],
            "eval_loss": [],
            "grad_norm": [],
            "learning_rate": [],
            "entropy": [],
            "num_tokens": [],
            "mean_token_accuracy": [],
            "epoch": []
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        step = state.global_step
        
        if "loss" in logs or "eval_loss" in logs:
            self.steps.append(step)
            
            for k in self.history.keys():
                if k in logs:
                    self.history[k].append(logs[k])
                else:
                    self.history[k].append(None)

def plot_all_metrics(recorder: MetricsRecorderCallback, out_dir: str):
    ensure_dir(out_dir)
    steps = recorder.steps
    metrics = recorder.history
    
    print(f">>> Plotting metrics to {out_dir}...")
    
    for name, values in metrics.items():
        xs_clean = []
        ys_clean = []
        
        min_len = min(len(steps), len(values))
        for i in range(min_len):
            v = values[i]
            if v is not None:
                xs_clean.append(steps[i])
                ys_clean.append(v)
        
        if not xs_clean:
            continue
            
        out_png = os.path.join(out_dir, f"{name}_vs_step.png")
        title = f"{name} over training steps"
        
        style = 'o-' if ('eval' in name or 'epoch' in name) else '-'
        
        plot_steps(xs_clean, ys_clean, ylabel=name, out_png=out_png, title=title, marker=style)
        print(f"    [Saved] {out_png}")

def main():
    dataset = process_dataset() 
    model, tokenizer = load_qwen_optimized()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",    
        eval_steps=50,             
        save_strategy="steps",
        save_steps=500,            
        save_total_limit=2,
        bf16=True,                 
        fp16=False,
        optim="adamw_torch",       
        gradient_checkpointing=True,
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none", 
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    metrics_recorder = MetricsRecorderCallback()

    print(">>> Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        #dataset_text_field="text",      
        #max_seq_length=MAX_SEQ_LENGTH,  
        #packing=False,                  
        args=training_args,
        callbacks=[metrics_recorder],   
    )

    print("="*40)
    print(f"Training Started on {torch.cuda.get_device_name(0)}")
    print(f"Total Batch Size: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print("="*40)
    trainer.train()
    plot_all_metrics(metrics_recorder, METRICS_FIG_DIR)

    print(f">>> Saving model to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\nTraining Finished Successfully")

if __name__ == "__main__":
    main()