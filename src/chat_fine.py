import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_dir = "./qwen-1.5b" 
lora_dir = "./qwen_medical_lora"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()

except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

def run_inference(model, tokenizer, prompt):
    medical_prompt = f"家长提问:{prompt}\n\n 医生回复:"
    model_inputs = tokenizer([medical_prompt], return_tensors="pt").to(model.device)
    stop_words = [tokenizer.eos_token_id]
    bad_keywords = ["[温馨提醒]", "以上是对", "Human:", "Human", "用户:", "[编辑本段]"]
    for kw in bad_keywords:
        try:
            kw_ids = tokenizer.encode(kw)
            if kw_ids:
                stop_words.append(kw_ids[0])
        except:
            pass 

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            
            eos_token_id=stop_words 
        )

    input_length = model_inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_length:]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    response = response.split("家长提问:")[0]

    for kw in bad_keywords:
        if kw in response:
            response = response.rsplit(kw, 1)[0]

    for tail in ["以上。", "以上 。", "以上 .", "以上"]:
        resp_strip = response.strip()
        if resp_strip.endswith(tail.strip()):
            idx = resp_strip.rfind("以上")
            response = resp_strip[:idx]
            break

    return response.strip()

if __name__ == "__main__":
    print("大模型基础与应用大作业第33组")
    print("-" * 50)
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ["退出", "quit", "exit"]:
            print("对话结束。")
            break
        # 1. 微调前 (Base)
        with model.disable_adapter():
            response_base = run_inference(model, tokenizer, user_input)
            print(f"\n[微调前]:\n{response_base}")
        print("-" * 50)
        # 2. 微调后 (LoRA)
        response_lora = run_inference(model, tokenizer, user_input)
        print(f"[微调后]:\n{response_lora}")
        
        print("-" * 50)