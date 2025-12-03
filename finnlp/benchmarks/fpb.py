import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from tqdm import tqdm
import torch
import pandas as pd

# Global dictionary for converting the Dataset's Integers to Text
dic = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

def format_example(example: dict) -> dict:
    # 1. Define the System Prompt
    system_prompt = "You are a financial sentiment analysis expert. Analyze the sentiment of the news."
    
    # 2. Format User Content
    user_content = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        user_content += f"Input: {example['input']}\n"

    # 3. Apply Llama 3.1 Template structure
    # <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    
    context = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
    context += f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
    context += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    target = example["output"]
    return {"context": context, "target": target}

def change_target(x):
    """
    Converts text (Model Output or Target String) back to Integers (0, 1, 2)
    for accurate Metric calculation.
    """
    if not isinstance(x, str):
        return -1 # Error case
        
    x = x.lower()
    
    if 'positive' in x:
        return 2
    elif 'negative' in x:
        return 0
    elif 'neutral' in x:
        return 1
    else:
        # If the model hallucinates something else, treat it as a wrong answer (or default to neutral)
        return -1 

def test_financial_sa(model, tokenizer, batch_size=8, prompt_fun=None):
    # --- CHANGE 1: Tokenizer Setup ---
    tokenizer.padding_side = 'left' 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    instructions = load_dataset("JackieYue/financial_sa")
    instructions = instructions["train"]
    instructions = instructions.train_test_split(seed=42)['test']
    instructions = instructions.to_pandas()
    instructions = instructions.rename(columns={"sentence": "input", "label": "output"})
    instructions["output"] = instructions["output"].map(dic)

    if prompt_fun is None:
        instructions["instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    else:
        instructions["instruction"] = instructions.apply(prompt_fun, axis=1)
    
    # Use the NEW format_example function
    instructions[["context", "target"]] = instructions.apply(format_example, axis=1, result_type="expand")

    print(f"\n\nPrompt example:\n{instructions['context'][0]}\n\n")

    context = instructions['context'].tolist()
    total_steps = (len(context) + batch_size - 1) // batch_size
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")

    out_text_list = []
    
    # --- CHANGE 2: Optimized Loop ---
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(total_steps)):
            tmp_context = context[i * batch_size : (i + 1) * batch_size]
            if not tmp_context: continue

            tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, truncation=True, max_length=512)
            for k in tokens.keys():
                tokens[k] = tokens[k].cuda()
            
            res = model.generate(
                **tokens, 
                max_new_tokens=10, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            
            res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
            
            # --- CHANGE 3: Robust Parsing ---
            batch_out = []
            for o in res_sentences:
                if "assistant" in o:
                    clean_out = o.split("assistant")[-1].strip()
                elif "Answer:" in o:
                    clean_out = o.split("Answer:")[-1].strip()
                else:
                    clean_out = o
                batch_out.append(clean_out)
                
            out_text_list += batch_out

    instructions["out_text"] = out_text_list
    instructions["new_target"] = instructions["target"].apply(change_target)
    instructions["new_out"] = instructions["out_text"].apply(change_target)

    acc = accuracy_score(instructions["new_target"], instructions["new_out"])
    f1_macro = f1_score(instructions["new_target"], instructions["new_out"], average="macro")
    f1_micro = f1_score(instructions["new_target"], instructions["new_out"], average="micro")
    f1_weighted = f1_score(instructions["new_target"], instructions["new_out"], average="weighted")

    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted: {f1_weighted}.")

    return instructions