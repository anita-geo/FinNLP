import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset
from tqdm import tqdm
import datasets
import torch

dic = {
    0:"negative",
    1:'positive',
    2:'neutral',
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
    if 'positive' in x or 'Positive' in x:
        return 'positive'
    elif 'negative' in x or 'Negative' in x:
        return 'negative'
    else:
        return 'neutral'

Python
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from tqdm import tqdm
import datasets
import torch
import pandas as pd

# Global dictionary for TFNS (0:Negative, 1:Positive, 2:Neutral)
dic = {
    0: "negative",
    1: "positive",
    2: "neutral",
}

def format_example(example: dict) -> dict:
    # 1. Define the System Prompt
    system_prompt = "You are a financial sentiment analysis expert. Analyze the sentiment of the tweet."
    
    # 2. Format User Content
    user_content = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        user_content += f"Input: {example['input']}\n"

    # 3. Apply Llama 3.1 Template structure
    # <|begin_of_text|><|start_header_id|>system...<|start_header_id|>user...<|start_header_id|>assistant
    context = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
    context += f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
    context += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    target = example["output"]
    return {"context": context, "target": target}

def change_target(x):
    """
    Converts text back to Integers. Returns 1 (Neutral) if model fails, to keep metrics valid.
    """
    if not isinstance(x, str):
        return 1 
    
    x = x.lower().strip()
    
    if 'positive' in x:
        return 1 # Note: TFNS uses 1 for Positive
    elif 'negative' in x:
        return 0 # TFNS uses 0 for Negative
    elif 'neutral' in x:
        return 2 # TFNS uses 2 for Neutral
    else:
        return 2 # Default to Neutral if unsure

def test_tfns(model, tokenizer, batch_size=8, prompt_fun=None):
    # --- CHANGE 1: Tokenizer Setup (Critical for Batch Inference) ---
    tokenizer.padding_side = 'left' 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Dataset
    dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')
    dataset = dataset['validation']
    dataset = dataset.to_pandas()
    dataset['label'] = dataset['label'].apply(lambda x: dic[x])
    
    if prompt_fun is None:
        dataset["instruction"] = 'What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.'
    else:
        dataset["instruction"] = dataset.apply(prompt_fun, axis=1)

    dataset.columns = ['input', 'output', 'instruction']
    
    # Use the NEW format_example function
    dataset[["context", "target"]] = dataset.apply(format_example, axis=1, result_type="expand")

    # Print example
    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    context = dataset['context'].tolist()
    
    total_steps = (len(context) + batch_size - 1) // batch_size
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")

    out_text_list = []
    
    # --- CHANGE 2: Optimized Inference Loop ---
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(total_steps)):
            tmp_context = context[i * batch_size : (i + 1) * batch_size]
            if not tmp_context: continue

            # Tokenize with truncation
            tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, truncation=True, max_length=512)
            for k in tokens.keys():
                tokens[k] = tokens[k].cuda()
            
            # Generate (Fast settings)
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
                # Handle cases where model outputs "assistant" header or "Answer:" prefix
                if "assistant" in o:
                    clean_out = o.split("assistant")[-1].strip()
                elif "Answer:" in o:
                    clean_out = o.split("Answer:")[-1].strip()
                else:
                    clean_out = o
                batch_out.append(clean_out)
                
            out_text_list += batch_out

    dataset["out_text"] = out_text_list
    dataset["new_target"] = dataset["target"].apply(change_target)
    dataset["new_out"] = dataset["out_text"].apply(change_target)

    acc = accuracy_score(dataset["new_target"], dataset["new_out"])
    f1_macro = f1_score(dataset["new_target"], dataset["new_out"], average="macro")
    f1_micro = f1_score(dataset["new_target"], dataset["new_out"], average="micro")
    f1_weighted = f1_score(dataset["new_target"], dataset["new_out"], average="weighted")

    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted: {f1_weighted}.")

    return dataset