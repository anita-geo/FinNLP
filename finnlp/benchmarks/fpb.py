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
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
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
    # 1. Load the specific JackieYue dataset
    instructions = load_dataset("JackieYue/financial_sa")
    instructions = instructions["train"]
    
    # 2. Split (using the same seed as your original code)
    instructions = instructions.train_test_split(seed=42)['test']
    instructions = instructions.to_pandas()
    
    # 3. Safe Column Renaming
    # The dataset has 'sentence' and 'label'. We map them to 'input' and 'output'.
    instructions = instructions.rename(columns={"sentence": "input", "label": "output"})
    
    # 4. Map Integer labels (0,1,2) to Text ("negative", etc.) using the global dic
    instructions["output"] = instructions["output"].map(dic)

    if prompt_fun is None:
        instructions["instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    else:
        instructions["instruction"] = instructions.apply(prompt_fun, axis=1)
    
    # Format the prompts
    instructions[["context", "target"]] = instructions.apply(format_example, axis=1, result_type="expand")

    # Print example for verification
    print(f"\n\nPrompt example:\n{instructions['context'][0]}\n\n")

    context = instructions['context'].tolist()
    
    # Calculate steps
    total_steps = (len(context) + batch_size - 1) // batch_size
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")

    out_text_list = []
    
    for i in tqdm(range(total_steps)):
        tmp_context = context[i * batch_size : (i + 1) * batch_size]
        
        # Safety check for empty batch
        if not tmp_context:
            continue

        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512, truncation=True)
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
            
        res = model.generate(**tokens, max_length=512)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
        
        # Safety check: Handle cases where model doesn't generate "Answer: "
        batch_out_text = []
        for o in res_sentences:
            if "Answer: " in o:
                batch_out_text.append(o.split("Answer: ")[1])
            else:
                # Append the whole output or an empty string if format is broken
                batch_out_text.append(o)
                
        out_text_list += batch_out_text
        torch.cuda.empty_cache()

    instructions["out_text"] = out_text_list
    
    # 5. Convert both Ground Truth and Prediction back to Integers for Metrics
    instructions["new_target"] = instructions["target"].apply(change_target)
    instructions["new_out"] = instructions["out_text"].apply(change_target)

    # Calculate metrics
    acc = accuracy_score(instructions["new_target"], instructions["new_out"])
    f1_macro = f1_score(instructions["new_target"], instructions["new_out"], average="macro")
    f1_micro = f1_score(instructions["new_target"], instructions["new_out"], average="micro")
    f1_weighted = f1_score(instructions["new_target"], instructions["new_out"], average="weighted")

    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted: {f1_weighted}. ")

    return instructions
