import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset
from tqdm import tqdm
import datasets
import torch

dic = {
    'strong negative':"negative",
    'moderately negative':"negative",
    'mildly negative':"neutral",
    'strong positive':"positive",
    'moderately positive':"positive",
    'mildly positive':'neutral',
    'neutral':'neutral',
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

def test_nwgi(model, tokenizer, batch_size=8, prompt_fun=None):
    # --- CHANGE 1: Tokenizer Setup ---
    tokenizer.padding_side = 'left' 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = datasets.load_dataset('oliverwang15/news_with_gpt_instructions')
    dataset = dataset['test'].to_pandas()
    dataset['output'] = dataset['label'].apply(lambda x: dic[x])

    if prompt_fun is None:
        dataset["instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    else:
        dataset["instruction"] = dataset.apply(prompt_fun, axis=1)
    
    dataset["input"] = dataset["news"]
    dataset = dataset[['input', 'output', 'instruction']]
    
    # Use the NEW format_example function defined above
    dataset[["context", "target"]] = dataset.apply(format_example, axis=1, result_type="expand")

    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    context = dataset['context'].tolist()
    total_steps = (len(context) + batch_size - 1) // batch_size
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")

    out_text_list = []
    
    # --- CHANGE 2: Optimized Loop ---
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(total_steps)):
            tmp_context = context[i * batch_size : (i + 1) * batch_size]
            
            # Tokenize with truncation
            tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, truncation=True, max_length=512)
            for k in tokens.keys():
                tokens[k] = tokens[k].cuda()
            
            # Generate with max_new_tokens instead of max_length
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
                # Llama 3.1 Instruct usually just outputs the answer, but sometimes repeats the prompt.
                # We split by "assistant" header if it exists in the decoded text, otherwise rely on keywords.
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
