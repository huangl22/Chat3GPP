from configs.model_configs import EMBED_CONFIG, LLM_CONFIG
from server.llm import load_llama3_model, generate_answer_llama3, generate_multiple_choice_prompt, generate_prompt
import json
import re
import numpy as np
import torch

def set_seed():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
set_seed()

question_path = ""
save_path = ""

with open(question_path, "r", encoding="utf-8") as f:
    all_questions = json.load(f)
    
llama3_model, llama3_tokenizer = load_llama3_model(model_name=LLM_CONFIG["llm_model"], device=LLM_CONFIG["llm_device"])
results = {}
for question_key, question_data in all_questions.items():
    
    query = question_data["question"]
    options = {key:value for key, value in question_data.items() if key.startswith("option")}
    prompt = generate_prompt(query, options)
    answer = generate_answer_llama3(llama3_model, llama3_tokenizer, prompt, max_length=1024)
    option_match = re.search(r'\boption\s*\d+\b', answer, re.IGNORECASE)
    answer = option_match.group(0).strip() if option_match else "Option not found."
    print(f"{question_key}:{answer}")
    results[question_key] = answer

with open(save_path, 'w') as f:
    res_str = json.dump(results, f, indent=2)
