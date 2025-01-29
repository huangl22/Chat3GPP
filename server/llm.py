from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.model_configs import MODEL_PATH
import torch
import numpy as np
def set_seed():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
set_seed()

def generate_prompt(question_text, options):
    
    prompt_template = """You are an expert on 3GPP standards. Please answer the multiple-choice question by selecting the correct option. Respond in the format: "answer": "option X: [selected option content]".

Question:
{question_text}

Options:
{options_text}

Answer:"""
    
    options_text = "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options.values())])
    
    prompt = prompt_template.format(
        question_text=question_text,
        options_text=options_text
    )
    
    return prompt


def generate_multiple_choice_prompt(retrieved_documents, question_text, options):
    
    prompt_template = """You are an expert on 3GPP standards. Based on the provided context, answer the multiple-choice question by selecting the correct option. Respond in the format: "answer": "option X: [selected option content]".

Context:
{retrieved_documents}

Question:
{question_text}

Options:
{options_text}

Instructions:
1. Carefully review the context provided to determine the correct answer.
2. If the context does not provide sufficient information, respond with "Insufficient context to answer."
3. Provide your answer in the exact format: "answer": "option X: [selected option content]".

Answer:
    """
    
    formatted_documents = "\n".join(retrieved_documents)
    
    options_text = "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options.values())])
    
    prompt = prompt_template.format(
        retrieved_documents=formatted_documents,
        question_text=question_text,
        options_text=options_text
    )
    
    return prompt

def load_llama3_model(model_name="llama-3-8b", device="auto"):
    model_path = MODEL_PATH["llm_model"][model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="device").eval()
    return model, tokenizer


def generate_answer_llama3(model, tokenizer,prompt, max_length=8192):
    
    prompt = prompt
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(inputs["input_ids"], max_length=max_length, eos_token_id=terminators, repetition_penalty=1.1)
    answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    return answer