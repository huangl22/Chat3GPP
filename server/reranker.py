import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from configs.model_configs import MODEL_PATH

def load_reranker(model_name="bge-reranker-large", device="cuda"):
    model_path = MODEL_PATH["reranker_model"][model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map="auto").eval()
    return model, tokenizer

def get_rerank_scores(model, tokenizer, query, docs):
    pairs = [[query, doc] for doc in docs]
    
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(model.device)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        return scores.tolist()