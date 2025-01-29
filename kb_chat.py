from retrievers.VectorRetriever import VectorRetrieval
from elasticsearch import Elasticsearch, helpers
from configs.model_configs import EMBED_CONFIG, LLM_CONFIG
from server.embedding import load_embeddings
from server.llm import load_llama3_model, generate_answer_llama3, generate_multiple_choice_prompt
import json
import re
retrieval_classes = {
    "vector": VectorRetrieval
}

question_path = ""
save_path = ""

def search_docs(query, es_client, retrieval_type="vector", index_name="kb_vector", embed_model=EMBED_CONFIG["embed_model"], top_k1=1000, top_k2=5):
    kb_service = retrieval_classes[retrieval_type](embed_model, es_client=es_client)
    docs = kb_service.search_rrf(index_name, query, top_k1=top_k1, top_k2=top_k2)
    return docs

es_client = Elasticsearch("https://localhost:9200",
                          http_auth=("elastic", ""),
                          verify_certs=False
)

if es_client.ping():
    print("Successfully connected to Elasticsearch!")
    embed_model = load_embeddings(model=EMBED_CONFIG["embed_model"], device=EMBED_CONFIG["embed_device"])
    llama3_model, llama3_tokenizer = load_llama3_model(model_name=LLM_CONFIG["llm_model"], device=LLM_CONFIG["llm_device"])
else:
    print("Could not connect to Elasticsearch.")


with open(question_path, "r", encoding="utf-8") as f:
    all_questions = json.load(f)

flag = False
results = {}
for question_key, question_data in all_questions.items():
    query = question_data["question"]
    options = {key:value for key, value in question_data.items() if key.startswith("option")}
    docs = search_docs(query, es_client, retrieval_type="vector", index_name="kb_vector", embed_model=embed_model)
    prompt = generate_multiple_choice_prompt(docs, query, options)
    answer = generate_answer_llama3(llama3_model, llama3_tokenizer, prompt, max_length=2048)

with open(save_path, 'w') as f:
    res_str = json.dump(results, f, indent=2)