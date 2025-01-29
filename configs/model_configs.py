EMBED_CONFIG = {
    "embed_model": "bge-m3",
    "embed_device": "cuda",
}

SPLITTER_CONFIG = {
    "chunk_size":256,
}

LLM_CONFIG = {
    "llm_model": "llama3-8B-instruct",
    "llm_device": "cuda",
}

RERANKER_CONFIG = {
    "reranker_model": "bge-reranker-large",
    "reranker_device": "cuda",
}

MODEL_PATH = {
    "embed_model": {
        "bge-large-en": "",
        "bge-m3": "",
    },
    "llm_model": {
        "llama3-8B-instruct": "",
        "qwen2.5-7B-instruct": "",
        "LLama-3-8B-Tele-it": ""
    },
    "reranker_model": {
        "bge-reranker-large": ""
    }
}
