from configs.model_configs import MODEL_PATH, EMBED_CONFIG, SPLITTER_CONFIG
from utils import detect_device
from typing import Literal, List, Dict
from langchain.embeddings.base import Embeddings


def embedding_device(device: str = None) -> Literal["cuda", "mps", "cpu"]:
    device = device or EMBED_CONFIG["embed_device"]
    if device not in ["cuda", "mps", "cpu"]:
        device = detect_device()
    return device


def load_embeddings(model, device) -> Embeddings:
    if model == "text-embedding-ada-002":
        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model=model,
                                      openai_api_key=MODEL_PATH["embed_model"][model],
                                      chunk_size=SPLITTER_CONFIG["chunk_size"])
    elif model == "bge-m3":
        from FlagEmbedding import BGEM3FlagModel
        embeddings = BGEM3FlagModel(MODEL_PATH["embed_model"][model],
                                    use_fp16=True,
                                    use_multiprocessing=False,
                                    devices="cuda:0")
    elif 'bge-' in model:
        from langchain.embeddings import HuggingFaceBgeEmbeddings
        if 'zh' in model:
            # for chinese model
            query_instruction = "为这个句子生成表示以用于检索相关文章："
        elif 'en' in model:
            # for english model
            query_instruction = "Represent this sentence for searching relevant passages:"
        else:
            # maybe ReRanker or else, just use empty string instead
            query_instruction = ""
        embeddings = HuggingFaceBgeEmbeddings(model_name=MODEL_PATH["embed_model"][model],
                                            model_kwargs={'device': "cuda:0"},
                                            query_instruction=query_instruction)
        if model == "bge-large-zh-noinstruct":  # bge large -noinstruct embedding
            embeddings.query_instruction = ""
    else:
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_PATH["embed_model"][model],
                                        model_kwargs={'device': device})
    
    return embeddings


def embed_texts(
        texts: List[str],
        embed_model: str = EMBED_CONFIG["embed_model"],
        to_query: bool = False,
):
    '''
    data=List[List[float]]
    '''
    embeddings = load_embeddings(model=embed_model, device=embedding_device())
    data = embeddings.embed_documents(texts)
    return data


def embed_documents(docs, embed_model) -> Dict:
    
    output = embed_model.encode(docs, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    embeddings = output['dense_vecs']
    
    if embeddings is not None:
        return {
            "texts": docs,
            "embeddings": embeddings,
        }