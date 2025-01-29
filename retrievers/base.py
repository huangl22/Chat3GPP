from abc import ABC, abstractmethod
from typing import Dict
from server.embedding import embed_documents

class BaseRetrieval(ABC):
    
    def __init__(self, embed_model, es_client):
        self.embed_model = embed_model
        self.es = es_client
    
    @abstractmethod
    def build_index(self, index_name, chunks, **kwargs):
        pass
    
    @abstractmethod
    def search(self, index_name, query, **kwargs):
        """搜索接口"""
    
    def _docs_to_embeddings(self, docs) -> Dict:
        return embed_documents(docs=docs, embed_model=self.embed_model)