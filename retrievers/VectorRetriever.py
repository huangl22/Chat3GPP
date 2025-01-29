from retrievers.base import BaseRetrieval
from elasticsearch.helpers import bulk
import numpy as np

class VectorRetrieval(BaseRetrieval):
    
    def get_documents_with_ids(self, doc_ids, index):
        response = self.es.mget(
            index = index,
            body = {
                "ids": doc_ids
            }
        )
        return [hit["_source"]["content"] for hit in response["docs"]]
    
    def calculate_rrf(self, bm25_results, vector_results, k=10):
        all_docs = {}
        
        for doc_id, rank in bm25_results:
            all_docs[doc_id] = {"bm25_rank": rank, "vector_rank": None}
        
        for doc_id, rank in vector_results:
            if doc_id not in all_docs:
                all_docs[doc_id] = {"bm25_rank": None, "vector_rank": rank}
            else:
                all_docs[doc_id]["vector_rank"] = rank
        
        for doc_id, ranks in all_docs.items():
            bm25_rank = ranks["bm25_rank"] if ranks["bm25_rank"] is not None else float("inf")
            vector_rank = ranks["vector_rank"] if ranks["vector_rank"] is not None else float("inf")
            
            rrf_score = (1 / (k + bm25_rank)) + (1 / (k + vector_rank))
            all_docs[doc_id]["rrf_score"] = rrf_score

        sorted_docs = sorted(all_docs.items(), key=lambda x: x[1]["rrf_score"], reverse=True)

        return sorted_docs

    def bm25_retrieve(self, query, index, size=1000):
        query_body = {
        "query": {
                "match": {
                    "content": query
                }
            },
            "size": size
        }
        
        response = self.es.search(index=index, body=query_body)
        
        return response
    
    def vector_retrieve(self, query, index, size=1000):
        query_embedding = self._docs_to_embeddings([query])
        query_body = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": query_embedding["embeddings"][0]
                        }
                    }
                }
            },
            "size": size
        }
        
        response = self.es.search(index=index, body=query_body)
        
        return response

    def build_index(self, index_name, chunks):
        
        if not self.es.indices.exists(index=index_name):
            properties = {
                "filename": {"type": "text"},
                "content": {"type": "text", "analyzer": "custom_analyzer"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 1024,
                    "index": True,
                    "similarity": "cosine"
                },
                "date": {"type": "date"}
            }

            self.es.indices.create(
                index=index_name,
                body={
                    "settings": {
                        "analysis": {
                            "tokenizer": {
                                "standard_tokenizer": {
                                    "type": "standard"
                                }
                            },
                            "filter": {
                                "custom_stop_filter": {
                                    "type": "stop",
                                    "stopwords_path": "stopwords.txt"
                                },
                            },
                            "analyzer": {
                                "custom_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard_tokenizer",
                                    "filter": [
                                        "lowercase",
                                        "custom_stop_filter",
                                        ]
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": properties
                    }
                }
            )
        
        embeddings = self._docs_to_embeddings(chunks["text"])
        actions = [
            {
                "_index": index_name,
                "_source": {
                    "filename": chunks["filename"],
                    "content": chunk,
                    "embedding": embedding,
                    "date": chunks['date']
                }
            }
            for chunk, embedding in zip(chunks["text"], embeddings["embeddings"])
        ]
        bulk(self.es, actions)
        
    def search_rrf(self, index_name, query, top_k1=1000, top_k2=5, **kwargs):
        
        response = self.bm25_retrieve(query, index_name, top_k1)
        
        bm25_ranks = [(hit["_id"], idx + 1) for idx, hit in enumerate(response["hits"]["hits"])]
        
        response = self.vector_retrieve(query, index_name, top_k1)
        
        vector_ranks = [(hit["_id"], idx + 1) for idx, hit in enumerate(response["hits"]["hits"])]
        
        final_sorted_docs = self.calculate_rrf(bm25_ranks, vector_ranks)
        sorted_doc_ids = [doc_id for doc_id, result in final_sorted_docs][:int(1/10 * top_k1)]
        docs = self.get_documents_with_ids(sorted_doc_ids, index_name)
        
        output_1 = self.embed_model.encode([query], return_dense=False, return_sparse=False, return_colbert_vecs=True)
        output_2 = self.embed_model.encode(docs, return_dense=False, return_sparse=False, return_colbert_vecs=True)
        
        colbert_scores = []
        for i in range(len(docs)):
            colbert_score = float(self.embed_model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][i]))
            colbert_scores.append(colbert_score)
        zipped = list(zip(docs, colbert_scores))
        sorted_zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
        sorted_docs = [doc for doc, _ in sorted_zipped][:top_k2]
        sorted_scores = [score for _, score in sorted_zipped][:top_k2]
        
        return sorted_docs,sorted_scores