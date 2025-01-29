from datetime import datetime
from docx import Document
from text_splitter.TSdocx_splitter import TSDocTextSplitter
from retrievers.VectorRetriever import VectorRetrieval
from configs.model_configs import EMBED_CONFIG
from utils import list_files_from_folder, list_kbs_from_folder
from pathlib import Path
from datetime import datetime
from elasticsearch import Elasticsearch, helpers
from server.embedding import load_embeddings
from configs.model_configs import MODEL_PATH, EMBED_CONFIG, SPLITTER_CONFIG
import warnings

# warnings.filterwarnings("ignore")
retrieval_classes = {
    "vector": VectorRetrieval
}


def process_docx(file_path):
    
    chunks = {}
    file = Document(file_path)
    splitter = TSDocTextSplitter()
    content = splitter.split_text(file)
    chunks["text"] = content
    chunks["filename"] = str(Path(file_path).as_posix())
    chunks["date"] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    return chunks

def files2db(retrieval_type, es_client, index_name, kb_files, embed_model):
    kb_service = retrieval_classes[retrieval_type](embed_model=embed_model, es_client=es_client)
    for kb_file in kb_files:
        try:
            chunks = process_docx(kb_file)
            if chunks["text"]:
                print(f"adding {kb_file} to db")
                kb_service.build_index(index_name, chunks)
            else:
                continue
        except Exception as e:
            print(f"Error processing {kb_file}: {e}", flush=True)
            continue


def folder2db(kb_names, retrieval_type, es_client, index_name, embed_model):
    kb_names = kb_names or list_kbs_from_folder()
    
    for kb_name in kb_names:
        kb_files = list_files_from_folder(kb_name)
        files2db(retrieval_type, es_client, index_name, kb_files, embed_model)


es_client = Elasticsearch("https://localhost:9200",
                          http_auth=("elastic", ""),
                          verify_certs=False)

if es_client.ping():
    print("Successfully connected to Elasticsearch!")
    embed_model = load_embeddings(model=EMBED_CONFIG["embed_model"], device=EMBED_CONFIG["embed_device"])
else:
    print("Could not connect to Elasticsearch.")
    
if __name__ == "__main__":
    folder2db(None, retrieval_type="vector", es_client=es_client, index_name="index", embed_model=embed_model)