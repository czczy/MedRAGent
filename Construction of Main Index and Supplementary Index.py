import xml.etree.ElementTree as ET
import numpy as np
import faiss
import pickle
import requests
from typing import List
from langchain.embeddings.base import Embeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
import os
# ========================== Configuration Section ==========================
# Get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use relative paths
XML_PATH = os.path.join(BASE_DIR, "")  # supp2025.xml or desc2025.xml
SAVE_INDEX_PATH = os.path.join(BASE_DIR, "")  # supple_index.faiss or mesh_index.faiss
SAVE_METADATA_PATH = os.path.join(BASE_DIR, "")  # supple_metadata.pkl or mesh_metadata.pkl
SILICON_API_KEY = ''  # Replace with actual API key
DEEPSEEK_API_KEY = ''  # Replace with actual API key
CHUNK_SIZE = 100  # User-configurable chunk size [New configuration parameter]

# ==================== SiliconFlow Embedding Model Wrapper ====================
class ProBAAIbgem3Embeddings(Embeddings):
    def __init__(self, API_KEY_SILICON: str):
        self.api_key = API_KEY_SILICON
        self.url = "https://api.siliconflow.cn/v1/embeddings"
        self.max_retries = 3

    def _truncate_text(self, text: str) -> str:
        return text[:8000]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        for i, text in enumerate(texts):
            text = self._truncate_text(text)
            payload = {"model": "Pro/BAAI/bge-m3", "input": text, "encoding_format": "float"}

            for _ in range(self.max_retries):
                try:
                    response = requests.post(self.url, json=payload, headers=headers, timeout=30)
                    if response.status_code == 200:
                        emb = response.json()['data'][0]['embedding']
                        if len(emb) == 1024:
                            embeddings.append(emb)
                            break
                        else:
                            print(f"Abnormal dimension returned for text {i}: {len(emb)}")
                    else:
                        print(f"Request failed, status code: {response.status_code}, response: {response.text}")
                except Exception as e:
                    print(f"Request failed: {str(e)}")
            else:
                print(f"Embedding generation failed for text [{text[:50]}...], skipping")
                embeddings.append([])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ======================== Data Processing Section ========================
def parse_mesh_xml(xml_path: str):
    """Parse MeSH XML file and extract text content and metadata"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    texts = []
    metadata = []

    for record in root.findall('SupplementalRecord'):
        ui = record.find('SupplementalRecordUI').text
        name = record.find('SupplementalRecordName/String').text
        concept_list = record.find('ConceptList')

        tree_numbers = []
        if concept_list is not None:
            for concept in concept_list.findall('Concept'):
                for tn in concept.findall('TreeNumberList/TreeNumber'):
                    tree_numbers.append(tn.text)

        text_content = f"{name} [Classification: {' '.join(tree_numbers)}]" if tree_numbers else name
        texts.append(text_content)
        metadata.append({'UI': ui, 'Name': name})

    return texts, metadata


def create_vector_store(texts: List[str], metadata: List[dict], embedder: ProBAAIbgem3Embeddings,
                        chunk_size: int = CHUNK_SIZE):  # [Modified function signature]
    """Create vector store and save to local"""
    dimension = 1024  # Set according to embedding model dimension
    index = faiss.IndexFlatIP(dimension)
    valid_metadata = []

    # Process texts in chunks [New chunking logic]
    for i in range(0, len(texts), chunk_size):
        batch_texts = texts[i:i + chunk_size]
        batch_meta = metadata[i:i + chunk_size]

        print(f"Processing batch {i // chunk_size + 1}/{(len(texts) - 1) // chunk_size + 1} ({len(batch_texts)} texts)")

        embeddings = embedder.embed_documents(batch_texts)

        # Filter and collect valid data
        valid_embeddings = []
        valid_batch_meta = []
        for emb, meta in zip(embeddings, batch_meta):
            if len(emb) == 1024:
                valid_embeddings.append(emb)
                valid_batch_meta.append(meta)
            else:
                print(f"Skipping invalid entry: {meta['Name'][:50]}...")

        if not valid_embeddings:
            print("No valid embeddings in current batch, skipping")
            continue

        # Convert to numpy array and normalize
        embeddings_np = np.array(valid_embeddings, dtype='float32')
        faiss.normalize_L2(embeddings_np)

        # Add to index and collect metadata
        index.add(embeddings_np)
        valid_metadata.extend(valid_batch_meta)

    # Save index and metadata
    faiss.write_index(index, SAVE_INDEX_PATH)
    with open(SAVE_METADATA_PATH, 'wb') as f:
        pickle.dump(valid_metadata, f)
    print(f"Vector store creation completed, processed {len(valid_metadata)} valid entries"

# ======================== Execution Flow ========================
if __name__ == "__main__":
    # Process XML and create vector store
    texts, metadata = parse_mesh_xml(XML_PATH)
    embedder = ProBAAIbgem3Embeddings(SILICON_API_KEY)
    create_vector_store(texts, metadata, embedder, CHUNK_SIZE)  # [Modified call method]
