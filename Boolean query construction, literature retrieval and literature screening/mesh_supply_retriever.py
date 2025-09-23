import re
import numpy as np
import faiss
import pickle
from typing import Dict, List, Optional
from langchain.schema import HumanMessage
from langchain_openai.chat_models.base import BaseChatOpenAI

from config import MAIN_INDEX_PATH, MAIN_METADATA_PATH, SUPPLE_INDEX_PATH, SUPPLE_METADATA_PATH, DEEPSEEK_API_KEY
from embeddings import ProBAAIbgem3Embeddings

# Initialize LLM
import os
os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com" # If you want to use kimi, input "https://api.moonshot.cn/v1"
llm_deepseek = BaseChatOpenAI(model='deepseek-chat', max_retries=3) # If you want to use kimi, input "kimi-k2-0711-preview"

class EnhancedMeshRetriever:
    def __init__(self):
        # Load index and metadata
        self.main_index = faiss.read_index(MAIN_INDEX_PATH)
        self.supple_index = faiss.read_index(SUPPLE_INDEX_PATH)

        with open(MAIN_METADATA_PATH, 'rb') as f:
            self.main_metadata = pickle.load(f)
        with open(SUPPLE_METADATA_PATH, 'rb') as f:
            self.supple_metadata = pickle.load(f)

        self.embedder = ProBAAIbgem3Embeddings()
        self.llm = llm_deepseek

    def extract_nouns(self, text: str) -> List[str]:
        """Improved compound noun extraction method"""
        prompt = f"""
            Please extract all nouns from the text, following these rules:  
                1. Keep compound nouns intact (e.g., randomised clinical trials → Randomized Controlled Trials, before surgery → before surgery)  
                2. Do not split phrases connected by conjunctions  
                3. Output one noun per line  
                4. No explanation needed  
                Text content: {text}
            """

        response = self.llm.invoke([HumanMessage(content=prompt)]).content
        return self._postprocess_nouns(response)

    def _postprocess_nouns(self, raw_output: str) -> List[str]:
        """Noun post-processing"""
        # Common compound terms mapping table
        COMPOUND_TERMS = {
            'rct': 'Randomized Controlled Trial',
            'randomised clinical trials': 'Randomized Controlled Trial'
        }

        nouns = [
            re.sub(r'^[\W_]+', '', n.strip())  # Remove leading non-alphanumeric characters
            for n in raw_output.split('\n')
            if n.strip()
        ]
        return [
            COMPOUND_TERMS.get(n.lower(), n.capitalize())
            for n in nouns
            if n  # Filter empty strings
        ]

    def search_mesh(self, query: str, threshold: float = 0.6) -> Dict[str, Optional[str]]:
        """Enhanced term retrieval"""
        nouns = self.extract_nouns(query)
        results = {}
        print("Extracted nouns:", nouns)

        for noun in nouns:
            # Generate query vector
            query_vec = self.embedder.embed_query(noun)
            query_vec = np.array(query_vec, dtype='float32').reshape(1, -1)
            faiss.normalize_L2(query_vec)

            # First query main index
            main_distances, main_indices = self.main_index.search(query_vec, k=1)

            if main_distances[0][0] >= threshold:
                results[noun] = self.main_metadata[main_indices[0][0]]['Name']
                print(f"[Main index match] {noun} -> {results[noun]} (similarity: {main_distances[0][0]:.4f})")
            else:
                # Query supplementary index if not found in main index
                supple_distances, supple_indices = self.supple_index.search(query_vec, k=1)

                if supple_distances[0][0] >= threshold:
                    results[noun] = self.supple_metadata[supple_indices[0][0]]['Name']
                    print(f"[Supplementary index match] {noun} -> {results[noun]} (similarity: {supple_distances[0][0]:.4f})")
                else:
                    results[noun] = None
                    print(
                        f"[No match] {noun} Main index similarity: {main_distances[0][0]:.4f}, Supplementary index similarity: {supple_distances[0][0]:.4f}")

        return results

