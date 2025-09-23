import requests
from typing import List
from langchain.embeddings.base import Embeddings
from config import SILICON_API_KEY

class ProBAAIbgem3Embeddings(Embeddings):
    def __init__(self, api_key: str = SILICON_API_KEY):
        self.api_key = api_key
        self.url = "https://api.siliconflow.cn/v1/embeddings"
        self.max_retries = 3

    def _truncate_text(self, text: str) -> str:
        return text[:8000]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        for text in texts:
            text = self._truncate_text(text)
            payload = {"model": "Pro/BAAI/bge-m3", "input": text, "encoding_format": "float"}

            for _ in range(self.max_retries):
                try:
                    response = requests.post(self.url, json=payload, headers=headers, timeout=30)
                    if response.status_code == 200:
                        embeddings.append(response.json()['data'][0]['embedding'])
                        break
                except Exception as e:
                    print(f"Request failed: {str(e)}")
            else:
                embeddings.append([])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
