import time
import requests
from typing import Optional, List

def get_mesh_id(keyword: str) -> Optional[str]:
    """Get MeSH term ID (with retry mechanism)"""
    BASE_URL = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
    max_retries = 3
    retry_delay = 2  # Initial delay 2 seconds

    for attempt in range(max_retries + 1):  # 3 retries + initial attempt
        try:
            response = requests.get(
                BASE_URL,
                params={'label': keyword, 'match': 'exact', 'year': 'current'},
                timeout=20
            )
            if response.status_code == 200:
                for item in response.json():
                    if item.get('label', '').lower() == keyword.lower():
                        return item.get('resource', '').split('/')[-1]
            return None  # No matching item found

        except Exception as e:
            if attempt < max_retries:
                print(f"MeSH ID query timeout, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"MeSH ID query finally failed: {str(e)}")
                return None
    return None

def query_mesh_term(mesh_id: str) -> List[str]:
    """Query term details (with retry mechanism)"""
    BASE_URL = "https://id.nlm.nih.gov/mesh/lookup/details"
    max_retries = 3
    retry_delay = 2  # Initial delay 2 seconds

    for attempt in range(max_retries + 1):  # 3 retries + initial attempt
        try:
            response = requests.get(
                BASE_URL,
                params={'descriptor': mesh_id, 'includes': 'terms'},
                timeout=20  # Increase timeout
            )
            if response.status_code == 200:
                return [term['label'] for term in response.json().get('terms', [])]
            return []  # No results

        except Exception as e:
            if attempt < max_retries:
                print(f"Term query timeout, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Term query finally failed: {str(e)}")
                return []
    return []
