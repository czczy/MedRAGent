import os
import ssl
import certifi
from Bio import Entrez

# ========================== Configuration Section ==========================
# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths using relative paths
MAIN_INDEX_PATH = os.path.join(script_dir, "mesh_index.faiss")
MAIN_METADATA_PATH = os.path.join(script_dir, "mesh_metadata.pkl")
SUPPLE_INDEX_PATH = os.path.join(script_dir, "supple_index.faiss")
SUPPLE_METADATA_PATH = os.path.join(script_dir, "supple_metadata.pkl")

# API Key Configuration
SILICON_API_KEY = ""
DEEPSEEK_API_KEY = ""

Entrez.email = ""
Entrez.api_key = ""

# Default Logic Rules Configuration
DEFAULT_LOGIC_RULES = {
    'default': {'within': 'OR', 'between': 'AND'},
    'custom': {}
}

# SSL Configuration
ctx = ssl.create_default_context(cafile=certifi.where())

