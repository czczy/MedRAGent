import os
import re
import time
import pickle
import requests
import numpy as np
import faiss
from typing import Dict, List, Optional, Union, Tuple
from langchain.schema import HumanMessage
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.embeddings.base import Embeddings
import ssl
import certifi
from Bio import Entrez
import json
from datetime import datetime
from json_repair import loads as json_repair_loads, repair_json
import pandas as pd
from requests.exceptions import RequestException
# ========================== Configuration Section ==========================
# MeSH Index Configuration
# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths using relative paths
MAIN_INDEX_PATH = os.path.join(script_dir, "mesh_index.faiss")
MAIN_METADATA_PATH = os.path.join(script_dir, "mesh_metadata.pkl")
SUPPLE_INDEX_PATH = os.path.join(script_dir, "supple_index.faiss")
SUPPLE_METADATA_PATH = os.path.join(script_dir, "supple_metadata.pkl")

# API Key Configuration
SILICON_API_KEY = ''
DEEPSEEK_API_KEY = ''
# KIMI_API_KEY = ''

Entrez.email = ""
Entrez.api_key = ""

# Default Logic Rules Configuration
DEFAULT_LOGIC_RULES = {
    'default': {'within': 'OR', 'between': 'AND'},
    'custom': {}
}

# Best Practice Solution
ctx = ssl.create_default_context(cafile=certifi.where())

# ======================== Initialize Models ========================
# Set up kimi
# os.environ["OPENAI_API_KEY"] = KIMI_API_KEY
# os.environ["OPENAI_API_BASE"] = "https://api.moonshot.cn/v1"
# llm_kimi = BaseChatOpenAI(model="kimi-k2-0711-preview", max_retries=3)

# DeepSeek Model
os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"
llm_deepseek = BaseChatOpenAI(model='deepseek-chat', max_retries=3)

# ==================== SiliconFlow Embedding Model Wrapper ====================
class ProBAAIbgem3Embeddings(Embeddings):
    def __init__(self, api_key: str):
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


# ======================== RAG Retriever ========================
class EnhancedMeshRetriever:
    def __init__(self):
        # Load index and metadata
        self.main_index = faiss.read_index(MAIN_INDEX_PATH)
        self.supple_index = faiss.read_index(SUPPLE_INDEX_PATH)

        with open(MAIN_METADATA_PATH, 'rb') as f:
            self.main_metadata = pickle.load(f)
        with open(SUPPLE_METADATA_PATH, 'rb') as f:
            self.supple_metadata = pickle.load(f)

        self.embedder = ProBAAIbgem3Embeddings(SILICON_API_KEY)
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


# ======================== MeSH API Tools ========================
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


# ======================== Search Query Generation ========================
def generate_search_query(
        term_groups: Dict[str, dict],
        time_condition: str = "",
        logic_rules: Dict = DEFAULT_LOGIC_RULES,
        suffix_rules: Optional[Dict] = None
) -> str:
    default_within = logic_rules['default']['within']
    default_between = logic_rules['default']['between']
    query_parts = []

    def standardize_term(term: str) -> str:
        """Standardize term to ensure matching with concepts keys"""
        term = re.sub(r'[^a-zA-Z0-9,\- ]', '', term.strip())
        term = re.sub(r'\s+', ' ', term)
        term = re.sub(r',(\w)', r', \1', term)
        return ' '.join(word.capitalize() if word.islower() else word
                        for word in term.split())

    def apply_suffix(term: str, suffixes: List[str]) -> str:
        """Apply suffix to single term"""
        if not suffixes:
            return term

        suffixed_terms = []
        for suffix in suffixes:
            if term.startswith('"') and term.endswith('"'):
                base_term = term[1:-1]
                suffixed_terms.append(f'"{base_term}{suffix}"')
            else:
                suffixed_terms.append(f'{term}{suffix}')

        return f"({' OR '.join(suffixed_terms)})"

    suffix_rules = suffix_rules or {}

    for section, data in term_groups.items():
        concepts = data['concepts']
        custom_rule = data['custom_rule']
        section_suffixes = suffix_rules.get(section, {})

        if custom_rule:
            query_subparts = []
            for rule in custom_rule:
                terms = rule['terms']
                logic_within = rule.get('logic', default_within)
                group_terms = []

                for term in terms:
                    standardized_term = standardize_term(term)
                    matched_key = None
                    for key in concepts.keys():
                        if key.lower() == standardized_term.lower():
                            matched_key = key
                            break

                    term_suffixes = section_suffixes.get(term, [])
                    if not term_suffixes and '*' in section_suffixes:
                        term_suffixes = section_suffixes['*']

                    if matched_key:
                        # Key modification: apply suffix to each synonym
                        for synonym in concepts[matched_key]:
                            syn_std = standardize_term(synonym)
                            group_terms.append(apply_suffix(syn_std, term_suffixes))
                    else:
                        # Add original term directly (apply suffix)
                        group_terms.append(apply_suffix(standardized_term, term_suffixes))

                unique_terms = list(set(group_terms))
                if unique_terms:
                    connection = f" {logic_within} "
                    sub_query = f"({connection.join(unique_terms)})"
                    query_subparts.append(sub_query)

            if query_subparts:
                query_part = f"({' AND '.join(query_subparts)})"
                query_parts.append(query_part)
        else:
            concept_queries = []
            for noun, synonyms in concepts.items():
                term_suffixes = section_suffixes.get(noun, [])
                if not term_suffixes and '*' in section_suffixes:
                    term_suffixes = section_suffixes['*']

                terms = []
                # Key modification: apply suffix to each synonym
                for syn in synonyms:
                    std_syn = standardize_term(syn)
                    terms.append(apply_suffix(std_syn, term_suffixes))

                if terms:
                    # Deduplicate and merge
                    unique_terms = list(set(terms))
                    concept_queries.append(f"({' OR '.join(unique_terms)})")

            if concept_queries:
                query_parts.append(f"({' AND '.join(concept_queries)})")

    final_query = f' {default_between} '.join(query_parts)
    if time_condition:
        final_query += f" AND {time_condition}"
    final_query = final_query.replace(' ,', ',')
    final_query = re.sub(r'(\w)(AND|OR)', r'\1 \2', final_query)
    return final_query


def get_search_terms(
        picos_input: Dict[str, str],
        selected_sections: List[str],
        time_range: Optional[Union[Tuple[int, int], Tuple[str, str]]] = None,
        logic_rules: Dict = DEFAULT_LOGIC_RULES,
        required_sections: List[str] = None,
        suffix_rules: Optional[Dict] = None
) -> Optional[str]:
    if not selected_sections:
        raise ValueError("At least one PICOS module must be selected")

    filtered_picos = {k: v for k, v in picos_input.items()
                      if k in selected_sections and v.strip()}

    if required_sections:
        missing = [s for s in required_sections if s not in filtered_picos]
        if missing:
            raise ValueError(f"Missing required modules: {', '.join(missing)}")

    time_condition = ""
    if time_range:
        start, end = time_range
        if any(isinstance(t, str) and '/' in t for t in time_range):
            time_condition = f'"{start}"[Date - Publication] : "{end}"[Date - Publication]'
        else:
            time_condition = f'"{start}"[Year] : "{end}"[Year]'

    retriever = EnhancedMeshRetriever()
    term_groups = {}

    for section in filtered_picos.keys():
        if not picos_input.get(section):
            continue

        print(f"\n=== Processing {section} module ===")
        mesh_map = retriever.search_mesh(picos_input[section])
        concept_groups = {}

        for noun, mesh_term in mesh_map.items():
            terms = []
            terms.append(noun)  # Keep original noun

            if mesh_term:
                terms.append(mesh_term)
                if mesh_id := get_mesh_id(mesh_term):
                    if syns := query_mesh_term(mesh_id):
                        # Key modification: add all expanded terms to the list
                        terms.extend(syns)
                        print(f"Term expansion: {noun} → {syns}")

            # Deduplicate storage
            concept_groups[noun] = list(set(terms))

        term_groups[section] = {
            'concepts': concept_groups,
            'custom_rule': logic_rules['custom'].get(section)
        }

    return generate_search_query(
        term_groups,
        time_condition,
        logic_rules,
        suffix_rules
    )


# ======================== PubMed Tools ========================
def fetch_pmids(search_query: str) -> Dict:
    """Optimized PMID retrieval, solving the 10,000 record limit issue"""
    MAX_RETRIES = 8
    BATCH_SIZE = 5000
    WAIT_TIME = 1.5

    # 1. Get total record count
    for attempt in range(MAX_RETRIES):
        try:
            handle = Entrez.esearch(db="pubmed", term=search_query, retmax=0)
            record = Entrez.read(handle)
            total = int(record["Count"])
            handle.close()
            print(f"Total matching documents: {total}")
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise Exception(f"Unable to get total document count: {str(e)}")
            time.sleep(2 ** attempt)

    # 2. If record count exceeds 9999, use time-sliced retrieval strategy
    if total > 9999:
        print(f"Record count exceeds 9999 ({total}), enabling time-sliced retrieval strategy...")
        return fetch_pmids_by_year_range(search_query, total)

    # 3. For small batches, use conventional method
    pmids = []
    batch_count = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(batch_count):
        retstart = batch_num * BATCH_SIZE
        actual_batch_size = min(BATCH_SIZE, total - retstart)

        for attempt in range(MAX_RETRIES):
            try:
                print(f"Fetching batch {batch_num + 1}/{batch_count} (ID {retstart + 1}-{retstart + actual_batch_size})")
                handle = Entrez.esearch(
                    db="pubmed",
                    term=search_query,
                    retstart=retstart,
                    retmax=BATCH_SIZE
                )
                record = Entrez.read(handle)
                pmids.extend(record["IdList"])
                handle.close()
                time.sleep(WAIT_TIME)
                break
            except Exception as e:
                wait = 3 + 2 * attempt
                print(f"Batch {batch_num + 1} fetch failed ({str(e)}). Waiting {wait} seconds to retry...")
                time.sleep(wait)
                if attempt == MAX_RETRIES - 1:
                    print(f"⚠️ Warning: Unable to fetch PMIDs for batch {retstart}-{retstart + BATCH_SIZE}")

    return {
        "total": total,
        "pmids": list(set(pmids)),
        "webenv": "",
        "query_key": ""
    }


def fetch_pmids_by_year_range(search_query: str, total: int) -> Dict:
    """Retrieve documents exceeding 9999 by year range"""
    MAX_RETRIES = 5
    YEAR_CHUNKS = 5  # Number of years to retrieve each time
    current_year = datetime.now().year
    start_year = 1800  # Earliest reasonable year

    all_pmids = []
    year_start = start_year
    total_fetched = 0

    print("Starting time-sliced retrieval...")

    while year_start <= current_year:
        year_end = min(year_start + YEAR_CHUNKS - 1, current_year)
        date_query = f'("{year_start}/01/01"[Date - Publication] : "{year_end}/12/31"[Date - Publication])'
        full_query = f"({search_query}) AND {date_query}"

        print(f"Retrieving documents from {year_start}-{year_end}...")

        for attempt in range(MAX_RETRIES):
            try:
                # Get document count for this time period
                handle = Entrez.esearch(db="pubmed", term=full_query, retmax=0)
                record = Entrez.read(handle)
                count = int(record["Count"])
                handle.close()

                if count == 0:
                    print(f"  {year_start}-{year_end}: 0 documents")
                    year_start = year_end + 1
                    break

                print(f"  {year_start}-{year_end}: {count} documents")

                # Get all PMIDs for this time period
                pmids = []
                for retstart in range(0, count, 5000):
                    handle = Entrez.esearch(
                        db="pubmed",
                        term=full_query,
                        retstart=retstart,
                        retmax=min(5000, count - retstart)
                    )
                    record = Entrez.read(handle)
                    pmids.extend(record["IdList"])
                    handle.close()
                    time.sleep(1)

                unique_pmids = list(set(pmids))
                all_pmids.extend(unique_pmids)
                total_fetched += len(unique_pmids)
                print(f"  Retrieved {len(unique_pmids)} documents, total {total_fetched}/{total}")
                year_start = year_end + 1
                time.sleep(2)
                break

            except Exception as e:
                wait = 5 + 3 * attempt
                print(f"Retrieval for {year_start}-{year_end} failed ({str(e)}). Waiting {wait} seconds to retry...")
                time.sleep(wait)
                if attempt == MAX_RETRIES - 1:
                    print(f"⚠️ Unable to retrieve documents from {year_start}-{year_end}, skipping this period")
                    year_start = year_end + 1

    return {
        "total": total,
        "pmids": list(set(all_pmids)),
        "webenv": "",
        "query_key": ""
    }


def fetch_articles(pmids: List[str]) -> List:
    """Optimized article retrieval with error handling and batch control"""
    articles = []
    BATCH_SIZE = 200  # Reduce batch size
    MAX_RETRIES = 6
    WAIT_TIME = 1.2

    total_batches = (len(pmids) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(pmids))
        batch = pmids[start:end]

        print(f"Fetching articles {start + 1}-{end}/{len(pmids)} (batch {batch_num + 1}/{total_batches})")

        for attempt in range(MAX_RETRIES):
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=batch,
                    rettype="xml",
                    retmode="xml"
                )
                data = Entrez.read(handle)
                articles.extend(data['PubmedArticle'])
                handle.close()
                time.sleep(WAIT_TIME)
                break
            except Exception as e:
                wait = 2 + 3 * attempt
                print(f"Article batch {batch_num + 1} fetch failed ({str(e)}). Waiting {wait} seconds to retry...")
                time.sleep(wait)
                if attempt == MAX_RETRIES - 1:
                    print(f"⚠️ Unable to fetch article batch {batch_num + 1}")

    return articles


def parse_article(article: Dict) -> Optional[Dict]:
    """Parse article and mark abstract existence"""
    try:
        medline = article['MedlineCitation']
        article_data = medline['Article']

        abstract_text = []
        has_abstract = False
        if 'Abstract' in article_data:
            for part in article_data['Abstract'].get('AbstractText', []):
                content = part.get('#text', '') if isinstance(part, dict) else str(part)
                if content.strip():
                    abstract_text.append(content)
            has_abstract = len(abstract_text) > 0

        return {
            "PMID": str(medline['PMID']),
            "Title": article_data.get('ArticleTitle', 'no title'),
            "Abstract": ' '.join(abstract_text) if has_abstract else "no abstract",
            "HasAbstract": has_abstract,
            "Journal": article_data['Journal']['Title'],

            "PubDate": str(article_data['Journal']['JournalIssue'].get('PubDate', '')),
            "Authors": [f"{auth.get('LastName', '')} {auth.get('ForeName', '')}"
                        for auth in article_data.get('AuthorList', [])],
            "FullXML": str(article)
        }
    except Exception as e:
        print(f"Article parsing failed: {str(e)}")
        return None


# ======================== Analysis Tools ========================
def parse_response(content: str) -> Dict:
    """Enhanced security JSON parser"""
    try:
        cleaned = content.strip()
        # Add maximum length check
        if len(cleaned) > 100000:
            return {"relevant": False, "error_info": "response_too_large"}

        # Add secure parsing
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                return json_repair_loads(cleaned)
            except:
                # Add repair attempt limit
                repaired = repair_json(cleaned, max_attempts=3)
                return json.loads(repaired)
    except Exception as e:
        return {"relevant": False, "error_info": f"parse_error: {str(e)}"}


def analyze_articles(
        articles: List[Dict],
        inclusion_criteria: Dict[str, str],
        exclusion_criteria: List[str]
) -> List[Dict]:
    """Complete analysis process (dynamically receive criteria)"""

    def format_criteria(criteria: Union[Dict, List]) -> str:
        """Standardize criteria format"""
        if isinstance(criteria, dict):
            return '\n'.join([f"{k.upper()}：{v}" for k, v in criteria.items() if v.strip()])
        return '\n'.join(exclusion_criteria)

    results = []
    MAX_ATTEMPTS = 3
    TIMEOUT = 60  # Set timeout

    for idx, article in enumerate(articles):
        # Basic information initialization
        base_info = {
            "PMID": article["PMID"],
            "Title": article["Title"],
            "Journal": article.get("Journal", ""),
            "PubDate": article.get("PubDate", ""),
            "Authors": article.get("Authors", []),
            "Abstract": article["Abstract"] if article["HasAbstract"] else "no abstract",
            "Analyzed": article["HasAbstract"],
            "Relevant": None,
            "Inclusion_Reasons": {},
            "Exclusion_Reasons": [],
            "Comparison_Match": ""
        }

        try:
            # Improved progress display
            if idx % 50 == 0 or idx == len(articles) - 1:
                print(f"\rAnalysis progress: {idx + 1}/{len(articles)} ({((idx + 1) / len(articles)) * 100:.1f}%)", end="",
                      flush=True)
            else:
                print(f"\rAnalysis progress: {idx + 1}/{len(articles)} ({((idx + 1) / len(articles)) * 100:.1f}%)", end="")

            # Skip articles without abstract
            if not article["HasAbstract"]:
                base_info["Exclusion_Reasons"] = ["no abstract"]
                results.append(base_info)
                continue

            # Dynamically generate analysis prompt
            prompt = f"""
            Literature Screening Rules Explanation:
            1. Inclusion Criteria Handling Principles:
                - All inclusion criteria must be met → Include the Literature
                - If the abstract and title contain explicit supportive descriptions → Consider the inclusion criterion met  
                - If the abstract and title contain explicit negative descriptions → Consider the inclusion criterion as unmet 
                - If no explicit supportive/negative description was found in the abstract and title → The inclusion criterion is considered met  

            2. Exclusion Criteria Handling Principles:
                - Meets any exclusion criterion → Exclude the Literature  
                - If the abstract and title contain explicit supportive descriptions → Consider the exclusion criterion met  
                - If the abstract and title contain explicit negative descriptions → Consider the exclusion criterion as unmet 
                - If no explicit supportive/negative descriptions are found in the abstract and title → The exclusion criteria is considered unmet    

            Analysis Criteria:  
            Inclusion Criteria:  
            {format_criteria(inclusion_criteria)}

            Exclusion Criteria:
            {format_criteria(exclusion_criteria)}

            You are a senior biostatistician. Please analyze whether the following literature meets the criteria according to the steps, rules, and standards above:  
            1. Check each inclusion criterion one by one:  
            - For each inclusion criterion, search for supportive/negative descriptions in the abstract and title, then judge the literature based on the inclusion criteria handling principles.  
            2. Check exclusion criteria:  
            - For each exclusion criterion, search for supportive/negative descriptions in the abstract and title, then judge the literature based on the exclusion criteria handling principles.  
            3. Final judgment:  
            - All inclusion criteria met + no exclusion criteria met → Include (true)  
            - Any inclusion criterion unmet → Exclude (false)  
            - Any exclusion criterion met → Exclude (false)
            4. Please return the result in English.  

            Please return the result in strict JSON format:  
            {{
                "relevant": true/false,
                "reasons": {{  
                    "population_match": "Whether the requirement is met (return an empty string if not applicable)",  
                    "intervention_match": "Whether the intervention content is included (return an empty string if not applicable)",  
                    "comparison_match": "Whether the comparison content is included (return an empty string if not applicable)",  
                    "outcome_match": "Whether the specified outcome measure is included (return an empty string if not applicable)",  
                    "study_design": "Whether the study design meets the requirements (return an empty string if not applicable)",  
                    "exclusion_reasons": [  
                    // If no exclusion reasons, return an empty string "", otherwise dynamically add multiple exclusion reasons  
                    "Reason 1: Specific description of the exclusion reason",  
                    "Reason 2: Specific description of the exclusion reason",  
                    "Reason 3: Specific description of the exclusion reason"  
                    ]  
            }}  
                 "metadata": {{
                    "pmid": "{article['PMID']}",
                    "title": {json.dumps(article['Title'], ensure_ascii=False)}
                }}
            }}

            Literature Abstract and Title: 
            {article['Abstract'], article['Title']}

            """

            # ===== Fix location: Model call inside each article loop =====
            # Call model for analysis (with retry mechanism)
            result = None
            for attempt in range(MAX_ATTEMPTS):
                try:
                    # Call with timeout
                    response = llm_deepseek.invoke(
                        [HumanMessage(content=prompt)],
                        timeout=TIMEOUT
                    )
                    result = parse_response(response.content)
                    break
                except (RequestException, TimeoutError) as e:
                    wait_time = min(2 ** attempt, 60)  # Maximum wait 60 seconds
                    print(f"\nArticle {article['PMID']} request timeout, waiting {wait_time} seconds to retry...")
                    time.sleep(wait_time)
                except Exception as e:
                    if attempt == MAX_ATTEMPTS - 1:
                        print(f"\nArticle {article['PMID']} analysis failed: {str(e)}")
                    else:
                        time.sleep(2 ** attempt)
            # ===== End of fix =====

            # Process analysis results
            if result:
                base_info.update({
                    "Relevant": result.get("relevant", False),
                    "Inclusion_Reasons": {
                        "population": result["reasons"].get("population_match", ""),
                        "intervention": result["reasons"].get("intervention_match", ""),
                        "comparison": result["reasons"].get("comparison_match", ""),
                        "outcome": result["reasons"].get("outcome_match", ""),
                        "study_design": result["reasons"].get("study_design", "")
                    },
                    "Exclusion_Reasons": [
                        reason for reason in result["reasons"].get("exclusion_reasons", [])
                        if reason.strip()
                    ]
                })

        except Exception as e:
            print(f"\nArticle {article['PMID']} processing exception: {str(e)}")
            base_info["Exclusion_Reasons"] = [f"Analysis exception: {str(e)}"]

        results.append(base_info)

    return results


def save_results(results: List[Dict], filename: str):
    """Enhanced result saving (add filename parameter)"""
    if not results:
        print("\nNo results to save")
        return

    try:
        processed_data = []
        for item in results:
            inclusion_status = "YES" if item["Relevant"] else "NO" if item["Analyzed"] else "Unanalyzed"

            row = {
                "PMID": item["PMID"],
                "Title": item["Title"],
                "Included": inclusion_status,
                "Population Match": item["Inclusion_Reasons"].get("population", ""),
                "Intervention Match": item["Inclusion_Reasons"].get("intervention", ""),
                "Comparison Match": item["Inclusion_Reasons"].get("comparison", ""),
                "Outcome Match": item["Inclusion_Reasons"].get("outcome", ""),
                "Study Design": item["Inclusion_Reasons"].get("study_design", ""),
                "Exclusion Reasons": "; ".join(item["Exclusion_Reasons"]),
                "Journal": item["Journal"],
                "Publication Date": item["PubDate"],
                "Authors": "; ".join(item["Authors"]),
                "Abstract": item["Abstract"]
            }
            processed_data.append(row)

        df = pd.DataFrame(processed_data)
        df.to_excel(filename, index=False)
        print(f"\nSuccessfully saved {len(df)} results to {filename}")

    except Exception as e:
        print(f"\nFailed to save results: {str(e)}")

# ======================== Main Workflow ========================
if __name__ == "__main__":
    print("=== Research Analysis System ===")

    # Configure validation file path
    VALIDATION_FILE = "Validation File (PubMed).xlsx"

    # ==================== User Input Area ====================
    # Input complete PICOS criteria and exclusion criteria (only define once here)
    FULL_PICOS = {
        'P': "Type 1 Diabetes Mellitus, adolescent",
        'I': "fasting during Ramadan",
        'C': "",
        'O': "hyperglycemia, hypoglycemia, diabetic ketoacidosis, changes in HbA1c, and weight changes",
        'S': "observational studies (e.g., prospective study, cohort studies, case-control studies, cross-sectional studies)"
    }

    EXCLUSION_CRITERIA = [
        "Animal studies",
        "Case reports",
        "Systematic reviews"
        "meta analysis"
    ]

    # Specify modules needed for search query generation (select from PICOS)
    SELECTED_SECTIONS = ['P', 'I']  # Only select needed modules

    # Updated custom rules example
    custom_rules = {
        'default': {'within': 'OR', 'between': 'AND'},
        'custom': {
        }
    }
    # Custom suffix rules
    suffix_rules = {
    }

    try:
        # Step 1: Generate search query (pass suffix rules)
        print("\n[Phase 1] Generating PubMed search query...")
        search_query = get_search_terms(
            picos_input=FULL_PICOS,
            selected_sections=SELECTED_SECTIONS,
            time_range=("2001/05/01", "2024/02/13"), # retriveal time
            logic_rules=custom_rules,  # DEFAULT_LOGIC_RULES
            suffix_rules=DEFAULT_LOGIC_RULES  # Pass custom suffix rules
        )

        if not search_query:
            raise ValueError("Search query generation failed")
        print(f"Generated search query:\n{search_query}")

        # Step 2: Get PMIDs
        print("\n[Phase 2] Getting document IDs...")
        search_result = fetch_pmids(search_query)
        print(f"Total matching documents: {search_result['total']}")

        # Step 3: Get document data
        print("\n[Phase 3] Downloading document data...")
        articles = fetch_articles(search_result["pmids"])
        parsed_articles = [art for art in (parse_article(a) for a in articles) if art]
        print(f"Valid documents: {len(parsed_articles)}")

        # Step 4: Analyze documents (dynamically pass criteria)
        print("\n[Phase 4] Performing deep analysis...")
        final_results = analyze_articles(
            parsed_articles,
            inclusion_criteria=FULL_PICOS,  # Use dynamically filtered criteria, inclusion_criteria_selected
            exclusion_criteria=EXCLUSION_CRITERIA
        )

        # Step 5: Save results
        print("\n[Phase 5] Saving analysis results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"report_Full_{timestamp}.xlsx"
        save_results(final_results, output_filename)

        # Final report
        print("\n=== Final Analysis Report ===")
        print(f"Total processed documents: {len(parsed_articles)}")
        print(f"Documents meeting criteria: {sum(1 for res in final_results if res['Relevant'])}")

        if final_results:
            exclusion_counts = pd.Series(
                [reason for art in final_results for reason in art["Exclusion_Reasons"]]
            ).value_counts()

    except Exception as e:
        print(f"\nSystem operation failed: {str(e)}")
    finally:

        print("\n=== Analysis process completed ===")
