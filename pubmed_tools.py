import time
from Bio import Entrez
from typing import Dict, List, Optional, Tuple
from datetime import datetime

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