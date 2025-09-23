import json
import time
import pandas as pd
from typing import Dict, List, Union
from langchain.schema import HumanMessage
from requests.exceptions import RequestException
from json_repair import loads as json_repair_loads, repair_json

from config import DEEPSEEK_API_KEY
from mesh_supply_retriever import llm_deepseek

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

            # Model call for analysis (with retry mechanism)
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
