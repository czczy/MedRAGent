import pandas as pd
from datetime import datetime
from config import DEFAULT_LOGIC_RULES
from query_generator import get_search_terms
from pubmed_tools import fetch_pmids, fetch_articles, parse_article
from analysis_tools import analyze_articles, save_results

def main():
    print("=== Research Analysis System ===")

    # ==================== User Input Area ====================
    # Input complete PICOS criteria and exclusion criteria
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
        'custom': {}
                }
    #  custom_rules = {
    # 'default': {'within': 'OR', 'between': 'AND'},
    # 'custom': {
    # 'P': [
    #  {
    # 'terms': ['word A', 'word B'],
    # 'logic': 'OR',
    # 'group_name': 'countries'
    # },
    # {
    # 'terms': ['word C', 'word D'],
    # 'logic': 'OR',
    # 'group_name': 'ages'
    # }
    # ],
    # 'I': [
    #  {
    #  'terms': ['word E', 'word F'],
    # 'logic': 'OR',
    # 'group_name': 'interventions'
    #  }
    #  ]
    # }
    #  } indicating that ((word A OR word B) AND (word C OR word D)) AND (word E or word F). If 'custom' is {}, the query would be (word A) AND (word B) AND (word C) AND (word D) AND (word E) AND (word F).

    suffix_rules = {
    }

    # suffix_rules = {
    # 'P': {
    # '*': ['[MeSH Terms]', '[Title/Abstract]']
    #  },
    #  'I': {
    # 'word A': ['[MeSH Terms]']
    #  }
    # }   indicating that all words in P module add suffixes '[MeSH Terms]' and '[Title/Abstract]', word A in P module add suffix '[MeSH Terms]'.

    try:
        # Step 1: Generate search query (pass suffix rules)
        print("\n[Phase 1] Generating PubMed search query...")
        search_query = get_search_terms(
            picos_input=FULL_PICOS,
            selected_sections=SELECTED_SECTIONS,
            time_range=("2001/05/01", "2024/02/13"),
            logic_rules=DEFAULT_LOGIC_RULES, #DEFAULT_LOGIC_RULES: use "AND" between different words, and "OR" between the same words; custom_rules: user to define these relationships manually.
            suffix_rules=suffix_rules
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
            inclusion_criteria=FULL_PICOS,
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

if __name__ == "__main__":
    main()
