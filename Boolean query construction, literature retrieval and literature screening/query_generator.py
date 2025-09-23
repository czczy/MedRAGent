import re
from typing import Dict, List, Optional, Union, Tuple
from config import DEFAULT_LOGIC_RULES
from mesh_supply_retriever import EnhancedMeshRetriever
from mesh_supply_tools import get_mesh_id, query_mesh_term

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
