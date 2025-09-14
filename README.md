# MedRAGent
This is an automatic literature retrieval and screening system utilizing large language models with retrieval-augmented generation.

# Features
Data loading and processing with python

Screening report generation based on predefined PICOS, exclusion criteria, retrieval time, suffix rule and connection rule

Integration with word vector document, PubMed API and LLM API for automated Boolean query construcution and literature screening

# Prerequisites
Python 3.x

An Entrez API key and Entrez email address with access to PubMed

Any LLM API key (we use DeepSeek-V3-0324 and Kimi-K2-0711-preview)

# Installation
Clone the repository

Install the required dependencies

# Usage
Update data_loader.py with your Excel file's directory and name

Define your criteria in screening_text_generator.py

Set your OpenAI API key in gpt_integration.py

Run the scripts in the following order:

bash
python data_loader.py
python screening_text_generator.py
python gpt_integration.py
Contributing
Contributions are welcome! Please fork the repository and submit pull requests with your improvements.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.
