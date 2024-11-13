# -*- coding: utf-8 -*-
"""Assignment4_Prompt_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SFrGcW4Uz8DQcsgvSV-7LKQCfhgCyR60
"""

!pip install transformers datasets torch
!pip install huggingface_hub
!pip install datasets
!pip install pytrec_eval
!pip install tqdm
!pip install ranx
from huggingface_hub import login

#login("")

import re
from bs4 import BeautifulSoup

def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    # Remove non-alphabetic characters and retain spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    return text.strip()

import pandas as pd
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import re
from bs4 import BeautifulSoup
from tqdm import tqdm

# Load the FLAN-T5 model (use "flan-t5-large" or "flan-t5-xl" depending on available resources)
model_name = "google/flan-t5-large"  l
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)

df1 = pd.read_csv("result_bi_1.csv", delimiter=r"\t", header=None)
df1.columns = ["query_id", "unknown_col", "answer_id", "rank", "score", "method"]

df2 = pd.read_csv("result_bi_2.csv", delimiter=r"\t", header=None)
df2.columns = ["query_id", "unknown_col", "answer_id", "rank", "score", "method"]

with open("topics_1.json", "r") as f:
    topics_1 = json.load(f)

with open("topics_2.json", "r") as f:
    topics_2 = json.load(f)

with open("Answers.json", "r") as f:
    answers = json.load(f)

# Convert topics and answers into dictionaries for quick lookup
def load_topics_dict(topics):
    return {str(topic["Id"]): topic["Title"] + " " + topic["Body"] + " " + " ".join(topic["Tags"]) for topic in topics}

topics_dict_1 = load_topics_dict(topics_1)
topics_dict_2 = load_topics_dict(topics_2)
answers_dict = {str(answer["Id"]): answer["Text"] for answer in answers}

# Function to clean text by removing HTML tags and non-alphabet characters
def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.strip()

# Define function to generate relevance scores using FLAN-T5
def generate_relevance_score_pipeline(query, doc_text):
    if doc_text is None:
        print("Warning: Document text is missing for an answer ID.")
        return 0.0

    query = clean_text(query)
    doc_text = clean_text(doc_text)

    prompt = f"""
    Please analyze the document based on the query provided and give a relevance score of a float between 0, or 2. Higher scores mean higher relevance.

    Here is an example for reference:
    Example 1:
    Query: "Prepay simcard for data in Australia"
    Document: "<p>I'm arriving in Sydney on Friday for a week. Ideally I'd like a pre-pay simcard for text and data (I need to check my stackexchange sites of course ;)).</p><p>However, I have no idea of what companies or options exist, and what would be the best for a traveller in terms of moderate data usage.</p><p>Also, is it possible to buy these at the airport on arrival?</p>"
    Tags: ['australia', 'cellphones', 'data-plans', 'pre-pay']
    Document Text: "<p>I did some research, and it looks like there are (at least) three major carriers that offer affordable pre-paid SIM cards with data plans:</p><ul><li><a href='http://www.virginmobile.com.au/prepaid-mobile-phones-plans/' rel='nofollow'>Virgin Mobile</a></li><li><a href='http://shop.vodafone.com.au/all-broadbands?id=700016' rel='nofollow'>Vodafone</a></li><li><a href='http://goo.gl/3vjf7' rel='nofollow'>Telstra</a></li></ul>"
    relevence: 2

    Go through the process of the relevence and then evaluate:

    Query: '{query}'
    Document: '{doc_text}'
    Relevance:
    """

    response = pipe(prompt, max_new_tokens=5, do_sample=False)[0]["generated_text"]

    # Extract the numerical score from the response (if neccesary)
    try:
        score = float(response.strip())  # Convert to float if possible
    except ValueError:
        score = 0.0  # Default to 0.0 if parsing fails

    return score

# Function to generate and save scores for a given topics dictionary and results DataFrame
def process_topics(topics_dict, results_df, output_filename):
    scored_results_all = {}

    for query_id in tqdm(results_df["query_id"].unique(), desc=f"Processing {output_filename}"):
        query_text = topics_dict.get(str(query_id), "")
        scored_results = []

        for _, row in results_df[results_df["query_id"] == query_id].iterrows():
            doc_id = str(row["answer_id"])
            doc_text = answers_dict.get(doc_id)

            # Generate relevance score for the document
            relevance_score = generate_relevance_score_pipeline(query_text, doc_text)
            scored_results.append((doc_id, relevance_score))

        # Save relevance scores for each query
        scored_results_all[query_id] = scored_results

    # Save relevance scores to a TSV file
    # This is wrong format, have a function to put in right format
    with open(output_filename, "w") as f:
        for query_id, doc_scores in scored_results_all.items():
            for doc_id, score in doc_scores:
                f.write(f"{query_id}\t0\t{doc_id}\t{score}\n")

# Process topics_1.json with result_bi_1.csv and topics_2.json with result_bi_2.csv, saving results in separate files
process_topics(topics_dict_1, df1, "generated_scores_1.tsv")
process_topics(topics_dict_2, df2, "generated_scores_2.tsv")

import pandas as pd

def format_to_trec(input_file, output_file, run_id="run1"):
    """
    Converts a results file into TREC format for Ranx compatibility.

    Parameters:
    - input_file: Path to the input TSV file.
    - output_file: Path to the output formatted TSV file.
    - run_id: Identifier for the run (default: "run1").
    """

    df = pd.read_csv(input_file, delimiter="\t", header=None)
    df.columns = ["query_id", "unused_col", "doc_id", "score"]

    df = df.sort_values(by=["query_id", "score"], ascending=[True, False])

    df["rank"] = df.groupby("query_id").cumcount() + 1

    df["Q0"] = "Q0"
    df["run_id"] = run_id

    df = df[["query_id", "Q0", "doc_id", "rank", "score", "run_id"]]

    df.to_csv(output_file, sep="\t", header=False, index=False)
    print(f"Formatted run file saved as '{output_file}'")


format_to_trec("generated_scores_1.tsv", "prompt2_1.tsv", run_id="run1")
format_to_trec("generated_scores_2.tsv", "prompt2_2.tsv", run_id="run2")