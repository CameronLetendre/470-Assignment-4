
# README

## Description

This script ranks documents for relevance to queries using the FLAN-T5 model. It processes two sets of queries and documents and saves the relevance scores in two separate output files. The script is computationally intensive and can take 4 hours or more to complete. Note that 2 separate files are provided for each prompt. The files were run in colab, we recommend that, but py files are provided. 

## Requirements

- **Data Files**: 
  - `result_bi_1.tsv` and `result_bi_2.tsv` – Required to run the code (available on Brightspace).
  - `topics_1.json`, `topics_2.json`, and `Answers.json` – Contain query and document text data.

- **Model**:
  - FLAN-T5 (`flan-t5-large` or `flan-t5-xl`) from Hugging Face Transformers library.
  - GPU recommended for faster processing.

## Instructions

1. **Install Dependencies**:
   - Ensure `transformers`, `tqdm`, `pandas`, `beautifulsoup4`, and `regex` libraries are installed.
   - Run the following command if necessary:
     ```bash
     pip install transformers tqdm pandas beautifulsoup4 regex
     ```

2. **Place Data Files**:
   - Download `result_bi_1.tsv` and `result_bi_2.tsv` from Brightspace and place them in the same directory as the script.
   - Ensure `topics_1.json`, `topics_2.json`, and `Answers.json` are also available in the same directory.

3. **Run the Script**:
   - Execute the script in a terminal or Python environment:
     ```bash
     python <script_name>.py
     ```
   - Note: The script may take 4+ hours to complete due to model processing time.

4. **Output**:
   - Two output files will be generated, each corresponding to one topics file:
     - `generated_scores_1.tsv` – Relevance scores for `topics_1.json` using `result_bi_1.tsv`.
     - `generated_scores_2.tsv` – Relevance scores for `topics_2.json` using `result_bi_2.tsv`.
   - **File Conversion**:
     - For evaluation, convert the output files to `prompt1_1.tsv` and `prompt2_2.tsv` as required by the evaluation script. This conversation is handled by a function in the provided script called format_to_trec. 

## Notes

- **GPU Recommended**: For faster performance, run the script on a machine with a GPU.
- **Progress Monitoring**: The script displays progress for each query and document, allowing you to monitor its status as it runs.
