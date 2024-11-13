import ranx
import matplotlib.pyplot as plt
import pandas as pd
import csv
from collections import defaultdict

metrics = ["precision@1", "precision@5", "ndcg@5", "mrr", "map"]

def calculate_metrics(qrel_path, topics_path):
    qrels = ranx.Qrels.from_file(qrel_path, kind="trec")
    results = ranx.Run.from_file(topics_path, kind="trec")

    evaluation = ranx.evaluate(qrels, results, metrics=metrics)

    all_metrics = list(evaluation.keys())

    metric_values = [evaluation[metric] for metric in all_metrics]

    for metric, value in zip(all_metrics, metric_values):
        print(f"{metric}: {value}")

    return metrics

def calculate_precision_at_5(qid, ranked_docs, qrels):
    """Calculate Precision@5 for a single query."""
    relevant_count = 0
    retrieved_count = min(5, len(ranked_docs))
    for doc_id, _ in ranked_docs[:retrieved_count]:
        doc_id = str(doc_id)
        if qid in qrels and doc_id in qrels[qid]:
            relevance = qrels[qid][doc_id]
            if relevance > 0:
                relevant_count += 1
    precision = relevant_count / retrieved_count if retrieved_count > 0 else 0
    return precision

def process_tsv(input_path, output_path):
    # Read the TSV file
    with open(input_path, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='\t')
        
        # Dictionary to store the first 5 entries for each unique value in the first column
        data = defaultdict(list)
        
        for row in reader:
            key = row[0]
            if len(data[key]) < 5:
                # Only take the first and third columns
                data[key].append([row[0], row[2], row[4]])
    
    # Write the processed data to a new TSV file
    with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        for key in data:
            for row in data[key]:
                writer.writerow(row)

def load_tsv(file_path):
    return pd.read_csv(file_path, sep='\t', header=None, names=['query_id', 'Q0', 'doc_id', 'rank', 'score', 'run_id'])

def calculate_p_at_5(tuned_cr, test_qrel):
    # Load the data
    tuned_cr_df = load_tsv(tuned_cr)
    test_qrel_df = load_tsv(test_qrel)

    # Dictionary to store the first 5 entries for each unique query_id
    data = defaultdict(list)
    
    for _, row in tuned_cr_df.iterrows():
        query_id = row['query_id']
        if len(data[query_id]) < 5:
            data[query_id].append(row['score'])
    
    # Calculate P@5 for each query_id
    p_at_5_scores = {}
    for query_id, scores in data.items():
        relevant_docs = test_qrel_df[test_qrel_df['query_id'] == query_id]['doc_id'].tolist()
        retrieved_docs = tuned_cr_df[tuned_cr_df['query_id'] == query_id]['doc_id'].tolist()[:5]
        relevant_retrieved_docs = [doc for doc in retrieved_docs if doc in relevant_docs]
        p_at_5 = len(relevant_retrieved_docs) / 5
        p_at_5_scores[query_id] = p_at_5

    return p_at_5_scores

def plot_precision_at_5(p_at_5_scores, sample_size=50):
    # Sort the scores in descending order
    p_at_5_scores_sorted = sorted(p_at_5_scores.items(), key=lambda x: x[1], reverse=True)

    # Systematically sample a subset of the data
    step = max(1, len(p_at_5_scores_sorted) // sample_size)
    p_at_5_scores_sampled = p_at_5_scores_sorted[::step]

    # Extract query IDs and scores for plotting
    query_ids = [query_id for query_id, _ in p_at_5_scores_sampled]
    scores = [score for _, score in p_at_5_scores_sampled]

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(p_at_5_scores_sampled)), scores)
    plt.xlabel('Query Topic IDs')
    plt.ylabel('P@5 Score')
    plt.title('P@5 Score for Each Query Topic')
    plt.xticks(range(len(p_at_5_scores_sampled)), query_ids, rotation=90)  # Add query IDs to x-axis
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig('p_at_5_scores_plot.png')
    plt.show()

def main():
    qrel_path = 'qrel_1.tsv'
    topics_path = 'untuned_re_results.csv'

    #process_tsv('tuned_cr.csv', 'top_5_tuned_cr.csv')
    p_at_5_scores = calculate_p_at_5(topics_path, qrel_path)
    plot_precision_at_5(p_at_5_scores, sample_size=50)

if __name__ == '__main__':
    main()