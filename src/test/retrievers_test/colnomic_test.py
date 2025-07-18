import os
import sys
import json
import pprint
from collections import OrderedDict

sys.path.append(os.path.abspath("/home/laura/vqa-ir-qa/src"))

from retrievers.colnomic import ColNomicRetriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder

if __name__ == "__main__":
    csv_path = "/home/laura/vqa-ir-qa/src/dataset/chunks/chunked_pages.csv"
    image_dir = "/home/laura/vqa-ir-qa/data/pages"
    output_path = "/home/laura/vqa-ir-qa/data/sorted_scores_colnomic.json"
    k_values = [1, 3, 5, 10]

    provider = DocumentProvider(csv_path)
    print("Document stats:", provider.stats)

    queries, qrels = QueryQrelsBuilder(csv_path).build()

    retriever = ColNomicRetriever(
        provider=provider,
        image_dir=image_dir,
        model_name="nomic-ai/colnomic-embed-multimodal-3b",
        device_map="cuda",
        batch_size=8,
    )

    run = retriever.search(queries, batch_size=8)

    # ✅ Sort run results by descending score (IMPORTANT!)
    sorted_run = {
        qid: OrderedDict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        for qid, scores in run.items()
    }

    # ✅ Save using query text as key (for human readability)
    sorted_data = {
        queries.get(qid, qid): sorted_run[qid]
        for qid in sorted_run
    }

    pprint.pprint(sorted_data)

    with open(output_path, "w") as f:
        json.dump(sorted_data, f, indent=4)

    # 📊 Show sample results
    print("\n=== Sample Query Results (run) ===")
    for qid, docs in list(run.items())[:3]:
        print(f"Query ID: {qid}")
        print(f"Query Text: {queries.get(qid)}")
        print("Top Retrieved Docs:")
        print(list(docs.items())[:5])

    print("\n=== Ground Truth Relevance (qrels) ===")
    for qid, docs in list(qrels.items())[:3]:
        print(f"Query ID: {qid}")
        print(docs)

    print("\n=== Overlap Between Run and Qrels ===")
    common_docs = {
        qid: set(run.get(qid, {}).keys()) & set(qrels.get(qid, {}).keys())
        for qid in qrels
    }
    for qid, common in common_docs.items():
        print(f"{qid}: {len(common)} hits")

    # ✅ Evaluate using sorted_run (uses qid as keys, required by evaluator)
    metrics = retriever.evaluate(sorted_run, qrels, k_values, verbose=True)
