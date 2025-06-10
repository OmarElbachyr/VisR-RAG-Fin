from evaluation.document_provider import DocumentProvider


if __name__ == "__main__":
    provider = DocumentProvider("src/dataset/chunked_pages.csv", use_nltk_preprocessor=True)
    # print(len(provider.ids), "docs", len(provider.chunk_to_page), "chunk→page pairs")
    # first_id = provider.ids[0]
    # print(first_id, "-> page", provider.chunk_to_page[first_id])
    ids, tokens = provider.get(kind="bm25")
    print(len(ids), "docs with tokens")
    print(len(tokens), "tokenized docs")

    print(len(tokens[0]), "tokens in first doc")
    print(len(tokens[1]), "tokens in 2nd doc")
    print(len(tokens[2]), "tokens in 3d doc")