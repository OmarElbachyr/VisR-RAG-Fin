import torch
from transformers import CLIPProcessor, CLIPModel
from rank_bm25 import BM25Okapi
import nltk
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn.functional as F
import logging
from colpali_engine.models import ColPali, ColPaliProcessor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


logger = logging.getLogger(__name__)

###vision models
##colpali
def colpali(query, documents):
    """
    Initialize ColPali model, encode query and documents, and rank documents by similarity.

    Args:
        query (str): The query string.
        documents (list): List of document strings (or inputs compatible with ColPali).

    Returns:
        list: Documents ranked by relevance to the query (descending).
    """

    logger.info("Loading ColPali model for visual indexing")

    vision_model = ColPali.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    ).eval()
    vision_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

    # Encode query
    query_inputs = vision_processor([query], return_tensors="pt").to("cuda")
    with torch.no_grad():
        query_features = vision_model.get_text_features(**query_inputs)  # Assuming a get_text_features method

    scores = []
    for doc in documents:
        doc_inputs = vision_processor([doc], return_tensors="pt").to("cuda")
        with torch.no_grad():
            doc_features = vision_model.get_text_features(**doc_inputs)
        # Cosine similarity between query and document features
        score = torch.nn.functional.cosine_similarity(query_features, doc_features).item()
        scores.append(score)

    # Sort documents by descending similarity score
    ranked_docs = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
    return ranked_docs


##clip
def clip(query, documents):
    """
    Initialize CLIP model, encode query and documents, and rank documents by similarity.

    Args:
        query (str): The text query.
        documents (list): List of documents (texts or images compatible with CLIPProcessor).

    Returns:
        list: Documents ranked by relevance to the query (descending order).
    """
    logger.info("Loading clip model for visual indexing")
    
    # Load model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Encode query
    inputs_query = processor(text=[query], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        query_features = model.get_text_features(**inputs_query)

    scores = []
    for doc in documents:
        # Detect if document is text or image, adjust processor input accordingly
        if isinstance(doc, str):
            inputs_doc = processor(text=[doc], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                doc_features = model.get_text_features(**inputs_doc)
        else:
            # Assume image input (PIL.Image or similar)
            inputs_doc = processor(images=doc, return_tensors="pt").to("cuda")
            with torch.no_grad():
                doc_features = model.get_image_features(**inputs_doc)

        # Compute cosine similarity
        score = torch.nn.functional.cosine_similarity(query_features, doc_features).item()
        scores.append(score)

    # Rank documents by descending similarity score
    ranked_docs = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
    return ranked_docs


###textual models
##bm25
def bm25(query, documents):
    """
    Initialize BM25 on the documents, then rank documents documents by BM25 relevance to the query.

    Args:
        query (str): The search query.
        documents (list of str): List of document strings.

    Returns:
        list: Documents ranked by BM25 score (descending).
    """

    logger.info("Loading bm25 model for visual indexing")
    # Tokenize documents
    tokenized_documents = [nltk.word_tokenize(doc.lower()) for doc in documents]
    # Initialize BM25 model
    bm25 = BM25Okapi(tokenized_documents)
    # Tokenize query
    tokenized_query = nltk.word_tokenize(query.lower())
    # Get BM25 scores for all documents
    scores = bm25.get_scores(tokenized_query)
    # Rank documents by score descending
    ranked_docs = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
    return ranked_docs


##dense
def dense(query, documents, model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=None):
    """
    Initialize dense retriever with SentenceTransformer, encode documents,
    then rank documents based on similarity to query.

    Args:
        query (str): The query string.
        documents (list of str): List of document strings.
        model_name (str): SentenceTransformer model name.
        top_k (int, optional): Number of top results to return. Return all if None.

    Returns:
        list of (int, float): List of tuples (doc_index, similarity_score) sorted by score desc.
    """

    logger.info("Loading dense model for visual indexing")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    # Encode documents
    embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)

    # Encode query
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # Compute cosine similarity scores
    scores = cosine_similarity(query_vec, embeddings)[0]

    # Sort by descending similarity
    sorted_indices = np.argsort(scores)[::-1]
    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]

    ranked_results = [(int(idx), float(scores[idx])) for idx in sorted_indices]
    return ranked_results


##hybrid
def hybrid(query, documents, dense_model_name="sentence-transformers/all-MiniLM-L6-v2", bm25_weight=0.5, dense_weight=0.5, top_k=None):
    """
    Initialize BM25 and dense retriever on the documents,
    rank documents by a weighted combination of BM25 and dense similarity scores.

    Args:
        query (str): Query string.
        documents (list of str): List of documents.
        dense_model_name (str): SentenceTransformer model name.
        bm25_weight (float): Weight for BM25 score in hybrid ranking.
        dense_weight (float): Weight for dense similarity score.
        top_k (int, optional): Number of top results to return; return all if None.

    Returns:
        list of (int, float): List of tuples (doc_index, hybrid_score) sorted by descending score.
    """

    logger.info("Loading hybrid model for visual indexing")
    # Tokenize documents for BM25
    tokenized_documents = [nltk.word_tokenize(doc.lower()) for doc in documents]
    bm25 = BM25Okapi(tokenized_documents)

    # Tokenize query for BM25
    tokenized_query = nltk.word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)

    # Initialize dense model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dense_model = SentenceTransformer(dense_model_name, device=device)
    documents_embeddings = dense_model.encode(documents, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    query_embedding = dense_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    dense_scores = cosine_similarity(query_embedding, documents_embeddings)[0]

    # Normalize BM25 scores to [0,1]
    bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    # Dense scores are cosine similarities, roughly in [-1,1], normalize to [0,1]
    dense_scores_norm = (dense_scores + 1) / 2

    # Combine scores
    hybrid_scores = bm25_weight * bm25_scores_norm + dense_weight * dense_scores_norm

    # Sort descending
    sorted_indices = np.argsort(hybrid_scores)[::-1]
    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]

    ranked_results = [(int(idx), float(hybrid_scores[idx])) for idx in sorted_indices]
    return ranked_results


##splade
def splade(query, documents, model_name="naver/splade-cocondenser-ensembledistil"):
    """
    Initialize SPLADE model and tokenizer, encode documents and query,
    and rank documents by similarity to query.

    Args:
        query (str): The query text.
        documents (list of str): List of document texts.
        model_name (str): SPLADE model identifier.

    Returns:
        list of (int, float): Document indices and similarity scores sorted descending.
    """
    logger.info("Loading splade model for visual indexing")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()

    def encode_texts(texts):
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            sparse_rep = torch.log1p(F.relu(logits))
            sparse_rep = torch.max(sparse_rep, dim=1).values  # (batch_size, vocab_size)
        return sparse_rep.cpu()

    # Encode documents and query
    doc_embs = encode_texts(documents)
    query_emb = encode_texts([query])[0]

    # Compute similarity (dot product)
    scores = torch.matmul(doc_embs, query_emb).numpy()

    # Rank descending
    ranked_indices = np.argsort(scores)[::-1]
    ranked_results = [(int(idx), float(scores[idx])) for idx in ranked_indices]

    return ranked_results


##colbert
def colbert_rank_documents(query, documents, model_name="bert-base-uncased", max_length=128, device=None):
    """
    Loads ColBERT-style BERT encoder and tokenizer, encodes documents and query,
    ranks documents by ColBERT max-similarity scoring.

    Args:
        query (str): Query string.
        documents (list of str): List of documents.
        model_name (str): HuggingFace transformer model to use.
        max_length (int): Max tokens for documents/query.
        device (str or None): 'cuda' or 'cpu'. Auto-detect if None.

    Returns:
        list of (int, float): Document indices and similarity scores sorted descending.
    """
    logger.info("Loading colbert model for visual indexing")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    def encode(texts):
        # Tokenize and encode to get token embeddings: (batch_size, seq_len, hidden_dim)
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use last hidden state as token embeddings
            token_embeddings = outputs.last_hidden_state  # (B, L, H)
            # Normalize embeddings for cosine similarity
            token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)
        return token_embeddings

    # Encode query and docs
    query_emb = encode([query])[0]           # (query_len, hidden_dim)
    doc_embs = encode(documents)              # (num_docs, doc_len, hidden_dim)

    scores = []
    # For each document, compute maxsim:
    # For each query token, find max cosine sim with any doc token, then sum over query tokens
    for i in range(len(documents)):
        doc_emb = doc_embs[i]                 # (doc_len, hidden_dim)
        # Cosine similarity matrix: (query_len, doc_len)
        sim_matrix = torch.matmul(query_emb, doc_emb.T)
        # Max over doc tokens per query token
        max_sim_per_query_token, _ = sim_matrix.max(dim=1)
        score = max_sim_per_query_token.sum().item()
        scores.append(score)

    # Rank descending
    ranked_indices = np.argsort(scores)[::-1]
    ranked_results = [(int(idx), float(scores[idx])) for idx in ranked_indices]
    return ranked_results



query = "a beautiful mountain landscape"
documents = [
    "doc content 1",
    "doc content 2",
    "doc content 3"
]


##Visual
#colipali
colipaliranked = colpali(query, documents)
print(colipaliranked)

#clip
clipranked = clip(query, documents)
print(clipranked)


##Textual
#bm25
bm25ranked = bm25(query, documents)
print(bm25ranked)

#dense
denseranked = dense(query, documents, top_k=2)
print(denseranked)

#hybrid
hybridranked = hybrid(query, documents, bm25_weight=0.7, dense_weight=0.3, top_k=3)
print(hybridranked)

#splade
spladeranked = splade(query, documents)
print(spladeranked)

#colbert
colbertranked = colbert_rank_documents(query, documents)
print(colbertranked)