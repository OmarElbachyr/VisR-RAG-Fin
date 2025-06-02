import os
import json
import torch
import pandas as pd
import numpy as np
import time
import logging
import argparse
import re
import uuid
import csv
from tqdm import tqdm
from io import BytesIO
from pdf2image import convert_from_path
import base64
import requests
from PIL import Image
import gc
from difflib import SequenceMatcher
import PyPDF2
import pytesseract
import traceback
import faiss
import transformers
import torch
import torchvision
from tqdm import tqdm
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from rank_bm25 import BM25Okapi
import nltk
from sentence_transformers import util

nltk.download('punkt')


# For embeddings and retrieval
try:
    import chromadb
    import chromadb.utils.embedding_functions as embedding_functions
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from rank_bm25 import BM25Okapi
except ImportError:
    pass


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("visdmrag.log"), logging.StreamHandler()]
)
logger = logging.getLogger("VisDoMRAG")



def setup_pipeline(config):
    # Unpack config
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    llm_model = config["llm_model"]
    vision_retriever = config["vision_retriever"]
    text_retriever = config["text_retriever"]
    top_k = config.get("top_k", 5)
    api_keys = config.get("api_keys", {})
    chunk_size = config.get("chunk_size", 3000)
    chunk_overlap = config.get("chunk_overlap", 300)
    force_reindex = config.get("force_reindex", False)
    qa_prompt = config.get("qa_prompt", "Answer the question objectively based on the context provided.")
    
    # Accept CSV path or fallback
    dataset_csv = config.get("csv_path")
    if not dataset_csv:
        dataset_csv = os.path.join(data_dir, f"{os.path.basename(data_dir)}.csv")
    
    # Create necessary directories
    os.makedirs(f"{output_dir}/{llm_model}_vision", exist_ok=True)
    os.makedirs(f"{output_dir}/{llm_model}_text", exist_ok=True)
    os.makedirs(f"{output_dir}/{llm_model}_visdmrag", exist_ok=True)
    os.makedirs(f"{data_dir}/retrieval", exist_ok=True)
    
    # Initialize LLM
    _initialize_llm(config)
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_csv}")
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"CSV file not found: {dataset_csv}")
    df = pd.read_csv(dataset_csv)
    
    # Initialize document cache and retrieval resources
    document_cache = {}
    _initialize_retrieval_resources(config)
    
    return {
        "config": config,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "llm_model": llm_model,
        "vision_retriever": vision_retriever,
        "text_retriever": text_retriever,
        "top_k": top_k,
        "api_keys": api_keys,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "force_reindex": force_reindex,
        "qa_prompt": qa_prompt,
        "dataset_csv": dataset_csv,
        "df": df,
        "document_cache": document_cache
    }



def _initialize_llm(self):
    """Initialize the LLM based on the selected model."""
    if self.llm_model == "gpt4o":
        if not self.api_keys.get("openai"):
            raise ValueError("OpenAI API key is required")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_keys["openai"])
        logger.info("Initialized GPT-4 (via OpenAI client)")
    
    elif self.llm_model == "qwen":
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            
            self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            min_pixels = 256*28*28
            max_pixels = 640*28*28
            self.qwen_processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct", 
                min_pixels=min_pixels, 
                max_pixels=max_pixels
            )
            self.process_vision_info = process_vision_info
            logger.info("Initialized Qwen2-VL model")
        except ImportError:
            raise ImportError("Required packages for Qwen not found. Install transformers and qwen_vl_utils.")
    else:
        raise ValueError(f"Unsupported LLM model: {self.llm_model}")
    

def _initialize_retrieval_resources(self):
    """Initialize resources needed for retrieval."""
    # Check if we need to compute visual embeddings
    self.vision_retrieval_file = f"{self.data_dir}/retrieval/retrieval_{self.vision_retriever}.csv"
    
    # Check if we need to compute textual embeddings
    self.text_retrieval_file = f"{self.data_dir}/retrieval/retrieval_{self.text_retriever}.csv"
    
    if self.vision_retriever in ["colpali", "colqwen"]:
        if self.vision_retriever == "colpali":
            try:
                from colpali_engine.models import ColPali, ColPaliProcessor
                logger.info("Loading ColPali model for visual indexing")
                self.vision_model = ColPali.from_pretrained(
                    "vidore/colpali-v1.2", 
                    torch_dtype=torch.bfloat16, 
                    device_map="cuda"
                ).eval()
                self.vision_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
            except ImportError:
                raise ImportError("ColPali models not found. Please install colpali_engine.")
        elif self.vision_retriever == "colqwen":
            try:
                from colpali_engine.models import ColQwen2, ColQwen2Processor
                logger.info("Loading ColQwen model for visual indexing")
                self.vision_model = ColQwen2.from_pretrained(
                    "vidore/colqwen2-v0.1", 
                    torch_dtype=torch.bfloat16, 
                    device_map="cuda"
                ).eval()
                self.vision_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
            except ImportError:
                raise ImportError("ColPali/ColQwen models not found. Please install colpali_engine.")
            
    elif self.vision_retriever == "clip":
        try:
            from transformers import CLIPProcessor, CLIPModel
            logger.info("Loading CLIP model for visual indexing")
            self.vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
            self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except ImportError:
            raise ImportError("Transformers not found. Install with `pip install transformers`.")
    else:
        raise ValueError(f"Unsupported visual retriever: {self.vision_retriever}")

    if self.text_retriever == "bm25":
        # Prepare BM25 index
        self.tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in self.corpus]
        self.bm25_model = BM25Okapi(self.tokenized_corpus)
        self.st_embedding_function = None  # Dense not used
    elif self.text_retriever in ["minilm", "mpnet", "bge"]:
        model_map = {
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
            "mpnet": "sentence-transformers/all-mpnet-base-v2",
            "bge": "BAAI/bge-base-en-v1.5"
        }
        self.text_model_name = model_map[self.text_retriever]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.st_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.text_model_name, device=self.device
        )
        # Encode documents
        self.dense_embeddings = self.st_embedding_function(self.corpus)

    elif self.text_retriever == "hybrid":
        # Hybrid: BM25 + Dense
        self.tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in self.corpus]
        self.bm25_model = BM25Okapi(self.tokenized_corpus)
        model_map = {
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
            "mpnet": "sentence-transformers/all-mpnet-base-v2",
            "bge": "BAAI/bge-base-en-v1.5"
        }
        self.text_model_name = model_map[self.hybrid_dense_model]  # e.g., set to "minilm"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.st_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.text_model_name, device=self.device
        )
        self.dense_embeddings = self.st_embedding_function(self.corpus)
            

    #faiss dense retriever
    elif self.text_retriever == "faiss_dense":
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Initializing FAISS-based dense retriever")

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.text_model_name = "sentence-transformers/all-mpnet-base-v2"
            self.dense_model = SentenceTransformer(self.text_model_name, device=self.device)

            self.faiss_index_file = f"{self.data_dir}/retrieval/faiss_index.idx"

            if os.path.exists(self.faiss_index_file):
                logger.info("Loading FAISS index from file")
                self.faiss_index = faiss.read_index(self.faiss_index_file)
            else:
                logger.info("Creating new FAISS index")
                dim = self.dense_model.get_sentence_embedding_dimension()
                self.faiss_index = faiss.IndexFlatL2(dim)

        except ImportError:
            raise ImportError("Required libraries for FAISS not found. Install with `pip install faiss-cpu`.")
        else:
            raise ValueError(f"Unsupported text retriever: {self.text_retriever}")
        
    elif self.text_retriever == "splade":
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            import torch
            import torch.nn.functional as F
            from scipy.sparse import csr_matrix
            logger.info("Loading SPLADE model for sparse retrieval")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.text_model_name = "naver/splade-v3"
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
            self.splade_model = AutoModelForMaskedLM.from_pretrained(self.text_model_name).to(self.device).eval()
            def splade_encode(texts):
                """Sparse encoding for SPLADE."""
                inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    output = self.splade_model(**inputs).logits
                    # LogReLU activation (SPLADE uses log(1 + ReLU(x)))
                    sparse_rep = torch.log1p(F.relu(output))
                    # max over tokens
                    sparse_rep = torch.max(sparse_rep, dim=1).values
                return sparse_rep.cpu()
            self.splade_encode = splade_encode  # Save function for later use
        except ImportError:
            raise ImportError("Transformers and torch required. Install with `pip install transformers torch`.")


##faiss
def index_documents_with_faiss(self, documents):
    """
    Encodes and adds documents to the FAISS index.
    Saves the index to disk.
    """
    logger = logging.getLogger(__name__)
    logger.info("Encoding documents for FAISS indexing")
    embeddings = self.dense_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
    self.faiss_index.add(np.array(embeddings, dtype='float32'))
    logger.info(f"FAISS index now contains {self.faiss_index.ntotal} vectors")
    faiss.write_index(self.faiss_index, self.faiss_index_file)
    logger.info(f"FAISS index written to {self.faiss_index_file}")

def query_faiss(self, query, top_k=5):
    """
    Encodes a query and searches the FAISS index.
    Returns top_k indices and distances.
    """
    import numpy as np
    logger = logging.getLogger(__name__)

    logger.info("Encoding query for FAISS retrieval")
    query_vec = self.dense_model.encode([query], convert_to_numpy=True)
    distances, indices = self.faiss_index.search(np.array(query_vec, dtype='float32'), top_k)
    return indices[0], distances[0]


##clip
def encode_image_with_clip(self, image):
    """
    Encodes a PIL image using CLIP model.
    """
    logger = logging.getLogger(__name__)
    inputs = self.vision_processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        image_features = self.vision_model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.cpu().numpy()
            
def encode_text_with_clip(self, text: str):
    """
    Encodes text using CLIP model.
    """
    inputs = self.vision_processor(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        text_features = self.vision_model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu().numpy()


def retrieve(self, query, modality="textual", top_k=5, alpha=0.5):
    if modality == "textual":
        corpus = self.text_chunks  # You must have this set somewhere earlier
        meta = self.text_metadata  # Should match chunk structure
    elif modality == "visual":
        corpus = self.visual_chunks
        meta = self.visual_metadata
    else:
        raise ValueError("Invalid modality type")

    if self.text_retriever == "bm25":
        tokenized_query = nltk.word_tokenize(query.lower())
        scores = self.bm25_model.get_scores(tokenized_query)
    elif self.text_retriever in ["minilm", "mpnet", "bge"]:
        query_emb = self.st_embedding_function([query])[0]
        scores = util.cos_sim(
            torch.tensor(query_emb).to(self.device),
            torch.tensor(self.dense_embeddings).to(self.device)
        )[0].cpu().numpy()
    elif self.text_retriever == "hybrid":
        tokenized_query = nltk.word_tokenize(query.lower())
        bm25_scores = self.bm25_model.get_scores(tokenized_query)

        query_emb = self.st_embedding_function([query])[0]
        dense_scores = util.cos_sim(
            torch.tensor(query_emb).to(self.device),
            torch.tensor(self.dense_embeddings).to(self.device)
        )[0].cpu().numpy()

        scores = alpha * np.array(bm25_scores) + (1 - alpha) * np.array(dense_scores)
    else:
        raise ValueError("Unknown retriever type")

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [meta[i] for i in top_indices]


def process_query(self, query_id):
    """
    Process a single query through the complete VisDoMRAG pipeline.
    
    Args:
        query_id (str): The query ID
        
    Returns:
        bool: Success status
    """
    try:
        # Get query information
        query_row = self.df[self.df['q_id'] == query_id].iloc[0]
        question = query_row['question']
        
        try:
            # Try to parse the answer field as a list/dict if it's in that format
            answer = eval(query_row['answer'])
        except:
            # If parsing fails, use as-is
            answer = query_row['answer']
        
        # Define file paths for outputs
        visual_file = f"{self.output_dir}/{self.llm_model}_vision/response_{str(query_id).replace('/','$')}.json"
        textual_file = f"{self.output_dir}/{self.llm_model}_text/response_{str(query_id).replace('/','$')}.json"
        combined_file = f"{self.output_dir}/{self.llm_model}_visdmrag/response_{str(query_id).replace('/','$')}.json"
        
        # Skip if the combined file already exists
        if os.path.exists(combined_file):
            logger.info(f"Combined file already exists for query {query_id}")
            return True
        
        # Process visual contexts if needed
        visual_response_dict = None
        if not os.path.exists(visual_file):
            logger.info(f"Generating visual response for query {query_id}")
            visual_contexts = self.retrieve(question, modality="visual")
            
            if visual_contexts:
                visual_response = self.generate_visual_response(question, visual_contexts)
                visual_response_dict = self.extract_sections(visual_response)
                
                # Add metadata
                visual_response_dict.update({
                    "question": question,
                    "document": [ctx['document_id'] for ctx in visual_contexts],
                    "gt_answer": answer,
                    "pages": [ctx['page_number'] for ctx in visual_contexts]
                })
                
                # Save visual response
                with open(visual_file, 'w') as file:
                    json.dump(visual_response_dict, file, indent=4)
                
                # Memory cleanup
                del visual_contexts
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Load existing visual response
            with open(visual_file, 'r') as file:
                visual_response_dict = json.load(file)
        
        # Process textual contexts if needed
        textual_contexts = self.retrieve(question, modality="textual")
        if not os.path.exists(textual_file):
            logger.info(f"Generating textual response for query {query_id}")
            textual_contexts = self.retrieve_textual_contexts(query_id)
            
            if textual_contexts:
                textual_response = self.generate_textual_response(question, textual_contexts)
                textual_response_dict = self.extract_sections(textual_response)
                
                # Add metadata
                textual_response_dict.update({
                    "question": question,
                    "document": [ctx['chunk_pdf_name'] for ctx in textual_contexts],
                    "gt_answer": answer,
                    "pages": [ctx['pdf_page_number'] for ctx in textual_contexts],
                    "chunks": "\n".join([ctx['chunk'] for ctx in textual_contexts])
                })
                
                # Save textual response
                with open(textual_file, 'w') as file:
                    json.dump(textual_response_dict, file, indent=4)
                
                # Memory cleanup
                del textual_contexts
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Load existing textual response
            with open(textual_file, 'r') as file:
                textual_response_dict = json.load(file)
        
        # Skip if either response is missing
        if not visual_response_dict or not textual_response_dict:
            logger.warning(f"Missing responses for query {query_id}")
            return False
        
        # Combine responses
        logger.info(f"Combining responses for query {query_id}")
        combined_sections = self.combine_responses(
            question, 
            visual_response_dict, 
            textual_response_dict,
            answer
        )
        
        # Create combined response
        combined_response = {
            "question": question,
            "answer": combined_sections.get("Final Answer", ""),
            "gt_answer": answer,
            "analysis": combined_sections.get("Analysis", ""),
            "conclusion": combined_sections.get("Conclusion", ""),
            "response1": visual_response_dict,
            "response2": textual_response_dict
        }
        
        # Save combined response
        with open(combined_file, 'w') as file:
            json.dump(combined_response, file, indent=4)
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
            
    except Exception as e:
        logger.error(f"Error processing query {query_id}: {str(e)}")
        return False






config = {
    "data_dir": "/path/to/data_directory",         # Directory containing your dataset CSV and retrieval files
    "output_dir": "/path/to/output_directory",     # Where outputs, models, results get saved
    "llm_model": "my_llm_model_name_or_path",      # Name or path to your language model
    "vision_retriever": "vision_retriever_config", # Config or identifier for vision retrieval (could be a model name, API key, etc.)
    "text_retriever": "text_retriever_config",     # Config or identifier for text retrieval

    # Optional parameters with defaults
    "top_k": 5,                                    # Number of top results to retrieve, default is 5
    "api_keys": {                                  # Optional API keys for services used by LLM or retrievers
        "openai": "your_openai_api_key",
        "other_service": "other_api_key"
    },
    "chunk_size": 3000,                            # Optional, chunk size for splitting documents, default 3000
    "chunk_overlap": 300,                          # Optional, chunk overlap, default 300
    "force_reindex": False,                        # Optional, force reindexing flag
    "qa_prompt": "Answer the question objectively based on the context provided.", # Optional, prompt template

    # Optional: path to CSV dataset directly, if not provided it uses data_dir + basename
    "csv_path": "/path/to/your_dataset.csv"
}


result = setup_pipeline(config)
df = result["df"]
print(df.head())

llm = result["llm_model_instance"]
vision_index = result["vision_index"]
text_index = result["text_index"]

# Use retrievers to find relevant documents or images
vision_results = vision_index.retrieve("query about images", top_k=result["top_k"])
text_results = text_index.retrieve("query about text", top_k=result["top_k"])

# Then use the LLM to answer questions based on the retrieved data
answer = llm.answer(question="What is this image about?", context=vision_results)
print(answer)