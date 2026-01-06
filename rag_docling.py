import os
import json
import base64
from pathlib import Path
from tempfile import mkdtemp
from typing import List, Dict, Any
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import requests
from PIL import Image


# Load environment variables
load_dotenv()

# Configuration from environment
PDF_PATH = os.getenv("PDF_PATH", "./paper.pdf")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "mahonzhan/all-MiniLM-L6-v2:latest")
CHUNKER_TOKENIZER = os.getenv("CHUNKER_TOKENIZER", "sentence-transformers/all-MiniLM-L6-v2")  # HuggingFace tokenizer for chunking
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen3-vl:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TOP_K = int(os.getenv("TOP_K", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
QUESTION = "what are the main authors of this paper and what is the main idea of the paper?"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
IMAGE_DIR = os.getenv("IMAGE_DIR", "./outputs/images")

def convert_pdf_to_chunks(pdf_path: str, tokenizer: str, extract_images: bool = True) -> tuple[List[Dict[str, Any]], Any, List[Dict[str, Any]]]:
    """Convert PDF to chunks using Docling and extract images."""
    print(f"Processing PDF: {pdf_path}")
    
    # Initialize the document converter with pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.images_scale = 2.0  # Higher resolution for better VLM processing
    pipeline_options.generate_picture_images = extract_images
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # Convert the document
    result = converter.convert(pdf_path)
    
    # Initialize chunker
    chunker = HybridChunker(tokenizer=tokenizer)
    
    # Get chunks from the document
    chunks = []
    chunk_iter = chunker.chunk(result.document)
    for chunk in chunk_iter:
        # Extract page number safely
        page_no = 0
        try:
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                doc_item = chunk.meta.doc_items[0]
                if hasattr(doc_item, "prov") and doc_item.prov:
                    prov = doc_item.prov[0]
                    page_no = getattr(prov, "page_no", 0)
        except (IndexError, AttributeError):
            pass
        
        chunks.append({
            "text": chunk.text,
            "metadata": {
                "source": pdf_path,
                "page": page_no,
            }
        })
    
    print(f"Created {len(chunks)} chunks")
    
    # Extract images from the document
    images_info = []
    if extract_images and hasattr(result.document, 'pictures'):
        print(f"Extracting images from document...")
        for idx, picture in enumerate(result.document.pictures):
            try:
                # Get image data
                if hasattr(picture, 'image') and picture.image:
                    image_data = picture.image.pil_image
                    
                    # Get page number
                    page_no = 0
                    if hasattr(picture, 'prov') and picture.prov:
                        page_no = getattr(picture.prov[0], 'page_no', 0) if picture.prov else 0
                    
                    images_info.append({
                        'index': idx,
                        'image': image_data,
                        'page': page_no,
                        'caption': getattr(picture, 'text', ''),
                    })
            except Exception as e:
                print(f"Error extracting image {idx}: {e}")
        
        print(f"Extracted {len(images_info)} images")
    
    return chunks, result.document, images_info


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def describe_image_with_vlm(image: Image.Image, model: str = VLM_MODEL, prompt: str = "Describe this image in detail.") -> str:
    """Describe an image using Ollama VLM."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    # Convert image to base64
    image_base64 = image_to_base64(image)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Error describing image with VLM: {e}")
        return f"Error: {str(e)}"


def save_images_and_descriptions(images_info: List[Dict[str, Any]], output_dir: str, vlm_model: str = VLM_MODEL) -> List[Dict[str, Any]]:
    """Save images and generate descriptions using VLM."""
    if not images_info:
        return []
    
    # Create image directory
    image_dir = Path(output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {len(images_info)} images with VLM...")
    
    image_chunks = []
    for img_info in images_info:
        idx = img_info['index']
        image = img_info['image']
        page = img_info['page']
        caption = img_info['caption']
        
        # Save image
        image_path = image_dir / f"image_{idx}_page_{page}.png"
        image.save(image_path)
        print(f"Saved image {idx} to {image_path}")
        
        # Generate description with VLM
        print(f"Generating description for image {idx} using {vlm_model}...")
        description = describe_image_with_vlm(image, vlm_model)
        print(f"Description for image {idx}: {clip_text(description, 100)}")
        
        # Create text chunk with image description
        text_content = f"[IMAGE {idx} - Page {page}]\n"
        if caption:
            text_content += f"Caption: {caption}\n"
        text_content += f"Description: {description}"
        
        image_chunks.append({
            'text': text_content,
            'metadata': {
                'source': str(image_path),
                'page': page,
                'type': 'image',
                'image_index': idx
            }
        })
    
    print(f"\nGenerated {len(image_chunks)} image description chunks")
    return image_chunks


def get_ollama_embedding(text: str, model: str) -> List[float]:
    """Get embedding from Ollama."""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": model, "prompt": text}
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("embedding", [])
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise


def create_embeddings(chunks: List[Dict[str, Any]], model_name: str) -> List[List[float]]:
    """Create embeddings for chunks using Ollama."""
    print(f"Creating embeddings with Ollama model: {model_name}")
    
    texts = [chunk["text"] for chunk in chunks]
    embeddings = []
    
    for i, text in enumerate(texts):
        if (i + 1) % 10 == 0:
            print(f"Processing chunk {i + 1}/{len(texts)}")
        embedding = get_ollama_embedding(text, model_name)
        embeddings.append(embedding)
    
    print(f"Created {len(embeddings)} embeddings")
    return embeddings


def setup_milvus_collection(collection_name: str = "docling_rag", dim: int = 384):
    """Set up Milvus collection for vector storage."""
    # Create temporary database path
    milvus_uri = str(Path(mkdtemp()) / "docling.db")
    
    # Connect to Milvus
    connections.connect("default", uri=milvus_uri)
    
    # Drop collection if exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]
    
    schema = CollectionSchema(fields=fields, description="Docling RAG collection")
    collection = Collection(name=collection_name, schema=schema)
    
    # Create index for vector field - using IP (cosine similarity)
    index_params = {
        "index_type": "FLAT",
        "metric_type": "IP",
        "params": {}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    return collection


def insert_into_milvus(collection: Collection, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
    """Insert chunks and embeddings into Milvus."""
    print(f"Inserting {len(chunks)} chunks into Milvus")
    
    # Normalize embeddings for cosine similarity
    normalized_embeddings = [normalize_embedding(emb) for emb in embeddings]
    
    data = [
        normalized_embeddings,
        [chunk["text"] for chunk in chunks],
        [chunk["metadata"] for chunk in chunks],
    ]
    
    collection.insert(data)
    collection.load()
    
    print("Data inserted and collection loaded")


def normalize_embedding(embedding: List[float]) -> List[float]:
    """Normalize embedding for cosine similarity."""
    import math
    norm = math.sqrt(sum(x * x for x in embedding))
    return [x / norm for x in embedding] if norm > 0 else embedding


def retrieve_context(collection: Collection, query: str, embed_model_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant context from Milvus."""
    print(f"Retrieving top {top_k} contexts for query")
    
    # Create query embedding using Ollama
    query_embedding = get_ollama_embedding(query, embed_model_id)
    query_embedding = normalize_embedding(query_embedding)
    
    # Search in Milvus using IP (cosine similarity)
    search_params = {"metric_type": "IP", "params": {}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text", "metadata"]
    )
    
    # Format results
    contexts = []
    for hits in results:
        for hit in hits:
            contexts.append({
                "text": hit.entity.get("text"),
                "metadata": hit.entity.get("metadata"),
                "score": hit.distance
            })
    
    return contexts


def query_ollama(prompt: str, model: str = OLLAMA_MODEL, temperature: float = TEMPERATURE) -> str:
    """Query Ollama API directly."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return f"Error: {str(e)}"


def create_rag_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    """Create RAG prompt with context."""
    context_text = "\n\n".join([
        f"[Context {i+1}]\n{ctx['text']}"
        for i, ctx in enumerate(contexts)
    ])
    
    prompt = f"""Context information is below.
---------------------
{context_text}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:"""
    
    return prompt


def clip_text(text: str, threshold: int = 100) -> str:
    """Clip text to threshold length."""
    return f"{text[:threshold]}..." if len(text) > threshold else text


def save_docling_output(document: Any, output_path: str):
    """Save Docling document to markdown."""
    print(f"Saving Docling output to: {output_path}")
    
    # Export to markdown
    markdown_content = document.export_to_markdown()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Docling output saved to {output_path}")


def save_rag_results(question: str, answer: str, contexts: List[Dict[str, Any]], output_path: str):
    """Save RAG results to markdown."""
    print(f"Saving RAG results to: {output_path}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""# RAG Results

**Generated**: {timestamp}

---

## Question

{question}

---

## Answer

{answer}

---

## Sources

"""
    
    for i, ctx in enumerate(contexts, 1):
        content += f"""### Source {i}

**Similarity Score**: {ctx['score']:.4f}

**Metadata**: {json.dumps(ctx['metadata'], indent=2)}

**Text**:

```
{ctx['text']}
```

---

"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"RAG results saved to {output_path}")


def main():
    """Main RAG pipeline."""
    print("=" * 80)
    print("RAG Pipeline with Docling + Ollama + VLM")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Create image directory
    image_dir = Path(IMAGE_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert PDF to chunks and extract images
    chunks, document, images_info = convert_pdf_to_chunks(PDF_PATH, CHUNKER_TOKENIZER)
    
    # Save Docling output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    docling_output_path = output_dir / f"docling_output_{timestamp}.md"
    save_docling_output(document, str(docling_output_path))
    
    print(f"\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}: {clip_text(chunk['text'], 100)}")
    print("...\n")
    
    # Step 1.5: Process images with VLM and add to chunks
    if images_info:
        image_chunks = save_images_and_descriptions(images_info, str(image_dir), VLM_MODEL)
        print(f"\nAdding {len(image_chunks)} image description chunks to the corpus")
        chunks.extend(image_chunks)
        print(f"Total chunks (text + images): {len(chunks)}")
    
    # Step 2: Create embeddings
    embeddings = create_embeddings(chunks, EMBED_MODEL_ID)
    
    # Step 3: Setup Milvus and insert data
    embedding_dim = len(embeddings[0]) if embeddings else 384
    collection = setup_milvus_collection(dim=embedding_dim)
    insert_into_milvus(collection, chunks, embeddings)
    
    # Step 4: Retrieve relevant context
    contexts = retrieve_context(collection, QUESTION, EMBED_MODEL_ID, top_k=TOP_K)
    
    # Step 5: Generate answer using Ollama
    print(f"\nQuestion: {QUESTION}\n")
    prompt = create_rag_prompt(QUESTION, contexts)
    
    print("Querying Ollama...")
    answer = query_ollama(prompt)
    
    # Save RAG results
    rag_output_path = output_dir / f"rag_results_{timestamp}.md"
    save_rag_results(QUESTION, answer, contexts, str(rag_output_path))
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nQuestion:\n{QUESTION}\n")
    print(f"Answer:\n{clip_text(answer, 500)}\n")
    
    print("\nSources:")
    for i, ctx in enumerate(contexts):
        print(f"\nSource {i + 1}:")
        print(f"  text: {json.dumps(clip_text(ctx['text'], 350))}")
        print(f"  metadata: {ctx['metadata']}")
        print(f"  similarity_score: {ctx['score']:.4f}")
    
    print("\n" + "=" * 80)
    print(f"\n✓ Docling output saved to: {docling_output_path}")
    print(f"✓ RAG results saved to: {rag_output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
