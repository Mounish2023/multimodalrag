# Multimodal RAG Retrieval Approaches

This repository explores and demonstrates different strategies for retrieving both text and images in a Multimodal Retrieval-Augmented Generation (RAG) system. Below are the five methods implemented, ranging from caption-based indexing to advanced citation-filtered retrieval.

## Approaches

### Method 1 & 2: Text-Based Indexing of Images
These methods rely on converting visual content into text representations, allowing standard text embedding models to handle the retrieval.

*   **Method 1: BLIP Captions**
    *   **Process**: Images are converted to variable-length captions using the BLIP model.
    *   **Embedding**: These captions are then converted to vector embeddings using standard text embedding models.
    *   **Notebook**: `method1_blip_openai_text_embeddings_faiss.ipynb`

*   **Method 2: GPT-4o Descriptions**
    *   **Process**: Images are analyzed by GPT-4o to generate detailed, comprehensive textual descriptions.
    *   **Embedding**: These detailed descriptions are converted to embeddings. This generally captures more nuance than simple captions.
    *   **Notebook**: `method2_gpt4o_image_desc_openai_text_embeddings_faiss.ipynb`

### Method 3 & 4: Multimodal Vector Space
These methods utilize models trained to embed both text and images into the same vector space, enabling direct cross-modal retrieval without explicit text conversion.

*   **Method 3: CLIP Embeddings**
    *   **Process**: Uses OpenAI's CLIP model to generate embeddings for both text chunks and images.
    *   **Retrieval**: A text query can directly retrieve semantically similar images based on vector distance in the shared space.
    *   **Notebook**: `method3_clip_multimodal_embeddings_faiss.ipynb`

*   **Method 4: Cohere Multimodal Embeddings**
    *   **Process**: Utilizes Cohere's multimodal embedding models (e.g., `embed-v4.0`).
    *   **Retrieval**: Images and text are indexed together, allowing for robust retrieval across languages and modalities.
    *   **Notebook**: `method4_cohere_multimodal_embeddings_faiss_answer.ipynb`

### Method 5: Citation-Filtered Image Retrieval
This serves as a hybrid, precision-focused approach that leverages the reasoning capabilities of the LLM to narrow down the search space.

*   **Process**:
    1.  **Ingestion**: Images are converted to text descriptions using GPT-4o. Both the original document text and these image descriptions are converted to Cohere embeddings and indexed.
    2.  **Generation**: The system generates a text answer to the user's query based on retrieved text context.
    3.  **Filtration**: The system extracts citations and page numbers directly from the generated answer.
    4.  **Retrieval**: The search space for images is strictly filtered to the pages cited in the answer.
    5.  **Selection**: Final image selection is performed using cosine similarity within that filtered subset.
*   **Notebook**: `method5_cohere_text_embeddings_citation_filtered_images.ipynb`
