```mermaid
flowchart LR
  A[Document\n(PDF/DOCX/PPT)] --> B[Parse / Extract\n(text, images, tables)]
  B --> C[Chunk / Normalize\n(text chunks + asset records)]
  C --> D[Summarize assets (optional)\n(images/tables)]
  C --> E[Embed text chunks]
  D --> F[Embed summaries]
  E --> G[(Vector Index\nFAISS)]
  F --> G
  G --> H[Retrieve Top-K]
  H --> I[LLM Answer + citations]
  I --> J{Post-filter assets?}
  J -->|yes| K[Filter by cited pages/window]
  K --> L[Final response\n(text+tables+images+cites)]
  J -->|no| L
