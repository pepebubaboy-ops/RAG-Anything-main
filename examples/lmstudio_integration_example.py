"""
OpenAI-Compatible Local Endpoint Integration Example with RAG-Anything

This example demonstrates how to integrate a local OpenAI-compatible endpoint
(e.g., LM Studio or Ollama) with RAG-Anything for document processing and querying.
"""

import os
import uuid
import asyncio
import base64
import argparse
from typing import List, Dict, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import AsyncOpenAI
import numpy as np

# Load environment variables
load_dotenv()

# RAG-Anything imports
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache

LM_BASE_URL = os.getenv("LLM_BINDING_HOST", "http://localhost:1234/v1")
LM_API_KEY = os.getenv("LLM_BINDING_API_KEY", "lm-studio")
LM_MODEL_FALLBACK = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LM_MODEL_GRAPH_EXTRACT = os.getenv(
    "LLM_MODEL_GRAPH_EXTRACT", os.getenv("LLM_MODEL_GRAPH", LM_MODEL_FALLBACK)
)
LM_MODEL_GRAPH_SUMMARY = os.getenv("LLM_MODEL_GRAPH_SUMMARY", LM_MODEL_GRAPH_EXTRACT)
LM_MODEL_QUERY = os.getenv("LLM_MODEL_QUERY", LM_MODEL_GRAPH_EXTRACT)
LM_MODEL_NAME = LM_MODEL_QUERY
LLM_ROUTER_DEBUG = os.getenv("LLM_ROUTER_DEBUG", "false").lower() == "true"
LM_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")
LM_BINDING = os.getenv("LLM_BINDING", "openai").lower()
EMBED_BASE_URL = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434/v1")
EMBED_API_KEY = os.getenv("EMBEDDING_BINDING_API_KEY", "ollama")
EMBED_BINDING = os.getenv("EMBEDDING_BINDING", "ollama").lower()
RAG_PARSE_METHOD = os.getenv("RAG_PARSE_METHOD", "txt")

GRAPH_EXTRACT_MARKERS = (
    "extract entities and relationships from the input text",
    "knowledge graph specialist responsible for extracting entities and relationships",
    "<entity_types>",
)
GRAPH_SUMMARY_MARKERS = (
    "synthesize a list of descriptions",
    "description list:",
    "merged description",
)
QUERY_ANSWER_MARKERS = (
    "reference document list",
    "### references",
    "knowledge graph data",
    "document chunks",
)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _classify_llm_task(prompt: str, system_prompt: Optional[str]) -> str:
    prompt_text = (prompt or "").lower()
    system_text = (system_prompt or "").lower()
    full_text = f"{system_text}\n{prompt_text}"

    if _contains_any(full_text, GRAPH_EXTRACT_MARKERS):
        return "graph_extract"
    if _contains_any(full_text, GRAPH_SUMMARY_MARKERS):
        return "graph_summary"
    if _contains_any(full_text, QUERY_ANSWER_MARKERS):
        return "query_answer"
    return "default"


def _select_model_for_task(task: str) -> str:
    if task == "graph_extract":
        return LM_MODEL_GRAPH_EXTRACT
    if task == "graph_summary":
        return LM_MODEL_GRAPH_SUMMARY
    if task == "query_answer":
        return LM_MODEL_QUERY
    return LM_MODEL_FALLBACK


def _to_ollama_host(base_url: str) -> str:
    """Convert OpenAI-compatible base URL to native Ollama host URL.

    Example:
    - http://localhost:11434/v1 -> http://localhost:11434
    """
    parsed = urlparse(base_url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    # Fallback for unexpected values
    return base_url.replace("/v1", "").rstrip("/")


async def lmstudio_llm_model_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    **kwargs,
) -> str:
    """Top-level LLM function for LightRAG (pickle-safe)."""
    task = _classify_llm_task(prompt, system_prompt)
    selected_model = _select_model_for_task(task)
    if LLM_ROUTER_DEBUG:
        print(
            f"🧭 Router task='{task}' model='{selected_model}' "
            f"(fallback='{LM_MODEL_FALLBACK}')"
        )

    return await openai_complete_if_cache(
        model=selected_model,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=LM_BASE_URL,
        api_key=LM_API_KEY,
        **kwargs,
    )


async def lmstudio_embedding_async(texts: List[str]) -> np.ndarray:
    """Top-level embedding function for LightRAG (pickle-safe).

    NOTE:
    We intentionally use the OpenAI-compatible embeddings endpoint directly
    instead of lightrag.llm.openai.openai_embed, because openai_embed is
    decorated with a fixed default embedding_dim=1536. That causes dimension
    mismatch errors with local models like nomic-embed-text (768 dims).
    """
    # For Ollama, prefer the native endpoint implementation from LightRAG.
    # It handles long inputs better (automatic truncation behavior).
    if EMBED_BINDING == "ollama":
        from lightrag.llm.ollama import ollama_embed

        embeddings = await ollama_embed.func(
            texts=texts,
            embed_model=LM_EMBED_MODEL,
            host=_to_ollama_host(EMBED_BASE_URL),
            api_key=EMBED_API_KEY,
        )
        return np.array(embeddings, dtype=np.float32)

    # Fallback for other OpenAI-compatible endpoints (LM Studio, etc.)
    # IMPORTANT: for OpenAI-compatible servers, this must typically end with "/v1".
    client = AsyncOpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY)
    try:
        response = await client.embeddings.create(model=LM_EMBED_MODEL, input=texts)
        vectors: List[List[float]] = []
        for item in response.data:
            emb = item.embedding
            if isinstance(emb, list):
                vectors.append([float(v) for v in emb])
            elif isinstance(emb, str):
                decoded = np.frombuffer(
                    base64.b64decode(emb), dtype=np.float32
                ).tolist()
                vectors.append(decoded)
            else:
                raise TypeError(f"Unsupported embedding payload type: {type(emb)}")
        return np.array(vectors, dtype=np.float32)
    finally:
        await client.close()


async def detect_embedding_dimension() -> int:
    """Probe current embedding endpoint/model and return actual vector dim."""
    probe_vectors = await lmstudio_embedding_async(["dimension probe"])
    if probe_vectors.ndim != 2 or probe_vectors.shape[0] < 1:
        raise RuntimeError(
            f"Unexpected embedding output shape: {getattr(probe_vectors, 'shape', None)}"
        )
    return int(probe_vectors.shape[1])


class LMStudioRAGIntegration:
    """Integration class for a local OpenAI-compatible endpoint with RAG-Anything."""

    def __init__(self):
        # LM Studio configuration using standard LLM_BINDING variables
        self.base_url = os.getenv("LLM_BINDING_HOST", "http://localhost:1234/v1")
        self.api_key = os.getenv("LLM_BINDING_API_KEY", "lm-studio")
        self.model_name = LM_MODEL_QUERY
        self.model_graph_extract = LM_MODEL_GRAPH_EXTRACT
        self.model_graph_summary = LM_MODEL_GRAPH_SUMMARY
        self.model_fallback = LM_MODEL_FALLBACK
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"
        )
        self.configured_embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))
        self.embedding_dim = self.configured_embedding_dim
        self.parse_method = RAG_PARSE_METHOD

        # RAG-Anything configuration
        # Use a fresh working directory each run to avoid legacy doc_status schema conflicts
        self.config = RAGAnythingConfig(
            working_dir=f"./rag_storage_lmstudio/{uuid.uuid4()}",
            parser="mineru",
            parse_method=self.parse_method,
            enable_image_processing=False,
            enable_table_processing=False,
            enable_equation_processing=False,
        )
        print(f"📁 Using working_dir: {self.config.working_dir}")
        print("🧭 LLM router configuration:")
        print(f"  - graph_extract: {self.model_graph_extract}")
        print(f"  - graph_summary: {self.model_graph_summary}")
        print(f"  - query_answer : {self.model_name}")
        print(f"  - default      : {self.model_fallback}")

        self.rag = None

    async def test_connection(self) -> bool:
        """Test local endpoint connection."""
        try:
            print(f"🔌 Testing endpoint connection at: {self.base_url}")
            client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
            models = await client.models.list()
            print(f"✅ Connected successfully! Found {len(models.data)} models")

            # Show available models
            print("📊 Available models:")
            for i, model in enumerate(models.data[:5]):
                marker = "🎯" if model.id == self.model_name else "  "
                print(f"{marker} {i+1}. {model.id}")

            if len(models.data) > 5:
                print(f"  ... and {len(models.data) - 5} more models")

            return True
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            print("\n💡 Troubleshooting tips:")
            print("1. Ensure your local server is running (Ollama or LM Studio)")
            print("2. Verify endpoint URL and port")
            print("3. Ensure the selected model is available and loaded")
            print(f"4. Current endpoint: {self.base_url}")
            return False
        finally:
            try:
                await client.close()
            except Exception:
                pass

    async def test_chat_completion(self) -> bool:
        """Test basic chat functionality."""
        try:
            print(f"💬 Testing chat with model: {self.model_name}")
            client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {
                        "role": "user",
                        "content": "Hello! Please confirm you're working and tell me your capabilities.",
                    },
                ],
                max_tokens=100,
                temperature=0.7,
            )

            result = response.choices[0].message.content.strip()
            print("✅ Chat test successful!")
            print(f"Response: {result}")
            return True
        except Exception as e:
            print(f"❌ Chat test failed: {str(e)}")
            return False
        finally:
            try:
                await client.close()
            except Exception:
                pass

    # Deprecated factory helpers removed to reduce redundancy

    def embedding_func_factory(self, embedding_dim: int):
        """Create a completely serializable embedding function."""
        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,  # nomic-embed-text-v1.5 context length
            func=lmstudio_embedding_async,
        )

    async def initialize_rag(self):
        """Initialize RAG-Anything with local endpoint functions."""
        print("Initializing RAG-Anything with local endpoint...")

        try:
            detected_dim = await detect_embedding_dimension()
            if detected_dim != self.configured_embedding_dim:
                print(
                    f"⚠️ EMBEDDING_DIM mismatch: .env={self.configured_embedding_dim}, "
                    f"endpoint/model returned={detected_dim}. Using detected value."
                )
            self.embedding_dim = detected_dim
            print(
                f"🧮 Embedding model '{self.embedding_model}' dimension: {self.embedding_dim}"
            )

            self.rag = RAGAnything(
                config=self.config,
                llm_model_func=lmstudio_llm_model_func,
                embedding_func=self.embedding_func_factory(self.embedding_dim),
            )

            # Compatibility: avoid writing unknown field 'multimodal_processed' to LightRAG doc_status
            # Older LightRAG versions may not accept this extra field in DocProcessingStatus
            async def _noop_mark_multimodal(doc_id: str):
                return None

            self.rag._mark_multimodal_processing_complete = _noop_mark_multimodal

            print("✅ RAG-Anything initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ RAG initialization failed: {str(e)}")
            return False

    async def process_document_example(self, file_path: str, parse_method: str = None):
        """Example: Process a document with LM Studio backend."""
        if not self.rag:
            print("❌ RAG not initialized. Call initialize_rag() first.")
            return

        try:
            print(f"📄 Processing document: {file_path}")
            effective_parse_method = parse_method or self.parse_method
            print(f"🧩 Parser: {self.config.parser}, parse_method: {effective_parse_method}")
            await self.rag.process_document_complete(
                file_path=file_path,
                output_dir="./output_lmstudio",
                parse_method=effective_parse_method,
                display_stats=True,
            )
            print("✅ Document processing completed!")
        except Exception as e:
            print(f"❌ Document processing failed: {str(e)}")

    async def query_examples(self):
        """Example queries with different modes."""
        if not self.rag:
            print("❌ RAG not initialized. Call initialize_rag() first.")
            return

        # Example queries
        queries = [
            ("What are the main topics in the processed documents?", "hybrid"),
            ("Summarize any tables or data found in the documents", "local"),
            ("What images or figures are mentioned?", "global"),
        ]

        print("\n🔍 Running example queries...")
        for query, mode in queries:
            try:
                print(f"\nQuery ({mode}): {query}")
                result = await self.rag.aquery(query, mode=mode)
                print(f"Answer: {result[:200]}...")
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")

    async def interactive_query_loop(self, mode: str = "hybrid"):
        """Interactive query loop for asking questions about processed content."""
        if not self.rag:
            print("❌ RAG not initialized. Call initialize_rag() first.")
            return

        valid_modes = {"local", "global", "hybrid", "naive", "mix", "bypass"}
        current_mode = mode if mode in valid_modes else "hybrid"

        print("\n🧠 Interactive Q&A mode is ready.")
        print("Type your question and press Enter.")
        print("Commands: 'exit' to quit, '/mode <name>' to switch query mode.")
        print(f"Current mode: {current_mode}")

        while True:
            try:
                user_query = (await asyncio.to_thread(input, "\nYou> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Exiting interactive mode.")
                break

            if not user_query:
                continue

            lowered = user_query.lower()
            if lowered in {"exit", "quit", "q", ":q"}:
                print("👋 Exiting interactive mode.")
                break

            if lowered.startswith("/mode "):
                requested_mode = lowered.split(maxsplit=1)[1].strip()
                if requested_mode in valid_modes:
                    current_mode = requested_mode
                    print(f"✅ Query mode switched to: {current_mode}")
                else:
                    print(
                        "❌ Invalid mode. Use one of: "
                        + ", ".join(sorted(valid_modes))
                    )
                continue

            try:
                answer = await self.rag.aquery(user_query, mode=current_mode)
                print(f"\nAssistant ({current_mode})> {answer}")
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")

    async def simple_query_example(self):
        """Example basic text query with sample content."""
        if not self.rag:
            print("❌ RAG not initialized")
            return

        try:
            print("\nAdding sample content for testing...")

            # Create content list in the format expected by RAGAnything
            content_list = [
                {
                    "type": "text",
                    "text": """LM Studio Integration with RAG-Anything

This integration demonstrates how to connect LM Studio's local AI models with RAG-Anything's document processing capabilities. The system uses:

- LM Studio for local LLM inference
- nomic-embed-text-v1.5 for embeddings (768 dimensions)
- RAG-Anything for document processing and retrieval

Key benefits include:
- Privacy: All processing happens locally
- Performance: Direct API access to local models
- Flexibility: Support for various document formats
- Cost-effective: No external API usage""",
                    "page_idx": 0,
                }
            ]

            # Insert the content list using the correct method
            await self.rag.insert_content_list(
                content_list=content_list,
                file_path="lmstudio_integration_demo.txt",
                # Use a unique doc_id to avoid collisions and doc_status reuse across runs
                doc_id=f"demo-content-{uuid.uuid4()}",
                display_stats=True,
            )
            print("✅ Sample content added to knowledge base")

            print("\nTesting basic text query...")

            # Simple text query example
            result = await self.rag.aquery(
                "What are the key benefits of this LM Studio integration?",
                mode="hybrid",
            )
            print(f"✅ Query result: {result[:300]}...")

        except Exception as e:
            print(f"❌ Query failed: {str(e)}")


async def main(
    file_path: Optional[str] = None,
    mode: str = "hybrid",
    interactive: bool = True,
    parse_method: str = "txt",
):
    """Main example function."""
    print("=" * 70)
    print("Local Endpoint + RAG-Anything Integration Example")
    print("=" * 70)

    # Initialize integration
    integration = LMStudioRAGIntegration()

    # Test connection
    if not await integration.test_connection():
        return False

    print()
    if not await integration.test_chat_completion():
        return False

    # Initialize RAG
    print("\n" + "─" * 50)
    if not await integration.initialize_rag():
        return False

    if file_path:
        await integration.process_document_example(file_path, parse_method=parse_method)
        if interactive:
            await integration.interactive_query_loop(mode=mode)
        else:
            await integration.query_examples()
    else:
        # Default smoke-test path (no external file required)
        await integration.simple_query_example()

    print("\n" + "=" * 70)
    print("Integration example completed successfully!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    print("🚀 Starting local endpoint integration example...")
    parser = argparse.ArgumentParser(
        description="Run local endpoint integration example (LM Studio/Ollama)"
    )
    parser.add_argument(
        "--file",
        help="Optional path to a document for full processing + query flow",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["local", "global", "hybrid", "naive", "mix", "bypass"],
        help="Default query mode for interactive Q&A",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive Q&A and run built-in example queries instead",
    )
    parser.add_argument(
        "--parse-method",
        default=RAG_PARSE_METHOD,
        choices=["auto", "txt", "ocr"],
        help="MinerU parse method (txt recommended for text PDFs/books)",
    )
    args = parser.parse_args()

    success = asyncio.run(
        main(
            file_path=args.file,
            mode=args.mode,
            interactive=not args.no_interactive,
            parse_method=args.parse_method,
        )
    )

    exit(0 if success else 1)
