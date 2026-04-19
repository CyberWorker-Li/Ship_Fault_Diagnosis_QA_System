import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    top_k: int = int(os.getenv("AI_ASSISTANT_TOP_K", "3"))
    graph_html_output: str = os.getenv("AI_ASSISTANT_GRAPH_HTML", "knowledge_graph.html")
    data_dir: str = os.getenv("AI_ASSISTANT_DATA_DIR", "data")
    chunk_size: int = int(os.getenv("AI_ASSISTANT_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("AI_ASSISTANT_CHUNK_OVERLAP", "200"))
    embedding_model: str = os.getenv("AI_ASSISTANT_EMBEDDING_MODEL", "BAAI/bge-m3")
    reranker_model: str = os.getenv("AI_ASSISTANT_RERANKER_MODEL", "BAAI/bge-reranker-base")
    enable_rerank: bool = _env_bool("AI_ASSISTANT_ENABLE_RERANK", True)
    hf_endpoint: str = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    local_files_only: bool = _env_bool("AI_ASSISTANT_LOCAL_FILES_ONLY", False)
    enable_llm_rewrite: bool = _env_bool("AI_ASSISTANT_ENABLE_LLM_REWRITE", True)
    enable_llm_rerank: bool = _env_bool("AI_ASSISTANT_ENABLE_LLM_RERANK", True)
    enable_llm_answer: bool = _env_bool("AI_ASSISTANT_ENABLE_LLM_ANSWER", True)
    llm_framework: str = os.getenv("AI_ASSISTANT_LLM_FRAMEWORK", "langchain")
    llm_api_key: str = os.getenv("AI_ASSISTANT_LLM_API_KEY", "")
    llm_base_url: str = os.getenv("AI_ASSISTANT_LLM_BASE_URL", "http://localhost:11434/v1")
    llm_rerank_model: str = os.getenv("AI_ASSISTANT_LLM_RERANK_MODEL", "deepseek-r1:7b")
    llm_answer_model: str = os.getenv("AI_ASSISTANT_LLM_ANSWER_MODEL", "deepseek-r1:7b")
    llm_api_url: str = os.getenv("AI_ASSISTANT_LLM_API_URL", "")
    llm_timeout: int = int(os.getenv("AI_ASSISTANT_LLM_TIMEOUT", "45"))
    retrieval_candidate_k: int = int(os.getenv("AI_ASSISTANT_RETRIEVAL_CANDIDATE_K", "8"))
    enable_llm_polish: bool = _env_bool("AI_ASSISTANT_ENABLE_LLM_POLISH", True)
    polish_api_key: str = os.getenv("AI_ASSISTANT_POLISH_API_KEY", "")
    polish_base_url: str = os.getenv("AI_ASSISTANT_POLISH_BASE_URL", "")
    polish_model: str = os.getenv("AI_ASSISTANT_POLISH_MODEL", "deepseek-chat")
    enable_graph_on_start: bool = _env_bool("AI_ASSISTANT_ENABLE_GRAPH_ON_START", False)
    graph_cache_path: str = os.getenv("AI_ASSISTANT_GRAPH_CACHE_PATH", os.path.join(os.getenv("AI_ASSISTANT_DATA_DIR", "data"), "graph_store.pkl"))
    graph_build_max_chunks: int = int(os.getenv("AI_ASSISTANT_GRAPH_BUILD_MAX_CHUNKS", "0"))
    enable_debug_output: bool = _env_bool("AI_ASSISTANT_ENABLE_DEBUG_OUTPUT", False)
    concise_answer: bool = _env_bool("AI_ASSISTANT_CONCISE_ANSWER", True)
    enable_index_cache: bool = _env_bool("AI_ASSISTANT_ENABLE_INDEX_CACHE", True)
    cache_dir: str = os.getenv("AI_ASSISTANT_CACHE_DIR", os.path.join(os.getenv("AI_ASSISTANT_DATA_DIR", "data"), ".cache"))
    cache_version: str = os.getenv("AI_ASSISTANT_CACHE_VERSION", "v1")
    enable_session_memory: bool = _env_bool("AI_ASSISTANT_ENABLE_SESSION_MEMORY", True)
    session_max_turns: int = int(os.getenv("AI_ASSISTANT_SESSION_MAX_TURNS", "4"))
    session_max_chars: int = int(os.getenv("AI_ASSISTANT_SESSION_MAX_CHARS", "1200"))
    enable_neo4j_graph: bool = _env_bool("AI_ASSISTANT_ENABLE_NEO4J_GRAPH", False)
    neo4j_uri: str = os.getenv("AI_ASSISTANT_NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("AI_ASSISTANT_NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("AI_ASSISTANT_NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("AI_ASSISTANT_NEO4J_DATABASE", "neo4j")
    neo4j_clear_on_build: bool = _env_bool("AI_ASSISTANT_NEO4J_CLEAR_ON_BUILD", False)
    doc_top_n: int = int(os.getenv("AI_ASSISTANT_DOC_TOP_N", "2"))