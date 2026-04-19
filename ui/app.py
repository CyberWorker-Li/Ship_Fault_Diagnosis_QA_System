from __future__ import annotations

import inspect
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

import streamlit as st

from knowledge.shared.config import Settings
from knowledge.vector_store.rag_answerer import RagAnswerer
from knowledge.graph_store.graph_store import KnowledgeGraphStore
from knowledge.graph_store.triple_extractor import TripleExtractor
from knowledge.nlu.analyzer import NLUAnalyzer
from knowledge.retrieval.strategy_router import StrategyRouter

from knowledge.interfaces.cli import (
    build_retriever_from_documents,
    _clean_question,
    _hierarchical_retrieve,
    _llm_rerank_hits,
    _compress_hits,
    _to_vector_evidences,
    _build_graph_evidence,
    build_prompt,
)

from ui.neo4j_queries import (
    get_driver,
    fetch_review_queue,
    search_edges,
    approve_edge,
    reject_edge,
    apply_suggested_label,
    set_label,
)


def _history_tail_text(max_chars: int, pending_user: str = "") -> str:
    items = st.session_state.get("history", [])
    blocks = []
    pending_user = str(pending_user or "").strip()
    for m in items[-12:]:
        role = str(m.get("role", "")).strip()
        content = str(m.get("content", "")).strip()
        if role == "user" and pending_user and content == pending_user:
            continue
        if role == "user":
            blocks.append(f"用户: {content}")
        elif role == "assistant":
            blocks.append(f"助手: {content}")
    txt = "\n".join(blocks).strip()
    if max_chars <= 0:
        return ""
    return txt[-max_chars:]


def _evidence_to_dict(e: Any) -> Dict[str, Any]:
    if is_dataclass(e):
        d = asdict(e)
    else:
        d = {
            "content": getattr(e, "content", ""),
            "kind": getattr(e, "kind", ""),
            "source_ref": getattr(e, "source_ref", ""),
            "score": getattr(e, "score", None),
        }
    d["content"] = str(d.get("content") or "")
    d["kind"] = str(d.get("kind") or "")
    d["source_ref"] = str(d.get("source_ref") or "")
    return d


@st.cache_resource(show_spinner=False)
def _init_runtime() -> Dict[str, Any]:
    settings = Settings()
    retriever, bm25, all_chunks, doc_count, chunk_count = build_retriever_from_documents(settings)
    answerer = RagAnswerer(settings)
    graph_store = KnowledgeGraphStore(settings)
    extractor = TripleExtractor()
    nlu = NLUAnalyzer()
    router = StrategyRouter()

    if settings.enable_graph_on_start:
        chunks = all_chunks[: settings.graph_build_max_chunks] if settings.graph_build_max_chunks > 0 else all_chunks
        graph_store.build_from_chunks(chunks, extractor, use_llm=True)

    return {
        "settings": settings,
        "retriever": retriever,
        "bm25": bm25,
        "all_chunks": all_chunks,
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "answerer": answerer,
        "graph_store": graph_store,
        "nlu": nlu,
        "router": router,
    }


def _ask(question: str, pending_user: str = "") -> Dict[str, Any]:
    rt = _init_runtime()
    settings: Settings = rt["settings"]
    retriever = rt["retriever"]
    bm25 = rt["bm25"]
    all_chunks = rt["all_chunks"]
    answerer: RagAnswerer = rt["answerer"]
    graph_store: KnowledgeGraphStore = rt["graph_store"]
    nlu: NLUAnalyzer = rt["nlu"]
    router: StrategyRouter = rt["router"]

    q = _clean_question(question)
    plan = nlu.analyze(q)
    plan.entities = [e.replace("的", "") for e in plan.entities if e]
    plan.top_k = settings.top_k
    route_mode = router.route(plan)

    raw_hits, used_llm_rewrite = _hierarchical_retrieve(q, plan, retriever, bm25, all_chunks, settings)
    reranked_hits, llm_used, llm_reason = _llm_rerank_hits(q, raw_hits, settings, plan.intent, plan.entities)
    hits = _compress_hits(q, reranked_hits, plan.intent, plan.entities)

    current_top_k = plan.top_k + 2 if plan.question_type == "multi_hop" else plan.top_k
    hits = hits[:current_top_k]

    evidences = _to_vector_evidences(hits, all_chunks)
    if route_mode in {"graph_first", "hybrid"}:
        graph_evidences = _build_graph_evidence(graph_store, plan.entities)
        if route_mode == "graph_first":
            evidences = graph_evidences + evidences
        else:
            evidences.extend(graph_evidences)

    prompt = build_prompt(q, evidences)

    ctx = ""
    if getattr(settings, "enable_session_memory", False):
        ctx = _history_tail_text(int(getattr(settings, "session_max_chars", 1200) or 1200), pending_user=pending_user)

    sig = inspect.signature(answerer.answer)
    if "conversation_context" in sig.parameters:
        answer = answerer.answer(q, evidences, intent=plan.intent, conversation_context=ctx)
    else:
        answer = answerer.answer(q, evidences, intent=plan.intent)

    return {
        "answer": answer,
        "prompt": prompt,
        "evidences": [_evidence_to_dict(e) for e in evidences],
        "route_mode": route_mode,
        "intent": plan.intent,
        "entities": plan.entities,
        "used_llm_rewrite": used_llm_rewrite,
        "llm_rerank_used": llm_used,
        "llm_rerank_reason": llm_reason,
    }


st.set_page_config(page_title="Ship Fault Diagnosis QA", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []
if "show_debug" not in st.session_state:
    st.session_state.show_debug = True
if "show_evidence" not in st.session_state:
    st.session_state.show_evidence = True
if "show_prompt" not in st.session_state:
    st.session_state.show_prompt = False

rt = _init_runtime()
settings: Settings = rt["settings"]

st.title("Ship Fault Diagnosis QA Assistant")
st.caption(f"文档数={rt['doc_count']}  文本块数={rt['chunk_count']}  Neo4j={settings.enable_neo4j_graph}")

tab_qa, tab_expert = st.tabs(["问答", "专家审图"])

st.sidebar.subheader("问答设置")
st.sidebar.checkbox("显示证据", key="show_evidence")
st.sidebar.checkbox("显示 Prompt", key="show_prompt")
st.sidebar.checkbox("显示调试信息", key="show_debug")
if st.sidebar.button("清空会话"):
    st.session_state.history = []

with tab_qa:
    for m in st.session_state.history:
        role = str(m.get("role", "assistant") or "assistant")
        content = str(m.get("content", "") or "")
        meta = m.get("meta") or {}
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)
            if role != "assistant":
                continue
            if st.session_state.show_evidence and meta.get("evidences"):
                with st.expander("证据", expanded=False):
                    st.json(meta.get("evidences"))
            if st.session_state.show_debug and meta.get("debug"):
                with st.expander("调试信息", expanded=False):
                    st.json(meta.get("debug"))
            if st.session_state.show_prompt and meta.get("prompt"):
                with st.expander("Prompt", expanded=False):
                    st.code(str(meta.get("prompt") or ""))

    user_q = st.chat_input("请输入问题（支持多轮追问）")
    if user_q and user_q.strip():
        user_q = user_q.strip()
        st.session_state.history.append({"role": "user", "content": user_q})

        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("检索与生成中..."):
                res = _ask(user_q)
            st.markdown(res["answer"])

            debug = {
                "route_mode": res["route_mode"],
                "intent": res["intent"],
                "entities": res["entities"],
                "used_llm_rewrite": res["used_llm_rewrite"],
                "llm_rerank_used": res["llm_rerank_used"],
                "llm_rerank_reason": res["llm_rerank_reason"],
            }

            meta = {"evidences": res["evidences"], "prompt": res["prompt"], "debug": debug}
            st.session_state.history.append({"role": "assistant", "content": res["answer"], "meta": meta})

            if st.session_state.show_evidence:
                with st.expander("证据", expanded=False):
                    st.json(res["evidences"])
            if st.session_state.show_debug:
                with st.expander("调试信息", expanded=False):
                    st.json(debug)
            if st.session_state.show_prompt:
                with st.expander("Prompt", expanded=False):
                    st.code(res["prompt"])

        st.rerun()

with tab_expert:
    st.subheader("专家审图（极简）")
    driver = get_driver(settings)
    if driver is None:
        st.error("Neo4j 未启用或缺少密码。请在 run.bat 中设置 AI_ASSISTANT_ENABLE_NEO4J_GRAPH=true 与 AI_ASSISTANT_NEO4J_PASSWORD。")
    else:
        col_filters, col_list, col_detail = st.columns([1, 1, 1], vertical_alignment="top")

        with col_filters:
            st.write("筛选与搜索")
            search_text = st.text_input("搜索（节点名或关系名）", value="")
            mode = st.selectbox("搜索类型", options=[("按节点", "node"), ("按关系", "rel")], format_func=lambda x: x[0])
            only_candidates = st.checkbox("仅 DeepSeek 候选（edit/reject）", value=True)
            exclude_rejected = st.checkbox("排除已 rejected", value=True)
            limit = st.slider("加载条数", min_value=20, max_value=300, value=80, step=10)

            if st.button("刷新列表"):
                st.session_state.pop("expert_edges", None)

        if "expert_edges" not in st.session_state:
            if search_text.strip():
                st.session_state.expert_edges = search_edges(
                    driver,
                    q=search_text.strip(),
                    mode=mode[1],
                    limit=int(limit),
                    only_candidates=bool(only_candidates),
                    exclude_rejected=bool(exclude_rejected),
                )
            else:
                st.session_state.expert_edges = fetch_review_queue(
                    driver,
                    limit=int(limit),
                    only_candidates=bool(only_candidates),
                    exclude_rejected=bool(exclude_rejected),
                )

        edges = st.session_state.expert_edges

        with col_list:
            st.write(f"候选边：{len(edges)}")
            if edges:
                items = [
                    f"{i+1}. {e.s} — {e.label} — {e.o} | {e.verdict} {e.conf:.2f}"
                    + (f" | → {e.suggested}" if e.suggested else "")
                    for i, e in enumerate(edges)
                ]
                sel = st.radio("选择", options=list(range(len(items))), format_func=lambda i: items[i], index=0)
            else:
                sel = None
                st.info("当前没有候选。")

        with col_detail:
            if sel is not None and edges:
                e = edges[int(sel)]
                st.write(
                    {
                        "s": e.s,
                        "label": e.label,
                        "o": e.o,
                        "verdict": e.verdict,
                        "conf": e.conf,
                        "suggested_label": e.suggested,
                        "reason": e.reason,
                        "eid": e.eid,
                    }
                )

                note = st.text_input("备注（可选）", value="")
                new_label = st.text_input("手动改关系名（可选）", value=e.suggested or "")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    if st.button("Accept Suggestion", type="primary", disabled=not bool(e.suggested.strip())):
                        apply_suggested_label(driver, e.eid, note=note)
                        st.success("已写入：label = suggested_label")
                        st.session_state.pop("expert_edges", None)
                with c2:
                    if st.button("Reject"):
                        reject_edge(driver, e.eid, note=note)
                        st.success("已标记为 rejected")
                        st.session_state.pop("expert_edges", None)
                with c3:
                    if st.button("Approve"):
                        approve_edge(driver, e.eid, note=note)
                        st.success("已标记为 approved")
                        st.session_state.pop("expert_edges", None)
                with c4:
                    if st.button("Set Label", disabled=not bool(new_label.strip())):
                        set_label(driver, e.eid, new_label.strip(), note=note)
                        st.success("已更新关系名")
                        st.session_state.pop("expert_edges", None)

        try:
            driver.close()
        except Exception:
            pass