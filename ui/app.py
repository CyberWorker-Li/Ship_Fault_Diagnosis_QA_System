from __future__ import annotations

import inspect
import tempfile
import subprocess
import sys
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components

from knowledge.shared.config import Settings
from knowledge.vector_store.rag_answerer import RagAnswerer
from knowledge.graph_store.graph_store import KnowledgeGraphStore
from knowledge.graph_store.triple_extractor import TripleExtractor
from knowledge.nlu.analyzer import NLUAnalyzer
from knowledge.retrieval.strategy_router import StrategyRouter
from knowledge.graph_store.graph_visualizer import GraphVisualizer, RenderOptions

from knowledge.interfaces.cli import (
    build_retriever_from_documents,
    ingest_new_files,
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
    fetch_subgraph_triples,
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


def _run_ai_graph_audit(max_edges: int, batch_size: int, max_requests: int, timeout: int, max_output_tokens: int, apply: bool) -> tuple[bool, str, str]:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "tools" / "deepseek_graph_audit.py"

    base = str(__import__("os").getenv("AI_ASSISTANT_GRAPH_AUDIT_BASE_URL", "") or "").strip()
    model = str(__import__("os").getenv("AI_ASSISTANT_GRAPH_AUDIT_MODEL", "") or "").strip()
    key = str(__import__("os").getenv("AI_ASSISTANT_GRAPH_AUDIT_API_KEY", "") or "").strip()
    if not (base and model and key):
        return False, "", "AI助审配置缺失：请先设置 AI_ASSISTANT_GRAPH_AUDIT_BASE_URL / MODEL / API_KEY 并重启程序。"

    cmd = [
        sys.executable,
        str(script),
        "--max-edges", str(int(max_edges)),
        "--batch-size", str(int(batch_size)),
        "--max-requests", str(int(max_requests)),
        "--timeout", str(int(timeout)),
        "--max-output-tokens", str(int(max_output_tokens)),
    ]
    if apply:
        cmd.append("--apply")

    total_timeout = max(180, int(timeout) * max(1, int(max_requests)) + 120)
    try:
        p = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=total_timeout,
        )
    except Exception as e:
        return False, " ".join(cmd), f"执行失败: {e}"

    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if not out.strip():
        out = "脚本无输出，请检查网络、DeepSeek Key、Neo4j连接和参数设置。"
    return p.returncode == 0, " ".join(cmd), out[-8000:]


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
        try:
            if graph_store.neo4j_ready() and getattr(settings, "neo4j_clear_on_build", False):
                graph_store.neo4j_clear()
        except Exception:
            pass
        chunks = all_chunks[: settings.graph_build_max_chunks] if settings.graph_build_max_chunks > 0 else all_chunks
        if chunks:
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
        "extractor": extractor,
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

tab_qa, tab_graph, tab_expert = st.tabs(["问答", "图谱可视化", "专家审图"])

st.sidebar.subheader("问答设置")
st.sidebar.checkbox("显示证据", key="show_evidence")
st.sidebar.checkbox("显示 Prompt", key="show_prompt")
st.sidebar.checkbox("显示调试信息", key="show_debug")
if st.sidebar.button("清空会话"):
    st.session_state.history = []

st.sidebar.divider()
st.sidebar.subheader("资料增量导入")
uploaded_files = st.sidebar.file_uploader(
    "上传 .docx/.pdf/.txt（可多选）",
    type=["docx", "pdf", "txt"],
    accept_multiple_files=True,
)
if st.sidebar.button("导入新资料", disabled=not bool(uploaded_files)):
    new_dir = Path(settings.data_dir) / "new"
    new_dir.mkdir(parents=True, exist_ok=True)
    for f in uploaded_files or []:
        name = Path(f.name).name
        target = new_dir / name
        if target.exists():
            target = new_dir / f"{target.stem}_{len(f.getvalue())}{target.suffix}"
        target.write_bytes(f.getvalue())

    with st.spinner("正在导入新资料并更新索引/图谱..."):
        res = ingest_new_files(
            settings,
            retriever=rt["retriever"],
            bm25=rt["bm25"],
            all_chunks=rt["all_chunks"],
            graph_store=rt["graph_store"],
            extractor=rt["extractor"],
            rebuild_graph=True,
        )
    if res.get("bm25") is not None:
        rt["bm25"] = res["bm25"]
    rt["doc_count"] = int(res.get("doc_count", rt["doc_count"]))
    rt["chunk_count"] = int(res.get("chunk_count", rt["chunk_count"]))
    st.sidebar.success(f"导入完成：新增文件 {res['moved']}，文档数 {res['doc_count']}，文本块 {res['chunk_count']}。")
    st.rerun()

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

with tab_graph:
    st.subheader("知识图谱可视化（性能优化版）")
    driver = get_driver(settings)
    if driver is None:
        st.error("Neo4j 未启用或缺少密码，无法生成图谱。")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            seed_query = st.text_input("起点实体关键词（可空）", value="")
            hops = st.slider("扩展跳数", min_value=1, max_value=3, value=1)
            seed_limit = st.slider("起点实体上限", min_value=3, max_value=20, value=8)
        with c2:
            max_nodes = st.slider("渲染节点上限", min_value=50, max_value=400, value=150, step=10)
            max_edges = st.slider("渲染边上限", min_value=80, max_value=800, value=250, step=10)
            min_importance = st.slider("最小重要性", min_value=1, max_value=5, value=1)
        with c3:
            limit = st.slider("查询边上限", min_value=100, max_value=2000, value=600, step=50)
            min_support = st.slider("最小支持度", min_value=1, max_value=10, value=1)
            min_conf = st.slider("最小置信度", min_value=0, max_value=100, value=0, step=5)
            show_edge_labels = st.checkbox("显示边标签", value=False)

        if st.button("生成图谱", type="primary"):
            triples = fetch_subgraph_triples(
                driver,
                q=seed_query.strip(),
                hops=int(hops),
                seed_limit=int(seed_limit),
                limit=int(limit),
                exclude_rejected=True,
                min_support=int(min_support),
                min_confidence=float(min_conf) / 100.0,
            )
            if not triples:
                st.info("未查询到可视化三元组，请放宽筛选条件。")
            else:
                vis = GraphVisualizer(
                    RenderOptions(
                        max_nodes=int(max_nodes),
                        max_edges=int(max_edges),
                        min_importance=int(min_importance),
                        show_edge_labels=bool(show_edge_labels),
                        enable_physics=False,
                    )
                )
                vis.build(triples, seed_nodes=[seed_query.strip()] if seed_query.strip() else None, max_hops=int(hops))
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
                    tmp_html = f.name
                vis.render_html(tmp_html)
                html_text = Path(tmp_html).read_text(encoding="utf-8", errors="ignore")
                st.caption(f"子图节点={vis.graph.number_of_nodes()}，子图边={vis.graph.number_of_edges()}，原始三元组={len(triples)}")
                components.html(html_text, height=820, scrolling=True)

        try:
            driver.close()
        except Exception:
            pass

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

            with st.expander("AI助审", expanded=False):
                ai_max_edges = st.number_input("审查边数", min_value=20, max_value=2000, value=80, step=20)
                ai_batch_size = st.number_input("批大小", min_value=2, max_value=50, value=8, step=2)
                ai_max_requests = st.number_input("请求轮数", min_value=1, max_value=50, value=6, step=1)
                ai_timeout = st.number_input("请求超时(秒)", min_value=60, max_value=600, value=120, step=30)
                ai_max_out = st.number_input("输出Token上限", min_value=300, max_value=8000, value=1600, step=100)
                ai_apply = st.checkbox("写回数据库(--apply)", value=True)
                st.caption("提示：执行期间页面会等待，耗时约=请求轮数×请求超时；建议先用小参数试跑。")
                if st.button("AI助审", type="primary", key="ai_audit_btn"):
                    st.info("AI助审已启动，请等待日志返回...", icon="⏳")
                    with st.spinner("AI助审执行中..."):
                        ok, cmd, log = _run_ai_graph_audit(
                            max_edges=int(ai_max_edges),
                            batch_size=int(ai_batch_size),
                            max_requests=int(ai_max_requests),
                            timeout=int(ai_timeout),
                            max_output_tokens=int(ai_max_out),
                            apply=bool(ai_apply),
                        )
                    st.caption(cmd)
                    if ok:
                        st.success("AI助审执行完成")
                        st.session_state.pop("expert_edges", None)
                    else:
                        st.error("AI助审执行失败，请查看日志")
                    st.code(log or "(无输出)")

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