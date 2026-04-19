import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from neo4j import GraphDatabase


@dataclass
class EdgeRow:
    rel_eid: str
    s: str
    p: str
    o: str
    status: str
    support: int


def _read_bat_var(bat_path: Path, var_name: str) -> str:
    if not bat_path.exists():
        return ""
    pattern = re.compile(rf"^\s*set\s+{re.escape(var_name)}=(.*)\s*$", re.IGNORECASE)
    for line in bat_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.match(line.strip())
        if m:
            return m.group(1).strip().strip('"')
    return ""


def _get_deepseek_config() -> tuple[str, str, str]:
    # 1) 审图专用配置（推荐）
    base_url = (os.getenv("AI_ASSISTANT_GRAPH_AUDIT_BASE_URL", "") or "").strip()
    model = (os.getenv("AI_ASSISTANT_GRAPH_AUDIT_MODEL", "") or "").strip()
    api_key = (os.getenv("AI_ASSISTANT_GRAPH_AUDIT_API_KEY", "") or "").strip()

    # 2) 回退到POLISH/LLM配置
    if not base_url:
        base_url = (os.getenv("AI_ASSISTANT_POLISH_BASE_URL", "") or "").strip()
    if not model:
        model = (os.getenv("AI_ASSISTANT_POLISH_MODEL", "") or "").strip()
    if not api_key:
        api_key = (os.getenv("AI_ASSISTANT_POLISH_API_KEY", "") or "").strip()

    if not base_url:
        base_url = (os.getenv("AI_ASSISTANT_LLM_BASE_URL", "") or "").strip()
    if not model:
        model = (os.getenv("AI_ASSISTANT_LLM_ANSWER_MODEL", "") or "").strip()
    if not api_key:
        api_key = (os.getenv("AI_ASSISTANT_LLM_API_KEY", "") or "").strip()

    if not base_url or not model:
        bat_path = Path(__file__).resolve().parents[1] / "run.bat"
        if not base_url:
            base_url = _read_bat_var(bat_path, "AI_ASSISTANT_GRAPH_AUDIT_BASE_URL") or _read_bat_var(bat_path, "AI_ASSISTANT_POLISH_BASE_URL") or _read_bat_var(bat_path, "AI_ASSISTANT_LLM_BASE_URL")
        if not model:
            model = _read_bat_var(bat_path, "AI_ASSISTANT_GRAPH_AUDIT_MODEL") or _read_bat_var(bat_path, "AI_ASSISTANT_POLISH_MODEL") or _read_bat_var(bat_path, "AI_ASSISTANT_LLM_ANSWER_MODEL")
        if not api_key:
            api_key = _read_bat_var(bat_path, "AI_ASSISTANT_GRAPH_AUDIT_API_KEY") or _read_bat_var(bat_path, "AI_ASSISTANT_POLISH_API_KEY") or _read_bat_var(bat_path, "AI_ASSISTANT_LLM_API_KEY")

    base_url = base_url or "https://api.deepseek.com/v1"
    model = model or "deepseek-chat"
    api_key = api_key or ""
    return base_url, model, api_key


def _get_neo4j_config() -> tuple[str, str, str, str]:
    uri = (os.getenv("AI_ASSISTANT_NEO4J_URI", "") or "").strip()
    user = (os.getenv("AI_ASSISTANT_NEO4J_USER", "") or "").strip()
    password = (os.getenv("AI_ASSISTANT_NEO4J_PASSWORD", "") or "").strip()
    database = (os.getenv("AI_ASSISTANT_NEO4J_DATABASE", "") or "").strip()

    if not uri or not user or not password or not database:
        bat_path = Path(__file__).resolve().parents[1] / "run.bat"
        if not uri:
            uri = _read_bat_var(bat_path, "AI_ASSISTANT_NEO4J_URI")
        if not user:
            user = _read_bat_var(bat_path, "AI_ASSISTANT_NEO4J_USER")
        if not password:
            password = _read_bat_var(bat_path, "AI_ASSISTANT_NEO4J_PASSWORD")
        if not database:
            database = _read_bat_var(bat_path, "AI_ASSISTANT_NEO4J_DATABASE")

    uri = uri or "bolt://localhost:7687"
    user = user or "neo4j"
    database = database or "neo4j"
    return uri, user, password, database


def _chat_deepseek(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    max_output_tokens: int,
) -> tuple[str, Dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "stream": False,
        "max_tokens": int(max_output_tokens),
    }
    # DeepSeek云端优先启用JSON对象约束，降低parse_failed概率
    if "deepseek.com" in base_url:
        payload["response_format"] = {"type": "json_object"}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"deepseek http {resp.status_code}: {resp.text[:240]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    return str(content).strip(), usage


def _safe_json_obj(text: str) -> Dict[str, Any]:
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return {}


def _estimate_cost_rmb(usage: Dict[str, Any], in_per_1k: float, out_per_1k: float) -> float:
    try:
        prompt_tokens = float(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = float(usage.get("completion_tokens", 0) or 0)
    except Exception:
        return 0.0
    return (prompt_tokens / 1000.0) * float(in_per_1k) + (completion_tokens / 1000.0) * float(out_per_1k)


RELATION_WHITELIST = {
    "发生于", "涉及部件", "导致", "表现为", "触发告警", "适用工况", "检测参数", "诊断步骤",
    "维修步骤", "需要工具", "更换备件", "风险提示", "前置条件", "后续验证", "来源案例", "证据片段",
}

_TRIVIAL_RELATIONS = {
    "是", "有", "为", "以", "将", "通过", "要求", "作为", "应提前", "将会受到", "另一方面",
}


def _normalize_label(x: str) -> str:
    x = str(x or "").strip()
    x = re.sub(r"\s+", "", x)
    x = x.replace("\u3000", "").strip("\"'“”‘’`")
    x = re.sub(r"[，,。\.；;：:、/\\\[\]\(\)（）{}<>《》]+", "", x)
    return x


def _accept_edit(original_label: str, suggested_label: str) -> bool:
    o = _normalize_label(original_label)
    s = _normalize_label(suggested_label)
    if not s or s == o:
        return False
    if s not in RELATION_WHITELIST:
        return False
    if len(s) < 2 or len(s) > 8:
        return False
    if s in _TRIVIAL_RELATIONS:
        return False
    if len(o) <= 8:
        return False
    if len(s) > len(o) - 2:
        return False
    return True


def _rule_audit_edge(e: EdgeRow) -> Optional[Dict[str, Any]]:
    if not e.s or not e.o or not e.p:
        return {"verdict": "reject", "confidence": 0.99, "reason": "字段为空", "suggested_label": ""}
    if e.s == e.o:
        return {"verdict": "reject", "confidence": 0.99, "reason": "主语与宾语相同", "suggested_label": ""}
    if len(e.s) > 80 or len(e.o) > 80:
        return {"verdict": "reject", "confidence": 0.99, "reason": "实体名异常过长", "suggested_label": ""}
    if len(e.p) > 80 or "\n" in e.p or "\r" in e.p:
        return {"verdict": "reject", "confidence": 0.99, "reason": "关系名异常", "suggested_label": ""}
    if any(x in e.p.lower() for x in ["http://", "https://", "{", "}"]):
        return {"verdict": "reject", "confidence": 0.99, "reason": "关系名包含异常符号", "suggested_label": ""}
    return None


def _select_edges(session, limit: int, min_support: int) -> List[EdgeRow]:
    cypher = (
        "MATCH (a:Entity)-[r:REL]->(b:Entity) "
        "WITH a, r, b, coalesce(r.label,'') AS p "
        "WHERE p <> '' "
        "AND coalesce(r.status,'auto') <> 'rejected' "
        "AND coalesce(r['checked_by'],'') <> 'deepseek' "
        "WITH a, r, b, p, size(p) AS lp, "
        "(CASE WHEN size(p) >= 16 THEN 30 WHEN size(p) >= 12 THEN 20 WHEN size(p) >= 9 THEN 10 ELSE 0 END "
        "+ CASE WHEN p STARTS WITH '将' THEN 5 ELSE 0 END "
        "+ CASE WHEN p CONTAINS '为例' OR p CONTAINS '例如' OR p CONTAINS '另一方面' OR p CONTAINS '给出' THEN 5 ELSE 0 END "
        "+ CASE WHEN p CONTAINS '\\n' OR p CONTAINS '\\r' THEN 10 ELSE 0 END) AS score "
        "RETURN elementId(r) AS rel_eid, a.name AS s, p AS p, b.name AS o, "
        "coalesce(r.status,'auto') AS status, coalesce(r.support,1) AS support "
        "ORDER BY (CASE WHEN coalesce(r.status,'auto')='needs_review' THEN 1 ELSE 0 END) DESC, score DESC, lp DESC, coalesce(r.support,1) DESC "
        "LIMIT $limit"
    )
    rows = []
    for rec in session.run(cypher, limit=int(limit)):
        support = int(rec["support"] or 1)
        if support < int(min_support):
            continue
        rows.append(
            EdgeRow(
                rel_eid=str(rec["rel_eid"]),
                s=str(rec["s"] or "").strip(),
                p=str(rec["p"] or "").strip(),
                o=str(rec["o"] or "").strip(),
                status=str(rec["status"] or "auto"),
                support=support,
            )
        )
    return rows


def _mark_checked(session, rel_eid: str, verdict: str, confidence: float, reason: str, suggested_label: str, apply_reject: bool, checked_by: str = "deepseek") -> None:
    cypher = (
        "MATCH ()-[r:REL]->() WHERE elementId(r)=$eid "
        "SET r.checked_by=$checked_by, r.checked_at=datetime(), "
        "r.deepseek_verdict=$verdict, r.deepseek_confidence=$confidence, r.deepseek_reason=$reason "
        "FOREACH (_ IN CASE WHEN $suggested_label <> '' THEN [1] ELSE [] END | SET r.suggested_label=$suggested_label) "
        "FOREACH (_ IN CASE WHEN $apply_reject THEN [1] ELSE [] END | SET r.status='rejected') "
        "RETURN elementId(r) AS eid"
    )
    session.run(
        cypher,
        eid=rel_eid,
        checked_by=str(checked_by or "deepseek")[:32],
        verdict=verdict,
        confidence=float(confidence),
        reason=reason[:500],
        suggested_label=suggested_label[:80] if suggested_label else "",
        apply_reject=bool(apply_reject),
    )


def _build_prompt(edges: List[EdgeRow]) -> str:
    lines = []
    for i, e in enumerate(edges, 1):
        lines.append(
            f"[{i}] ({e.s})-[:{e.p}]->({e.o}) | label_len={len(e.p)} | status={e.status} | support={e.support}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-edges", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--max-requests", type=int, default=4)
    ap.add_argument("--min-support", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--auto-reject-threshold", type=float, default=0.92)
    ap.add_argument("--budget-rmb", type=float, default=50.0)
    ap.add_argument("--price-in-per-1k", type=float, default=0.002)
    ap.add_argument("--price-out-per-1k", type=float, default=0.003)
    ap.add_argument("--max-total-tokens", type=int, default=25000)
    ap.add_argument("--max-output-tokens", type=int, default=1200)
    args = ap.parse_args()

    logging.getLogger("neo4j").setLevel(logging.ERROR)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    base_url, model, api_key = _get_deepseek_config()
    if ("deepseek.com" in base_url) and not api_key:
        raise SystemExit("DeepSeek API key not found. Set AI_ASSISTANT_GRAPH_AUDIT_API_KEY (or AI_ASSISTANT_POLISH_API_KEY).")

    neo4j_uri, neo4j_user, neo4j_password, neo4j_db = _get_neo4j_config()
    if not neo4j_password:
        raise SystemExit("Neo4j password not found. Set AI_ASSISTANT_NEO4J_PASSWORD.")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session(database=neo4j_db) as session:
            candidates = _select_edges(session, limit=int(args.max_edges), min_support=int(args.min_support))

        if not candidates:
            print("No candidate edges.")
            return

        batch_size = max(1, min(int(args.batch_size), 50))
        batches: List[List[EdgeRow]] = [candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)]
        batches = batches[: max(1, int(args.max_requests))]

        print(f"Selected edges={len(candidates)}; batches={len(batches)}; dry_run={not args.apply}")

        system_prompt = (
            "你是船舶故障诊断知识图谱审查专家。"
            "输入是一批三元组(主语,关系,宾语)。请判断每条是否合理、关系是否需要规范化、是否明显错误。"
            "关系优先规范到以下集合：发生于、涉及部件、导致、表现为、触发告警、适用工况、检测参数、诊断步骤、维修步骤、需要工具、更换备件、风险提示、前置条件、后续验证、来源案例、证据片段。"
            "只输出JSON对象，格式: "
            "{\"items\":[{\"idx\":1,\"verdict\":\"keep|reject|edit\",\"confidence\":0.0-1.0,\"suggested_label\":\"...\",\"reason\":\"...\"}]}。"
            "判定规则：不确定就keep；reject只用于明显错误/无意义/不可能关系；edit仅用于关系名过长或不规范。"
        )
   
        total_checked = 0
        total_rejected = 0
        total_edit = 0

        total_tokens = 0
        total_cost_rmb = 0.0

        for bi, batch in enumerate(batches, 1):
            if total_tokens >= int(args.max_total_tokens) or total_cost_rmb >= float(args.budget_rmb):
                break

            rule_rejects = []
            to_send = []
            for e in batch:
                r = _rule_audit_edge(e)
                if r and r.get("verdict") == "reject" and float(r.get("confidence", 0)) >= float(args.auto_reject_threshold):
                    rule_rejects.append((e, r))
                else:
                    to_send.append(e)

            if args.apply and rule_rejects:
                with driver.session(database=neo4j_db) as session:
                    for e, r in rule_rejects:
                        _mark_checked(
                            session,
                            rel_eid=e.rel_eid,
                            verdict="reject",
                            confidence=float(r.get("confidence", 0.99)),
                            reason=str(r.get("reason", "rule_reject")),
                            suggested_label="",
                            apply_reject=True,
                            checked_by="rules",
                        )

            if not to_send:
                continue

            user_prompt = "请审查以下三元组：\n" + _build_prompt(to_send)
            content, usage = _chat_deepseek(
                base_url,
                api_key,
                model,
                system_prompt,
                user_prompt,
                timeout=int(args.timeout),
                max_output_tokens=int(args.max_output_tokens),
            )
            total_cost_rmb += _estimate_cost_rmb(usage, float(args.price_in_per_1k), float(args.price_out_per_1k))
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            if total_cost_rmb > float(args.budget_rmb) or total_tokens > int(args.max_total_tokens):
                break

            data = _safe_json_obj(content)
            items = data.get("items", []) if isinstance(data, dict) else []
            if not isinstance(items, list) or not items:
                # 解析失败自动重试一次（更强约束）
                retry_prompt = user_prompt + "\n\n只允许返回严格JSON：{\"items\":[{\"idx\":1,\"verdict\":\"keep|reject|edit\",\"confidence\":0-1,\"suggested_label\":\"\",\"reason\":\"\"}]}"
                retry_content, retry_usage = _chat_deepseek(
                    base_url,
                    api_key,
                    model,
                    system_prompt,
                    retry_prompt,
                    timeout=int(args.timeout),
                    max_output_tokens=int(args.max_output_tokens),
                )
                total_cost_rmb += _estimate_cost_rmb(retry_usage, float(args.price_in_per_1k), float(args.price_out_per_1k))
                total_tokens += int(retry_usage.get("total_tokens", 0) or 0)
                data = _safe_json_obj(retry_content)
                items = data.get("items", []) if isinstance(data, dict) else []

            if not isinstance(items, list) or not items:
                print(f"batch {bi}: parse_failed -> fallback_rules")
                if args.apply:
                    with driver.session(database=neo4j_db) as session:
                        for e in to_send:
                            _mark_checked(
                                session,
                                rel_eid=e.rel_eid,
                                verdict="keep",
                                confidence=0.5,
                                reason="parse_failed_fallback",
                                suggested_label="",
                                apply_reject=False,
                                checked_by="rules",
                            )
                            total_checked += 1
                continue

            with driver.session(database=neo4j_db) as session:
                for it in items:
                    try:
                        idx = int(it.get("idx"))
                        verdict = str(it.get("verdict", "keep")).strip().lower()
                        confidence = float(it.get("confidence", 0.5))
                        reason = str(it.get("reason", "")).strip()
                        suggested_label = str(it.get("suggested_label", "")).strip()
                    except Exception:
                        continue

                    if idx < 1 or idx > len(to_send):
                        continue

                    if verdict not in {"keep", "reject", "edit"}:
                        verdict = "keep"
                    if confidence < 0:
                        confidence = 0.0
                    if confidence > 1:
                        confidence = 1.0

                    original_p = to_send[idx - 1].p
                    rel_eid = to_send[idx - 1].rel_eid

                    if verdict == "edit":
                        suggested_label = _normalize_label(suggested_label)
                        if not _accept_edit(original_p, suggested_label):
                            verdict = "keep"
                            suggested_label = ""

                    apply_reject = bool(args.apply and verdict == "reject" and confidence >= float(args.auto_reject_threshold))
                    if args.apply:
                        _mark_checked(
                            session,
                            rel_eid=rel_eid,
                            verdict=verdict,
                            confidence=confidence,
                            reason=reason,
                            suggested_label=suggested_label if verdict == "edit" else "",
                            apply_reject=apply_reject,
                            checked_by="deepseek",
                        )

                    total_checked += 1
                    if verdict == "reject" and apply_reject:
                        total_rejected += 1
                    if verdict == "edit":
                        total_edit += 1

            time.sleep(0.2)

        print(
            f"checked={total_checked} rejected_applied={total_rejected} edited_suggested={total_edit} "
            f"tokens={total_tokens} est_cost_rmb={total_cost_rmb:.2f} budget_rmb={float(args.budget_rmb):.2f}"
        )
    finally:
        try:
            driver.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()