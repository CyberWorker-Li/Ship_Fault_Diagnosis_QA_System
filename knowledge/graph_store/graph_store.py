from typing import Dict, List, Tuple
import networkx as nx
from knowledge.vector_store.indexer import ChunkRecord
from knowledge.graph_store.triple_extractor import TripleExtractor

RELATION_CANONICAL: Dict[str, str] = {
    "发生": "发生于", "发生在": "发生于", "出现于": "发生于", "发生于": "发生于",
    "涉及": "涉及部件", "涉及部件": "涉及部件",
    "导致": "导致", "引起": "导致", "造成": "导致",
    "表现": "表现为", "表现为": "表现为",
    "触发": "触发告警", "触发告警": "触发告警",
    "适用": "适用工况", "适用工况": "适用工况",
    "检测": "检测参数", "检测参数": "检测参数",
    "诊断": "诊断步骤", "诊断步骤": "诊断步骤",
    "维修": "维修步骤", "维修步骤": "维修步骤",
    "需要": "需要工具", "需要工具": "需要工具",
    "更换": "更换备件", "更换备件": "更换备件",
    "风险": "风险提示", "风险提示": "风险提示",
    "前置": "前置条件", "前置条件": "前置条件",
    "验证": "后续验证", "后续验证": "后续验证",
    "来源": "来源案例", "来源案例": "来源案例",
    "证据": "证据片段", "证据片段": "证据片段",
}

RELATION_WHITELIST = {
    "发生于", "涉及部件", "导致", "表现为", "触发告警", "适用工况", "检测参数", "诊断步骤",
    "维修步骤", "需要工具", "更换备件", "风险提示", "前置条件", "后续验证", "来源案例", "证据片段",
}

class KnowledgeGraphStore:
    def __init__(self, settings=None):
        self.graph = nx.DiGraph()
        self._neo4j_driver = None
        self._neo4j_db = None
        if settings is not None and getattr(settings, "enable_neo4j_graph", False):
            uri = str(getattr(settings, "neo4j_uri", "") or "").strip()
            user = str(getattr(settings, "neo4j_user", "") or "").strip()
            password = str(getattr(settings, "neo4j_password", "") or "").strip()
            database = str(getattr(settings, "neo4j_database", "") or "").strip() or None
            if uri and user and password:
                try:
                    from neo4j import GraphDatabase

                    self._neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
                    self._neo4j_db = database
                    self._neo4j_init_schema()
                except Exception:
                    self._neo4j_driver = None
                    self._neo4j_db = None

    def _infer_entity_type(self, name: str) -> str:
        x = str(name or "").strip()
        if any(k in x for k in ["主机", "发电机", "变频器", "报警系统", "起锚机", "罗经", "雷达", "GMDSS"]): return "设备"
        if any(k in x for k in ["调节器", "调速器", "模块", "阀组", "冷却单元", "功率模块"]): return "子系统"
        if any(k in x for k in ["线圈", "熔断器", "轴承", "阀", "电容", "电池", "探头", "泵", "电磁阀"]): return "零部件"
        if any(k in x for k in ["报警", "跳闸", "停车", "无图像", "不稳", "失效", "无穷大"]): return "故障现象"
        if any(k in x for k in ["老化", "卡滞", "堵塞", "虚接", "装反", "击穿", "故障"]): return "故障原因"
        if any(k in x for k in ["ERROR", "FAIL", "红灯"]): return "告警代码"
        if any(k in x for k in ["工况", "高温", "潮湿", "振动", "停泊", "模式"]): return "工况/环境"
        if any(k in x for k in ["电阻", "压力", "电流", "油温", "容量", "间隙", "MPa", "MΩ", "℃"]): return "检测参数"
        if any(k in x for k in ["排查", "诊断", "测试", "测量", "分析法"]): return "诊断步骤"
        if any(k in x for k in ["清理", "更换", "调整", "浸泡", "涂抹", "检修", "维修"]): return "维修步骤"
        if any(k in x for k in ["万用表", "兆欧表", "钳形电流表", "塞尺", "模块", "修理包", "氮气瓶"]): return "工具/备件"
        if any(k in x for k in ["严禁", "须", "必须", "再操作", "泄压", "挂牌"]): return "注意事项"
        if any(k in x for k in ["爆炸", "失电", "拉缸", "报废", "滞留", "风险"]): return "风险"
        if any(k in x for k in ["案例", "报告", "分析"]): return "案例"
        return "文档片段"

    def _normalize_relation(self, pred: str) -> tuple[str, bool]:
        p = str(pred or "").strip()
        if not p:
            return "证据片段", True
        if p in RELATION_WHITELIST:
            return p, False
        for k, v in RELATION_CANONICAL.items():
            if k in p:
                return v, False
        return "证据片段", True

    def _specialize_relation(self, relation: str, subject: str, obj: str, s_type: str, o_type: str) -> tuple[str, bool]:
        r = str(relation or "").strip() or "证据片段"
        s = str(subject or "")
        o = str(obj or "")
        if r != "证据片段":
            return r, False

        if any(k in o for k in ["报警", "告警", "ERROR", "FAIL", "红灯"]):
            return "触发告警", False
        if o_type in {"检测参数"} or any(k in o for k in ["电阻", "压力", "电流", "油温", "容量", "间隙", "MPa", "MΩ", "℃"]):
            return "检测参数", False
        if o_type in {"诊断步骤"} or any(k in o for k in ["排查", "诊断", "测试", "测量", "检查"]):
            return "诊断步骤", False
        if o_type in {"维修步骤"} or any(k in o for k in ["清理", "更换", "调整", "检修", "维修", "浸泡", "涂抹"]):
            return "维修步骤", False
        if o_type in {"工具/备件"} or any(k in o for k in ["万用表", "兆欧表", "钳形电流表", "塞尺", "修理包", "工具", "备件"]):
            return "需要工具", False
        if o_type in {"工况/环境"} or any(k in o for k in ["工况", "高温", "潮湿", "振动", "停泊", "模式"]):
            return "适用工况", False
        if o_type in {"风险", "注意事项"} or any(k in o for k in ["风险", "爆炸", "失电", "严禁", "必须", "泄压", "挂牌"]):
            return "风险提示", False
        if o_type in {"案例"} or any(k in o for k in ["案例", "报告", "分析"]):
            return "来源案例", False

        if o_type == "零部件" and s_type in {"设备", "子系统"}:
            return "涉及部件", False
        if o_type in {"故障现象", "告警代码"} and s_type in {"设备", "子系统", "零部件"}:
            return "表现为", False
        if s_type == "故障原因" and o_type in {"故障现象", "告警代码"}:
            return "导致", False

        if any(k in s for k in ["老化", "卡滞", "堵塞", "虚接", "装反", "击穿", "故障"]) and any(k in o for k in ["报警", "跳闸", "停车", "失效", "不稳"]):
            return "导致", False

        return "证据片段", True

    def build_from_chunks(self, chunks: List[ChunkRecord], extractor: TripleExtractor, use_llm: bool = True):
        print(f"开始从 {len(chunks)} 个文本块中提取三元组...")
        rows = []
        triple_count = 0
        for i, chunk in enumerate(chunks, 1):
            records = extractor.extract_records(chunk.text, use_llm=use_llm) if hasattr(extractor, "extract_records") else [
                {"subject": t[0], "predicate": t[1], "object": t[2], "confidence": 0.7} for t in extractor.extract(chunk.text, use_llm=use_llm)
            ]
            if not records:
                if i % 10 == 0:
                    print(f"已处理 {i}/{len(chunks)} 个文本块...")
                continue
            source_ref = str(chunk.metadata.get("source", "") or "")
            chunk_id = str(chunk.chunk_id or "")
            chunk_excerpt = str(chunk.text or "")[:220]
            for rec in records:
                s = str(rec.get("subject", "")).strip()
                o = str(rec.get("object", "")).strip()
                s_type = str(rec.get("subject_type", "") or self._infer_entity_type(s))
                o_type = str(rec.get("object_type", "") or self._infer_entity_type(o))
                p, needs_review = self._normalize_relation(rec.get("predicate", ""))
                p, needs_review = self._specialize_relation(p, s, o, s_type, o_type)
                if not (s and p and o):
                    continue
                triple_count += 1
                self.graph.add_edge(s, o, label=p)
                if self._neo4j_driver is not None:
                    conf = float(rec.get("confidence", 0.7) or 0.7)
                    evidence_key = f"{source_ref}::{chunk_id}::{s}::{p}::{o}"
                    rows.append({
                        "s": s, "p": p, "o": o,
                        "s_type": s_type,
                        "o_type": o_type,
                        "source_ref": source_ref,
                        "chunk_id": chunk_id,
                        "confidence": max(0.0, min(1.0, conf)),
                        "rel_status": "needs_review" if needs_review else "auto",
                        "evidence_key": evidence_key,
                        "excerpt": chunk_excerpt,
                    })
            if i % 10 == 0:
                print(f"已处理 {i}/{len(chunks)} 个文本块...")

        print(f"三元组提取完成，共找到 {triple_count} 个。开始构建图谱...")
        if self._neo4j_driver is not None and rows:
            self._neo4j_upsert_triples(rows)

    def _find_matching_nodes(self, entity: str) -> List[str]:
        """寻找精确匹配或包含实体的节点"""
        if self.graph.has_node(entity):
            return [entity]
        # 模糊匹配：查找包含实体名称的所有节点
        matches = [node for node in self.graph.nodes if entity in node or node in entity]
        return matches[:3] # 最多返回3个最相关的

    def retrieve_subgraph_for_entities(self, entities: List[str], k: int = 2) -> List[Tuple[str, str, str]]:
        if not entities:
            return []
        if self._neo4j_driver is not None:
            try:
                return self._neo4j_retrieve_subgraph(entities, k=k)
            except Exception:
                pass

        relevant_triples = set()
        mapped_entities = {}
        for ent in entities:
            matches = self._find_matching_nodes(ent)
            if matches:
                mapped_entities[ent] = matches

        all_actual_nodes = [node for nodes in mapped_entities.values() for node in nodes]
        for node in all_actual_nodes:
            for u, _, data in self.graph.in_edges(node, data=True):
                relevant_triples.add((u, data["label"], node))
            for _, v, data in self.graph.out_edges(node, data=True):
                relevant_triples.add((node, data["label"], v))

        entity_names = list(mapped_entities.keys())
        if len(entity_names) >= 2:
            for i in range(len(entity_names)):
                for j in range(i + 1, len(entity_names)):
                    sources = mapped_entities[entity_names[i]]
                    targets = mapped_entities[entity_names[j]]
                    for s in sources:
                        for t in targets:
                            try:
                                for path in nx.all_simple_paths(self.graph, source=s, target=t, cutoff=k):
                                    for idx in range(len(path) - 1):
                                        u, v = path[idx], path[idx + 1]
                                        relevant_triples.add((u, self.graph.get_edge_data(u, v)["label"], v))
                            except (nx.NodeNotFound, nx.NetworkXNoPath):
                                continue

        return list(relevant_triples)

    def save_to(self, path: str) -> None:
        try:
            nx.write_gpickle(self.graph, path)
        except Exception:
            pass

    def load_from(self, path: str) -> None:
        try:
            self.graph = nx.read_gpickle(path)
        except Exception:
            self.graph = nx.DiGraph()

    def close(self) -> None:
        if self._neo4j_driver is not None:
            try:
                self._neo4j_driver.close()
            except Exception:
                pass

    def neo4j_ready(self) -> bool:
        return self._neo4j_driver is not None

    def neo4j_has_data(self) -> bool:
        if self._neo4j_driver is None:
            return False
        with self._neo4j_session() as s:
            rec = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()
            return int(rec["c"]) > 0 if rec and rec.get("c") is not None else False

    def neo4j_clear(self) -> None:
        if self._neo4j_driver is None:
            return
        with self._neo4j_session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    def _neo4j_session(self):
        if self._neo4j_db:
            return self._neo4j_driver.session(database=self._neo4j_db)
        return self._neo4j_driver.session()

    def _neo4j_init_schema(self) -> None:
        if self._neo4j_driver is None:
            return
        with self._neo4j_session() as s:
            for cypher in [
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE",
                "CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)",
                "CREATE INDEX entity_type_index IF NOT EXISTS FOR (n:Entity) ON (n.entity_type)",
                "CREATE INDEX rel_label_index IF NOT EXISTS FOR ()-[r:REL]-() ON (r.label)",
                "CREATE INDEX rel_status_index IF NOT EXISTS FOR ()-[r:REL]-() ON (r.status)",
                "CREATE CONSTRAINT evidence_key_unique IF NOT EXISTS FOR (e:Evidence) REQUIRE e.key IS UNIQUE",
            ]:
                try:
                    s.run(cypher)
                except Exception:
                    pass

    def _neo4j_upsert_triples(self, rows: list[dict]) -> None:
        with self._neo4j_session() as s:
            s.run(
                "UNWIND $rows AS row "
                "MERGE (a:Entity {name: row.s}) "
                "ON CREATE SET a.entity_type = row.s_type, a.created_at = datetime() "
                "ON MATCH SET a.entity_type = coalesce(a.entity_type, row.s_type), a.updated_at = datetime() "
                "MERGE (b:Entity {name: row.o}) "
                "ON CREATE SET b.entity_type = row.o_type, b.created_at = datetime() "
                "ON MATCH SET b.entity_type = coalesce(b.entity_type, row.o_type), b.updated_at = datetime() "
                "MERGE (a)-[r:REL {label: row.p}]->(b) "
                "ON CREATE SET r.source_ref = row.source_ref, r.chunk_id = row.chunk_id, r.confidence = row.confidence, r.status = row.rel_status, r.support = 1, r.created_at = datetime() "
                "ON MATCH SET r.support = coalesce(r.support, 1) + 1, r.updated_at = datetime(), r.confidence = CASE WHEN coalesce(r.confidence, 0.0) >= row.confidence THEN r.confidence ELSE row.confidence END "
                "MERGE (e:Evidence {key: row.evidence_key}) "
                "ON CREATE SET e.source_ref = row.source_ref, e.chunk_id = row.chunk_id, e.excerpt = row.excerpt, e.created_at = datetime() "
                "ON MATCH SET e.updated_at = datetime() "
                "MERGE (a)-[:HAS_EVIDENCE {label: row.p}]->(e) "
                "MERGE (e)-[:EVIDENCE_FOR {label: row.p}]->(b)",
                rows=rows,
            )

    def _neo4j_match_nodes(self, q: str) -> list[str]:
        q = str(q or "").strip()
        if not q:
            return []
        with self._neo4j_session() as s:
            rs = s.run(
                "MATCH (n:Entity) WHERE n.name CONTAINS $q OR $q CONTAINS n.name "
                "RETURN n.name AS name LIMIT 3",
                q=q,
            )
            return [str(r["name"]) for r in rs if r.get("name")]

    def _neo4j_retrieve_subgraph(self, entities: List[str], k: int = 2) -> List[Tuple[str, str, str]]:
        mapped = {}
        for ent in entities:
            matches = self._neo4j_match_nodes(ent)
            if matches:
                mapped[ent] = matches

        nodes = [n for ns in mapped.values() for n in ns]
        triples: set[tuple[str, str, str]] = set()

        if nodes:
            with self._neo4j_session() as s:
                rs = s.run(
                    "MATCH (a:Entity)-[r:REL]->(b:Entity) "
                    "WHERE a.name IN $nodes OR b.name IN $nodes "
                    "RETURN a.name AS s, r.label AS p, b.name AS o LIMIT 60",
                    nodes=nodes,
                )
                for r in rs:
                    sv, pv, ov = r.get("s"), r.get("p"), r.get("o")
                    if sv and pv and ov:
                        triples.add((str(sv), str(pv), str(ov)))

        ent_names = list(mapped.keys())
        if len(ent_names) >= 2:
            pairs = []
            for i in range(len(ent_names)):
                for j in range(i + 1, len(ent_names)):
                    a = mapped[ent_names[i]][0]
                    b = mapped[ent_names[j]][0]
                    pairs.append((a, b))
            for a, b in pairs[:6]:
                with self._neo4j_session() as s:
                    rs = s.run(
                        "MATCH (a:Entity {name:$a}), (b:Entity {name:$b}) "
                        "MATCH p=shortestPath((a)-[:REL*..$k]-(b)) "
                        "UNWIND relationships(p) AS r "
                        "RETURN startNode(r).name AS s, r.label AS p, endNode(r).name AS o LIMIT 60",
                        a=a,
                        b=b,
                        k=int(k),
                    )
                    for r in rs:
                        sv, pv, ov = r.get("s"), r.get("p"), r.get("o")
                        if sv and pv and ov:
                            triples.add((str(sv), str(pv), str(ov)))

        return list(triples)