from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase

from knowledge.shared.config import Settings


# 抑制Neo4j属性缺失等通知级日志，避免终端刷屏
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
logging.getLogger("neo4j").setLevel(logging.ERROR)


class Neo4jDriverWrapper:
    def __init__(self, driver, database: str = ""):
        self._driver = driver
        self._database = str(database or "").strip()

    def session(self):
        if self._database:
            return self._driver.session(database=self._database)
        return self._driver.session()

    def close(self):
        return self._driver.close()


@dataclass
class ReviewEdge:
    eid: str
    s: str
    label: str
    o: str
    verdict: str
    conf: float
    suggested: str
    reason: str


def get_driver(settings: Settings):
    if not getattr(settings, "enable_neo4j_graph", False):
        return None
    if not settings.neo4j_password:
        return None
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    return Neo4jDriverWrapper(driver, getattr(settings, "neo4j_database", ""))


def fetch_review_queue(
    driver,
    limit: int = 80,
    only_candidates: bool = True,
    exclude_rejected: bool = True,
) -> List[ReviewEdge]:
    where = []
    params: Dict[str, Any] = {"limit": int(limit)}

    if only_candidates:
        where.append("coalesce(r['checked_by'],'') = 'deepseek'")
        where.append("coalesce(r['deepseek_verdict'],'') IN ['edit','reject']")
    if exclude_rejected:
        where.append("coalesce(r.status,'auto') <> 'rejected'")

    cypher = (
        "MATCH (a:Entity)-[r:REL]->(b:Entity) "
        + ("WHERE " + " AND ".join(where) + " " if where else "")
        + "RETURN elementId(r) AS eid, a.name AS s, r.label AS label, b.name AS o, "
        "coalesce(r['deepseek_verdict'],'') AS verdict, coalesce(r['deepseek_confidence'],0) AS conf, "
        "coalesce(r['suggested_label'],'') AS suggested, coalesce(r['deepseek_reason'],'') AS reason "
        "ORDER BY coalesce(r['deepseek_confidence'],0) DESC "
        "LIMIT $limit"
    )

    out: List[ReviewEdge] = []
    with driver.session() as session:
        for r in session.run(cypher, **params):
            out.append(
                ReviewEdge(
                    eid=str(r.get("eid") or ""),
                    s=str(r.get("s") or ""),
                    label=str(r.get("label") or ""),
                    o=str(r.get("o") or ""),
                    verdict=str(r.get("verdict") or ""),
                    conf=float(r.get("conf") or 0.0),
                    suggested=str(r.get("suggested") or ""),
                    reason=str(r.get("reason") or ""),
                )
            )
    return out


def search_edges(
    driver,
    q: str,
    mode: str = "node",
    limit: int = 80,
    only_candidates: bool = True,
    exclude_rejected: bool = True,
) -> List[ReviewEdge]:
    q = str(q or "").strip()
    if not q:
        return []

    where = []
    params: Dict[str, Any] = {"q": q, "limit": int(limit)}

    if mode == "rel":
        where.append("r.label CONTAINS $q")
    else:
        where.append("(a.name CONTAINS $q OR b.name CONTAINS $q)")

    if only_candidates:
        where.append("coalesce(r['checked_by'],'') = 'deepseek'")
        where.append("coalesce(r['deepseek_verdict'],'') IN ['edit','reject']")
    if exclude_rejected:
        where.append("coalesce(r.status,'auto') <> 'rejected'")

    cypher = (
        "MATCH (a:Entity)-[r:REL]->(b:Entity) "
        + ("WHERE " + " AND ".join(where) + " " if where else "")
        + "RETURN elementId(r) AS eid, a.name AS s, r.label AS label, b.name AS o, "
        "coalesce(r['deepseek_verdict'],'') AS verdict, coalesce(r['deepseek_confidence'],0) AS conf, "
        "coalesce(r['suggested_label'],'') AS suggested, coalesce(r['deepseek_reason'],'') AS reason "
        "ORDER BY coalesce(r['deepseek_confidence'],0) DESC "
        "LIMIT $limit"
    )

    out: List[ReviewEdge] = []
    with driver.session() as session:
        for r in session.run(cypher, **params):
            out.append(
                ReviewEdge(
                    eid=str(r.get("eid") or ""),
                    s=str(r.get("s") or ""),
                    label=str(r.get("label") or ""),
                    o=str(r.get("o") or ""),
                    verdict=str(r.get("verdict") or ""),
                    conf=float(r.get("conf") or 0.0),
                    suggested=str(r.get("suggested") or ""),
                    reason=str(r.get("reason") or ""),
                )
            )
    return out


def approve_edge(driver, eid: str, note: str = "") -> None:
    eid = str(eid or "").strip()
    if not eid:
        return
    cypher = (
        "MATCH ()-[r:REL]->() WHERE elementId(r)=$eid "
        "SET r.status='approved', r.updated_at=datetime(), r.updated_by='expert' "
        "FOREACH (_ IN CASE WHEN $note <> '' THEN [1] ELSE [] END | SET r.note=$note) "
        "RETURN 1"
    )
    with driver.session() as session:
        session.run(cypher, eid=eid, note=str(note or "")[:300])


def reject_edge(driver, eid: str, note: str = "") -> None:
    eid = str(eid or "").strip()
    if not eid:
        return
    cypher = (
        "MATCH ()-[r:REL]->() WHERE elementId(r)=$eid "
        "SET r.status='rejected', r.updated_at=datetime(), r.updated_by='expert' "
        "FOREACH (_ IN CASE WHEN $note <> '' THEN [1] ELSE [] END | SET r.note=$note) "
        "RETURN 1"
    )
    with driver.session() as session:
        session.run(cypher, eid=eid, note=str(note or "")[:300])


def apply_suggested_label(driver, eid: str, note: str = "") -> None:
    eid = str(eid or "").strip()
    if not eid:
        return
    cypher = (
        "MATCH ()-[r:REL]->() WHERE elementId(r)=$eid "
        "WITH r, coalesce(r['suggested_label'],'') AS s "
        "WHERE s <> '' "
        "SET r.label = s, r.updated_at=datetime(), r.updated_by='expert' "
        "FOREACH (_ IN CASE WHEN $note <> '' THEN [1] ELSE [] END | SET r.note=$note) "
        "RETURN 1"
    )
    with driver.session() as session:
        session.run(cypher, eid=eid, note=str(note or "")[:300])


def set_label(driver, eid: str, new_label: str, note: str = "") -> None:
    eid = str(eid or "").strip()
    new_label = str(new_label or "").strip()
    if not eid or not new_label:
        return
    cypher = (
        "MATCH ()-[r:REL]->() WHERE elementId(r)=$eid "
        "SET r.label=$label, r.updated_at=datetime(), r.updated_by='expert' "
        "FOREACH (_ IN CASE WHEN $note <> '' THEN [1] ELSE [] END | SET r.note=$note) "
        "RETURN 1"
    )
    with driver.session() as session:
        session.run(cypher, eid=eid, label=new_label[:80], note=str(note or "")[:300])