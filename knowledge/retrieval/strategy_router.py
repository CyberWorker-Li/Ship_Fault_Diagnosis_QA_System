from knowledge.shared.schemas import QueryPlan


class StrategyRouter:
    def route(self, plan: QueryPlan) -> str:
        if plan.retrieval_mode in {"vector_first", "graph_first", "hybrid"}:
            return plan.retrieval_mode
        return "hybrid"