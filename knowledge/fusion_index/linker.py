from typing import Dict, List


class FusionLinker:
    def __init__(self):
        self.entity_to_chunks: Dict[str, List[str]] = {}
        self.chunk_to_entities: Dict[str, List[str]] = {}

    def link(self, entity: str, chunk_id: str) -> None:
        self.entity_to_chunks.setdefault(entity, [])
        if chunk_id not in self.entity_to_chunks[entity]:
            self.entity_to_chunks[entity].append(chunk_id)
        self.chunk_to_entities.setdefault(chunk_id, [])
        if entity not in self.chunk_to_entities[chunk_id]:
            self.chunk_to_entities[chunk_id].append(entity)

    def chunks_of_entity(self, entity: str) -> List[str]:
        return self.entity_to_chunks.get(entity, [])

    def entities_of_chunk(self, chunk_id: str) -> List[str]:
        return self.chunk_to_entities.get(chunk_id, [])