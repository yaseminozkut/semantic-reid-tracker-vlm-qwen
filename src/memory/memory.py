# src/memory/memory.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PersonMemory:
    def __init__(self, similarity_threshold=0.7):
        self.memory = {}  # {global_id: {"embedding": ..., "description": ..., "history": [...]}}
        self.next_global_id = 0
        self.similarity_threshold = similarity_threshold

    def add_person(self, embedding, description=None):
        """Add a new person to memory."""
        global_id = self.next_global_id
        self.memory[global_id] = {
            "embedding": embedding,
            "description": description,
            "history": []
        }
        self.next_global_id += 1
        return global_id

    def find_match(self, embedding):
        """Find the best matching person in memory."""
        if not self.memory:
            return None, -1

        best_id = None
        best_score = -1

        for global_id, person_data in self.memory.items():
            score = cosine_similarity([embedding], [person_data["embedding"]])[0, 0]
            if score > best_score:
                best_score = score
                best_id = global_id

        if best_score >= self.similarity_threshold:
            return best_id, best_score
        else:
            return None, best_score

    def get_person(self, global_id):
        """Get person data by global ID."""
        return self.memory.get(global_id)

    def update_person(self, global_id, embedding=None, description=None):
        """Update person data."""
        if global_id in self.memory:
            if embedding is not None:
                self.memory[global_id]["embedding"] = embedding
            if description is not None:
                self.memory[global_id]["description"] = description

    def get_all_ids(self):
        """Get all global IDs in memory."""
        return list(self.memory.keys())

    def get_memory_size(self):
        """Get number of persons in memory."""
        return len(self.memory)