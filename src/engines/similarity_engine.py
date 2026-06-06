"""
Similarity Engine for Image Similarity Search

This engine provides:
1. Find similar images by reference image record_id
2. Detect similar image groups across entire album
3. Uses FAISS index for fast vector similarity search
4. Supports similarity threshold filtering
5. Excludes reference image from results

Key Design:
- Image features are already in FAISS index (no re-extraction needed)
- Uses vector similarity search (inner product / cosine similarity)
- Supports min_similarity threshold filtering
- Excludes reference image itself from results
- detect_similar_groups uses Union-Find for transitive grouping

Performance:
- Direct FAISS index search (no re-encoding)
- Search time: ~2-5ms for single query
- detect_similar_groups: ~2-3s for 10,000 images
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from engines.base import BaseEngine
from engines.faiss_index_manager import FAISSIndexManager

from utils.logger import log


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure
    Used for grouping similar images with transitive relations
    """
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
    
    def get_groups(self) -> Dict[int, List[int]]:
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            root = self.find(i)
            groups[root].append(i)
        return dict(groups)


class SimilarityEngine(BaseEngine):
    def __init__(self):
        self.index_manager = FAISSIndexManager()
        
        log.info(f"SimilarityEngine initialized:")
        log.info(f"  - FAISS index manager ready")
    
    def _ensure_index_loaded(self):
        if not self.index_manager.is_loaded():
            if self.index_manager.index_file.exists():
                self.index_manager.load_index()
            else:
                raise RuntimeError(
                    "FAISS index not found. Please initialize index first:\n"
                    "  action: 'index_init'\n"
                    "  params: {'rebuild': false}"
                )
    
    def process(self, params: dict) -> dict:
        cmd = params.get("cmd", "")
        
        if cmd == "find_similar":
            return self._find_similar(params)
        
        elif cmd == "detect_similar_groups":
            return self._detect_similar_groups(params)
        
        elif cmd == "get_status":
            return self._get_status()
        
        else:
            raise ValueError(f"Unknown cmd: {cmd}")
    
    def _find_similar(self, params: dict) -> dict:
        """
        Find similar images for multiple reference images
        
        Args:
            params: {
                "record_ids": [1001, 1002, 1003],
                "top_k": 10,
                "min_similarity": 0.8
            }
        
        Returns:
            {
                "results": [
                    {"reference_id": 1001, "similar_images": [...]},
                    {"reference_id": 1002, "similar_images": [...]},
                    {"reference_id": 1003, "similar_images": [...]}
                ],
                "total_time_ms": 15.0,
                "batch_size": 3
            }
        """
        self._ensure_index_loaded()
        
        record_ids = params.get("record_ids", [])
        if not record_ids:
            raise ValueError("record_ids is required and must not be empty")
        
        top_k = params.get("top_k", 10)
        min_similarity = params.get("min_similarity", 0.0)
        
        start_time = time.time()
        batch_results = []
        
        for record_id in record_ids:
            reference_vector = self.index_manager.get_vector(record_id)
            
            if reference_vector is None:
                batch_results.append({
                    "reference_id": record_id,
                    "similar_images": [],
                    "total_count": 0,
                    "error": f"Reference image {record_id} not found in index"
                })
                continue
            
            exclude_ids = [record_id]
            results = self.index_manager.search_by_vector(
                reference_vector,
                top_k=top_k + 5,
                exclude_ids=exclude_ids
            )
            
            filtered_results = [
                r for r in results 
                if r["similarity"] >= min_similarity
            ][:top_k]
            
            batch_results.append({
                "reference_id": record_id,
                "similar_images": filtered_results,
                "total_count": len(filtered_results)
            })
        
        total_time_ms = (time.time() - start_time) * 1000
        
        log.info(f"Found similar images for {len(record_ids)} references in {total_time_ms:.2f}ms")
        
        return {
            "results": batch_results,
            "total_time_ms": round(total_time_ms, 2),
            "batch_size": len(record_ids)
        }
    
    def _detect_similar_groups(self, params: dict) -> dict:
        """
        Detect similar image groups across entire album
        
        This is an N-to-N similarity detection that finds all groups of similar images.
        Uses Union-Find algorithm for transitive grouping (A≈B, B≈C → A,B,C same group).
        
        Args:
            params: {
                "min_similarity": 0.9,
                "scope": "all",
                "record_ids": [],
                "top_k_per_image": 20
            }
        
        Returns:
            {
                "total_images": 10000,
                "total_groups": 15,
                "total_similar_pairs": 42,
                "groups": [
                    {
                        "group_id": 1,
                        "images": [
                            {"record_id": 1001, "representative": true},
                            {"record_id": 1005, "similarity_to_representative": 0.95}
                        ],
                        "avg_similarity": 0.93,
                        "pair_count": 3
                    }
                ],
                "query_time_ms": 2100
            }
        """
        self._ensure_index_loaded()
        
        min_similarity = params.get("min_similarity", 0.9)
        scope = params.get("scope", "all")
        record_ids = params.get("record_ids", [])
        top_k_per_image = params.get("top_k_per_image", 20)
        
        start_time = time.time()
        
        if scope == "partial" and record_ids:
            target_ids = [int(rid) for rid in record_ids]
        else:
            target_ids = self._get_all_record_ids()
        
        if not target_ids:
            return {
                "total_images": 0,
                "total_groups": 0,
                "total_similar_pairs": 0,
                "groups": [],
                "query_time_ms": 0.0
            }
        
        log.info(f"Detecting similar groups for {len(target_ids)} images (min_similarity={min_similarity})...")
        
        id_to_index = {rid: i for i, rid in enumerate(target_ids)}
        index_to_id = {i: rid for i, rid in enumerate(target_ids)}
        
        vectors = []
        valid_ids = []
        for rid in target_ids:
            vec = self.index_manager.get_vector(rid)
            if vec is not None:
                vectors.append(vec.flatten())
                valid_ids.append(rid)
        
        if not vectors:
            return {
                "total_images": len(target_ids),
                "total_groups": 0,
                "total_similar_pairs": 0,
                "groups": [],
                "query_time_ms": round((time.time() - start_time) * 1000, 2),
                "error": "No vectors found for given record IDs"
            }
        
        vectors_array = np.array(vectors, dtype=np.float32)
        
        log.info(f"  - Retrieved {len(vectors)} vectors from index")
        
        search_start = time.time()
        all_pairs = []
        
        batch_size = 1000
        for batch_start in range(0, len(vectors_array), batch_size):
            batch_end = min(batch_start + batch_size, len(vectors_array))
            batch_vectors = vectors_array[batch_start:batch_end]
            
            try:
                if hasattr(self.index_manager.index, 'index') and hasattr(self.index_manager.index.index, 'nprobe'):
                    self.index_manager.index.index.nprobe = self.index_manager.nprobe
                elif hasattr(self.index_manager.index, 'nprobe'):
                    self.index_manager.index.nprobe = self.index_manager.nprobe
            except (AttributeError, TypeError):
                pass
            
            distances, ids = self.index_manager.index.search(batch_vectors, top_k_per_image + 1)
            
            for i in range(len(batch_vectors)):
                query_idx = batch_start + i
                query_id = valid_ids[query_idx]
                
                for j in range(len(ids[i])):
                    neighbor_id = int(ids[i][j])
                    similarity = float(distances[i][j])
                    
                    if neighbor_id < 0:
                        continue
                    
                    if neighbor_id == query_id:
                        continue
                    
                    if similarity < min_similarity:
                        continue
                    
                    if neighbor_id not in id_to_index:
                        continue
                    
                    neighbor_idx = id_to_index[neighbor_id]
                    
                    if query_idx < neighbor_idx:
                        all_pairs.append((query_idx, neighbor_idx, similarity))
        
        search_time_ms = (time.time() - search_start) * 1000
        log.info(f"  - Found {len(all_pairs)} similar pairs in {search_time_ms:.2f}ms")
        
        uf = UnionFind(len(valid_ids))
        for idx1, idx2, _ in all_pairs:
            uf.union(idx1, idx2)
        
        raw_groups = uf.get_groups()
        
        groups = []
        group_id = 1
        for root, indices in raw_groups.items():
            if len(indices) < 2:
                continue
            
            group_images = []
            similarities_to_rep = []
            
            representative_idx = indices[0]
            representative_id = valid_ids[representative_idx]
            group_images.append({
                "record_id": representative_id,
                "representative": True
            })
            
            for idx in indices[1:]:
                img_id = valid_ids[idx]
                
                rep_vec = vectors_array[representative_idx]
                img_vec = vectors_array[idx]
                sim_to_rep = float(np.dot(rep_vec, img_vec))
                
                group_images.append({
                    "record_id": img_id,
                    "similarity_to_representative": round(sim_to_rep, 4)
                })
                similarities_to_rep.append(sim_to_rep)
            
            pair_count = sum(1 for idx1, idx2, _ in all_pairs 
                           if idx1 in indices or idx2 in indices)
            
            avg_sim = sum(similarities_to_rep) / len(similarities_to_rep) if similarities_to_rep else 0.0
            
            groups.append({
                "group_id": group_id,
                "images": group_images,
                "avg_similarity": round(avg_sim, 4),
                "pair_count": pair_count
            })
            group_id += 1
        
        groups.sort(key=lambda g: -g["pair_count"])
        for i, g in enumerate(groups):
            g["group_id"] = i + 1
        
        query_time_ms = (time.time() - start_time) * 1000
        
        log.info(f"  - Detected {len(groups)} similar groups in {query_time_ms:.2f}ms total")
        
        return {
            "total_images": len(target_ids),
            "total_groups": len(groups),
            "total_similar_pairs": len(all_pairs),
            "groups": groups,
            "query_time_ms": round(query_time_ms, 2),
            "min_similarity": min_similarity,
            "scope": scope,
            "top_k_per_image": top_k_per_image
        }
    
    def _get_all_record_ids(self) -> List[int]:
        """Get all record IDs from the index metadata"""
        if not self.index_manager.is_loaded():
            return []
        
        record_ids = []
        for key, meta in self.index_manager.metadata.items():
            if not meta.get("deleted", False):
                record_ids.append(meta.get("record_id", int(key)))
        
        return sorted(record_ids)
    
    def _get_status(self) -> dict:
        return {
            "index_manager": self.index_manager.get_status()
        }