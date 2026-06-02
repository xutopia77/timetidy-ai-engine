"""
Similarity Engine for Image Similarity Search

This engine provides:
1. Find similar images by reference image record_id
2. Uses FAISS index for fast vector similarity search
3. Supports similarity threshold filtering
4. Excludes reference image from results

Key Design:
- Image features are already in FAISS index (no re-extraction needed)
- Uses vector similarity search (inner product / cosine similarity)
- Supports min_similarity threshold filtering
- Excludes reference image itself from results

Performance:
- Direct FAISS index search (no re-encoding)
- Search time: ~2-5ms
- Supports batch similarity calculation
"""

import sys
import time
import numpy as np
from typing import Dict, List, Optional

from engines.base import BaseEngine
from engines.faiss_index_manager import FAISSIndexManager


class SimilarityEngine(BaseEngine):
    def __init__(self):
        self.index_manager = FAISSIndexManager()
        
        print(f"SimilarityEngine initialized:", file=sys.stderr)
        print(f"  - FAISS index manager ready", file=sys.stderr)
    
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
        action = params.get("action", "find_similar")
        
        if action == "find_similar":
            return self._find_similar(params)
        
        elif action == "find_similar_by_image":
            return self._find_similar_by_image(params)
        
        elif action == "batch_find_similar":
            return self._batch_find_similar(params)
        
        elif action == "get_status":
            return self._get_status()
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _find_similar(self, params: dict) -> dict:
        """
        Find similar images by reference image record_id
        
        Args:
            params: {
                "record_id": 1001,
                "top_k": 10,
                "min_similarity": 0.8
            }
        
        Returns:
            {
                "reference_id": 1001,
                "similar_images": [
                    {"record_id": 1005, "similarity": 0.95, "rank": 1},
                    {"record_id": 1020, "similarity": 0.89, "rank": 2}
                ],
                "query_time_ms": 3.2,
                "total_count": 2
            }
        """
        self._ensure_index_loaded()
        
        record_id = params.get("record_id")
        if record_id is None:
            raise ValueError("record_id is required")
        
        top_k = params.get("top_k", 10)
        min_similarity = params.get("min_similarity", 0.0)
        
        start_time = time.time()
        
        reference_vector = self.index_manager.get_vector(record_id)
        
        if reference_vector is None:
            return {
                "reference_id": record_id,
                "similar_images": [],
                "query_time_ms": 0.0,
                "total_count": 0,
                "error": f"Reference image {record_id} not found in index"
            }
        
        exclude_ids = [record_id]
        
        results = self.index_manager.search_by_vector(
            reference_vector,
            top_k=top_k + 5,
            exclude_ids=exclude_ids
        )
        
        filtered_results = []
        for result in results:
            if result["similarity"] >= min_similarity:
                filtered_results.append(result)
        
        filtered_results = filtered_results[:top_k]
        
        query_time_ms = (time.time() - start_time) * 1000
        
        print(f"Found {len(filtered_results)} similar images for record {record_id} in {query_time_ms:.2f}ms", file=sys.stderr)
        
        return {
            "reference_id": record_id,
            "similar_images": filtered_results,
            "query_time_ms": round(query_time_ms, 2),
            "total_count": len(filtered_results),
            "min_similarity": min_similarity,
            "top_k": top_k
        }
    
    def _find_similar_by_image(self, params: dict) -> dict:
        """
        Find similar images by image path (encode image first)
        
        Args:
            params: {
                "image_path": "/path/to/image.jpg",
                "top_k": 10,
                "min_similarity": 0.8
            }
        
        Returns:
            {
                "image_path": "/path/to/image.jpg",
                "similar_images": [...],
                "query_time_ms": 50.0,
                "encode_time_ms": 45.0,
                "search_time_ms": 5.0
            }
        """
        from engines.clip_encoder import CLIPEncoder
        
        self._ensure_index_loaded()
        
        image_path = params.get("image_path")
        if image_path is None:
            raise ValueError("image_path is required")
        
        top_k = params.get("top_k", 10)
        min_similarity = params.get("min_similarity", 0.0)
        
        start_time = time.time()
        
        clip_encoder = CLIPEncoder()
        
        encode_start = time.time()
        query_vector = clip_encoder.encode_image(image_path)
        encode_time_ms = (time.time() - encode_start) * 1000
        
        if query_vector.shape[0] != 1:
            query_vector = query_vector.reshape(1, -1)
        
        search_start = time.time()
        results = self.index_manager.search_by_vector(query_vector, top_k=top_k + 5)
        search_time_ms = (time.time() - search_start) * 1000
        
        filtered_results = []
        for result in results:
            if result["similarity"] >= min_similarity:
                filtered_results.append(result)
        
        filtered_results = filtered_results[:top_k]
        
        query_time_ms = (time.time() - start_time) * 1000
        
        print(f"Found {len(filtered_results)} similar images for {image_path} in {query_time_ms:.2f}ms", file=sys.stderr)
        
        return {
            "image_path": image_path,
            "similar_images": filtered_results,
            "query_time_ms": round(query_time_ms, 2),
            "encode_time_ms": round(encode_time_ms, 2),
            "search_time_ms": round(search_time_ms, 2),
            "total_count": len(filtered_results),
            "min_similarity": min_similarity,
            "top_k": top_k
        }
    
    def _batch_find_similar(self, params: dict) -> dict:
        """
        Find similar images for multiple reference images
        
        Args:
            params: {
                "record_ids": [1001, 1002, 1003],
                "top_k": 5,
                "min_similarity": 0.8
            }
        
        Returns:
            {
                "results": [
                    {"reference_id": 1001, "similar_images": [...]},
                    {"reference_id": 1002, "similar_images": [...]},
                    {"reference_id": 1003, "similar_images": [...]}
                ],
                "total_time_ms": 15.0
            }
        """
        self._ensure_index_loaded()
        
        record_ids = params.get("record_ids", [])
        if not record_ids:
            raise ValueError("record_ids is required and must not be empty")
        
        top_k = params.get("top_k", 5)
        min_similarity = params.get("min_similarity", 0.0)
        
        start_time = time.time()
        
        batch_results = []
        
        for record_id in record_ids:
            single_params = {
                "record_id": record_id,
                "top_k": top_k,
                "min_similarity": min_similarity
            }
            
            single_result = self._find_similar(single_params)
            batch_results.append(single_result)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        print(f"Batch similarity search completed for {len(record_ids)} images in {total_time_ms:.2f}ms", file=sys.stderr)
        
        return {
            "results": batch_results,
            "total_time_ms": round(total_time_ms, 2),
            "batch_size": len(record_ids)
        }
    
    def _get_status(self) -> dict:
        return {
            "index_manager": self.index_manager.get_status()
        }