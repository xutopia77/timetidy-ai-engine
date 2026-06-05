"""
Search Engine using FAISS Index

This engine provides:
1. Text-to-image search using FAISS index
2. Image feature extraction (CLIP Vision Encoder)
3. Text feature extraction (CLIP Text Encoder)
4. Fast search with ~2-5ms performance

Design based on plan-discuss.md:
- Uses FAISS IndexIDMap2(IndexIVFFlat)
- Returns record_id instead of file_path
- Supports filters (min_similarity)
- Text feature caching for performance optimization
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from functools import lru_cache

from engines.base import BaseEngine
from engines.faiss_index_manager import FAISSIndexManager
from engines.clip_encoder import CLIPEncoder


class SearchEngine(BaseEngine):
    def __init__(self):
        self.clip_encoder = CLIPEncoder()
        self.index_manager = FAISSIndexManager()
        
        self.text_cache_size = 100
        
        print(f"SearchEngine initialized:", file=sys.stderr)
        print(f"  - CLIP encoder ready", file=sys.stderr)
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
    
    @lru_cache(maxsize=100)
    def _encode_text_cached(self, text: str) -> np.ndarray:
        return self.clip_encoder.encode_text(text)
    
    def process(self, params: dict) -> dict:
        cmd = params.get("cmd", "")
        
        if cmd == "search":
            return self._search_by_text(params)
        
        elif cmd == "encode_image":
            return self._encode_image(params)
        
        elif cmd == "encode_text":
            return self._encode_text(params)
        
        elif cmd == "get_status":
            return self._get_status()
        
        else:
            raise ValueError(f"Unknown cmd: {cmd}")
    
    def _search_by_text(self, params: dict) -> dict:
        self._ensure_index_loaded()
        
        query = params.get("query", "")
        if not query:
            raise ValueError("query is required")
        
        top_k = params.get("top_k", 10)
        filters = params.get("filters", {})
        
        start_time = time.time()
        
        text_features = self._encode_text_cached(query)
        
        if text_features.shape[0] != 1:
            text_features = text_features.reshape(1, -1)
        
        results = self.index_manager.search(text_features, top_k)
        
        if filters:
            results = self._apply_filters(results, filters)
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return {
            "query": query,
            "results": results,
            "query_time_ms": round(query_time_ms, 2),
            "total_count": len(results),
            "filters_applied": bool(filters)
        }
    
    def _apply_filters(self, results: list, filters: dict) -> list:
        min_similarity = filters.get("min_similarity", 0.0)
        
        filtered_results = []
        for result in results:
            if result["similarity"] >= min_similarity:
                filtered_results.append(result)
        
        return filtered_results
    
    def _encode_image(self, params: dict) -> dict:
        file_path = params.get("file_path", "")
        if not file_path:
            raise ValueError("file_path is required")
        
        features = self.clip_encoder.encode_image(file_path)
        
        return {
            "file_path": file_path,
            "features": features.tolist(),
            "shape": features.shape,
            "feature_dim": features.shape[1]
        }
    
    def _encode_text(self, params: dict) -> dict:
        text = params.get("text", "")
        if not text:
            raise ValueError("text is required")
        
        features = self.clip_encoder.encode_text(text)
        
        return {
            "text": text,
            "features": features.tolist(),
            "shape": features.shape,
            "feature_dim": features.shape[1]
        }
    
    def _get_status(self) -> dict:
        return {
            "clip_encoder": self.clip_encoder.get_status(),
            "index_manager": self.index_manager.get_status(),
            "text_cache_info": {
                "cache_size": self.text_cache_size,
                "cached_queries": self._encode_text_cached.cache_info()
            }
        }