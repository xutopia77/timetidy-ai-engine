"""
FAISS Index Manager for AI Engine

This module manages FAISS vector index:
- IndexIDMap2(IndexIVFFlat) for 1万-10万张图片
- Supports deletion and custom IDs
- Persistent index files
- Index initialization, training, and maintenance
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
import faiss


class FAISSIndexManager:
    def __init__(self, index_dir: str = None, feature_dim: int = 1024):
        if index_dir is None:
            project_root = Path(__file__).parent.parent
            index_dir = str(project_root / "data" / "index")
        
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_dim = feature_dim
        
        self.index_file = self.index_dir / "faiss_index.bin"
        self.metadata_file = self.index_dir / "ai_metadata.json"
        self.config_file = self.index_dir / "index_config.json"
        
        self.index = None
        self.metadata = {}
        
        self.nlist = 100
        self.nprobe = 10
        
        print(f"FAISS Index Manager initialized:", file=sys.stderr)
        print(f"  - Index directory: {self.index_dir}", file=sys.stderr)
        print(f"  - Feature dimension: {feature_dim}", file=sys.stderr)
    
    def is_loaded(self) -> bool:
        return self.index is not None
    
    def load_index(self) -> bool:
        if not self.index_file.exists():
            print(f"Index file not found: {self.index_file}", file=sys.stderr)
            return False
        
        if not self.metadata_file.exists():
            print(f"Metadata file not found: {self.metadata_file}", file=sys.stderr)
            return False
        
        try:
            print(f"Loading FAISS index from {self.index_file}...", file=sys.stderr)
            self.index = faiss.read_index(str(self.index_file))
            
            print(f"Loading metadata from {self.metadata_file}...", file=sys.stderr)
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            
            print(f"Index loaded successfully:", file=sys.stderr)
            print(f"  - Total vectors: {self.index.ntotal}", file=sys.stderr)
            print(f"  - Metadata entries: {len(self.metadata)}", file=sys.stderr)
            
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}", file=sys.stderr)
            return False
    
    def build_index(self, features: np.ndarray, record_ids: list, image_hashes: list) -> bool:
        if len(features) == 0:
            print("No features to build index", file=sys.stderr)
            return False
        
        if len(features) != len(record_ids) or len(features) != len(image_hashes):
            print("Mismatch in features, record_ids, and image_hashes lengths", file=sys.stderr)
            return False
        
        try:
            print(f"Building FAISS index for {len(features)} images...", file=sys.stderr)
            
            features_array = np.array(features, dtype=np.float32)
            
            if features_array.shape[1] != self.feature_dim:
                print(f"Feature dimension mismatch: {features_array.shape[1]} vs {self.feature_dim}", file=sys.stderr)
                return False
            
            nlist = min(self.nlist, max(1, int(np.sqrt(len(features)))))
            
            print(f"Creating IndexIVFFlat with nlist={nlist}...", file=sys.stderr)
            quantizer = faiss.IndexFlatIP(self.feature_dim)
            index_ivf = faiss.IndexIVFFlat(quantizer, self.feature_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            
            print(f"Training index with {len(features)} vectors...", file=sys.stderr)
            index_ivf.train(features_array)
            
            print(f"Creating IndexIDMap2 wrapper...", file=sys.stderr)
            self.index = faiss.IndexIDMap2(index_ivf)
            
            ids_array = np.array(record_ids, dtype=np.int64)
            
            print(f"Adding {len(features)} vectors to index...", file=sys.stderr)
            self.index.add_with_ids(features_array, ids_array)
            
            self.metadata = {}
            for i, record_id in enumerate(record_ids):
                self.metadata[str(record_id)] = {
                    "vector_id": record_id,
                    "record_id": record_id,
                    "image_hash": image_hashes[i],
                    "deleted": False
                }
            
            self.save_index()
            
            print(f"Index built successfully:", file=sys.stderr)
            print(f"  - Total vectors: {self.index.ntotal}", file=sys.stderr)
            print(f"  - Metadata entries: {len(self.metadata)}", file=sys.stderr)
            
            return True
            
        except Exception as e:
            print(f"Error building index: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return False
    
    def save_index(self) -> bool:
        if self.index is None:
            print("No index to save", file=sys.stderr)
            return False
        
        try:
            print(f"Saving FAISS index to {self.index_file}...", file=sys.stderr)
            faiss.write_index(self.index, str(self.index_file))
            
            print(f"Saving metadata to {self.metadata_file}...", file=sys.stderr)
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            config = {
                "feature_dim": self.feature_dim,
                "nlist": self.nlist,
                "nprobe": self.nprobe,
                "total_vectors": self.index.ntotal,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            
            print(f"Index saved successfully", file=sys.stderr)
            return True
            
        except Exception as e:
            print(f"Error saving index: {e}", file=sys.stderr)
            return False
    
    def search(self, query_features: np.ndarray, top_k: int = 10) -> list:
        if not self.is_loaded():
            raise RuntimeError("Index not loaded")
        
        if query_features.shape[1] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: {query_features.shape[1]} vs {self.feature_dim}")
        
        query_array = np.array(query_features, dtype=np.float32)
        
        try:
            if hasattr(self.index, 'index') and hasattr(self.index.index, 'nprobe'):
                self.index.index.nprobe = self.nprobe
                print(f"Set nprobe={self.nprobe} for index.index", file=sys.stderr)
            elif hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
                print(f"Set nprobe={self.nprobe} for index", file=sys.stderr)
            else:
                print(f"Index does not have nprobe attribute, using default settings", file=sys.stderr)
        except (AttributeError, TypeError) as e:
            print(f"Warning: Failed to set nprobe parameter: {e}", file=sys.stderr)
            print(f"Continuing search with default nprobe settings", file=sys.stderr) 
        
        
        distances, ids = self.index.search(query_array, top_k)
        
        results = []
        for i in range(len(ids[0])):
            record_id = int(ids[0][i])
            similarity = float(distances[0][i])
            
            if record_id >= 0 and str(record_id) in self.metadata:
                meta = self.metadata[str(record_id)]
                if not meta.get("deleted", False):
                    results.append({
                        "record_id": record_id,
                        "similarity": similarity,
                        "rank": i + 1
                    })
        
        return results
    
    def add_vector(self, features: np.ndarray, record_id: int, image_hash: str) -> bool:
        if not self.is_loaded():
            raise RuntimeError("Index not loaded")
        
        try:
            features_array = np.array(features, dtype=np.float32)
            if features_array.shape[0] != 1:
                features_array = features_array.reshape(1, -1)
            
            ids_array = np.array([record_id], dtype=np.int64)
            
            self.index.add_with_ids(features_array, ids_array)
            
            self.metadata[str(record_id)] = {
                "vector_id": record_id,
                "record_id": record_id,
                "image_hash": image_hash,
                "deleted": False
            }
            
            self.save_index()
            
            return True
            
        except Exception as e:
            print(f"Error adding vector: {e}", file=sys.stderr)
            return False
    
    def remove_vector(self, record_id: int) -> bool:
        if not self.is_loaded():
            raise RuntimeError("Index not loaded")
        
        try:
            ids_to_remove = np.array([record_id], dtype=np.int64)
            self.index.remove_ids(ids_to_remove)
            
            if str(record_id) in self.metadata:
                self.metadata[str(record_id)]["deleted"] = True
            
            self.save_index()
            
            return True
            
        except Exception as e:
            print(f"Error removing vector: {e}", file=sys.stderr)
            return False
    
    def get_status(self) -> dict:
        return {
            "loaded": self.is_loaded(),
            "total_vectors": self.index.ntotal if self.is_loaded() else 0,
            "metadata_entries": len(self.metadata),
            "index_file": str(self.index_file),
            "metadata_file": str(self.metadata_file),
            "feature_dim": self.feature_dim,
            "nlist": self.nlist,
            "nprobe": self.nprobe
        }