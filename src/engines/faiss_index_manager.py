"""
FAISS Index Manager for AI Engine

This module manages FAISS vector index:
- IndexIDMap2(IndexIVFFlat) for 1万-10万张图片
- Supports deletion and custom IDs
- Persistent index files
- Index initialization, training, and maintenance
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Optional
import faiss

from utils.logger import log


class FAISSIndexManager:
    def __init__(self, index_dir: str = None, feature_dim: int = 1024):
        if index_dir is None:
            # project_root is ai-engine directory (parent of src)
            project_root = Path(__file__).parent.parent.parent
            index_dir = str(project_root / "ai-data" / "index")
        
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
        
        log.info(f"FAISS Index Manager initialized: Index directory: {self.index_dir}, Feature dimension: {feature_dim}")
    
    def is_loaded(self) -> bool:
        return self.index is not None
    
    def create_empty_index(self) -> bool:
        """
        Create an empty index that can add vectors one by one
        
        Returns:
            bool - True if successful, False otherwise
        """
        try:
            log.info(f"Creating empty FAISS index...")
            
            # For small datasets, use IndexFlatIP (brute force) which doesn't require training
            # This is more reliable and faster for <10000 images
            quantizer = faiss.IndexFlatIP(self.feature_dim)
            index_ivf = faiss.IndexIVFFlat(quantizer, self.feature_dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Enable direct_map with Hashtable for ID-based operations
            # Hashtable supports remove operations with IDSelectorArray
            index_ivf.direct_map = faiss.DirectMap()
            index_ivf.direct_map.type = faiss.DirectMap.Hashtable
            
            # Use more training points: at least 39 * nlist as recommended by FAISS
            n_train = max(self.nlist * 40, 1000)  # At least 1000 training points
            dummy_features = np.random.rand(n_train, self.feature_dim).astype('float32')
            log.info(f"Training index with {n_train} dummy vectors...")
            index_ivf.train(dummy_features)
            
            # Use IndexIVFFlat directly (not IndexIDMap2) for better remove/add support
            self.index = index_ivf
            
            self.metadata = {}
            
            log.info(f"Empty index created successfully")
            
            return True
            
        except Exception as e:
            log.error(f"Error creating empty index: {e}")
            import traceback
            log.error(traceback.format_exc())
            return False
    
    def load_index(self) -> bool:
        if not self.index_file.exists():
            log.info(f"Index file not found: {self.index_file}")
            return False
        
        if not self.metadata_file.exists():
            log.info(f"Metadata file not found: {self.metadata_file}")
            return False
        
        try:
            log.info(f"Loading FAISS index from {self.index_file}...")
            self.index = faiss.read_index(str(self.index_file))
            
            log.info(f"Loading metadata from {self.metadata_file}...")
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            
            log.info(f"Index loaded successfully:Total vectors: {self.index.ntotal}, Metadata entries: {len(self.metadata)}")
            
            return True
            
        except Exception as e:
            log.error(f"Error loading index: {e}")
            return False
    
    def build_index(self, features: np.ndarray, record_ids: list, image_hashes: list) -> bool:
        if len(features) == 0:
            log.info("No features to build index")
            return False
        
        if len(features) != len(record_ids) or len(features) != len(image_hashes):
            log.info("Mismatch in features, record_ids, and image_hashes lengths")
            return False
        
        try:
            log.info(f"Building FAISS index for {len(features)} images...")
            
            features_array = np.array(features, dtype=np.float32)
            
            if features_array.shape[1] != self.feature_dim:
                log.info(f"Feature dimension mismatch: {features_array.shape[1]} vs {self.feature_dim}")
                return False
            
            nlist = min(self.nlist, max(1, int(np.sqrt(len(features)))))
            
            log.info(f"Creating IndexIVFFlat with nlist={nlist}...")
            quantizer = faiss.IndexFlatIP(self.feature_dim)
            index_ivf = faiss.IndexIVFFlat(quantizer, self.feature_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            
            index_ivf.direct_map = faiss.DirectMap()
            index_ivf.direct_map.type = faiss.DirectMap.Array
            
            log.info(f"Training index with {len(features)} vectors...")
            index_ivf.train(features_array)
            
            log.info(f"Creating IndexIDMap2 wrapper...")
            self.index = faiss.IndexIDMap2(index_ivf)
            
            ids_array = np.array(record_ids, dtype=np.int64)
            
            log.info(f"Adding {len(features)} vectors to index...")
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
            
            log.info(f"Index built successfully:Total vectors: {self.index.ntotal}, Metadata entries: {len(self.metadata)}")
            
            return True
            
        except Exception as e:
            log.error(f"Error building index: {e}")
            import traceback
            log.error(traceback.format_exc())
            return False
    
    def save_index(self) -> bool:
        if self.index is None:
            log.info("No index to save")
            return False
        
        try:
            faiss.write_index(self.index, str(self.index_file))
            
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
            return True
            
        except Exception as e:
            log.error(f"Error saving index: {e}")
            return False
    
    def search(self, query_features: np.ndarray, top_k: int = 10) -> list:
        if not self.is_loaded():
            raise RuntimeError("Index not loaded")
        
        if query_features.shape[1] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: {query_features.shape[1]} vs {self.feature_dim}")
        
        query_array = np.array(query_features, dtype=np.float32)
        
        try:
            # Try to set nprobe for IVF index
            if hasattr(self.index, 'index'):
                # IndexIDMap2 wrapper - need to downcast inner index
                try:
                    ivf_index = faiss.downcast_index(self.index.index)
                    if hasattr(ivf_index, 'nprobe'):
                        ivf_index.nprobe = min(self.nprobe, ivf_index.nlist)
                        log.info(f"Set nprobe={ivf_index.nprobe} for IVF index (nlist={ivf_index.nlist})")
                except Exception as e:
                    log.error(f"Warning: Failed to downcast index: {e}")
            elif hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
                log.info(f"Set nprobe={self.nprobe} for index")
            else:
                log.info(f"Index does not have nprobe attribute, using default settings")
        except (AttributeError, TypeError) as e:
            log.error(f"Warning: Failed to set nprobe parameter: {e}")
            log.info(f"Continuing search with default nprobe settings") 
        
        
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
            log.error(f"Error adding vector: {e}")
            return False
    
    def remove_vector(self, record_id: int) -> bool:
        if not self.is_loaded():
            raise RuntimeError("Index not loaded")
        
        try:
            # Use IDSelectorArray for DirectMap.Hashtable
            ids_to_remove = np.array([record_id], dtype=np.int64)
            selector = faiss.IDSelectorArray(ids_to_remove)
            self.index.remove_ids(selector)
            
            if str(record_id) in self.metadata:
                self.metadata[str(record_id)]["deleted"] = True
            
            self.save_index()
            
            return True
            
        except Exception as e:
            log.error(f"Error removing vector: {e}")
            return False
    
    def get_vector(self, record_id: int) -> Optional[np.ndarray]:
        """
        Get vector by record_id
        
        Args:
            record_id: int - record ID
        
        Returns:
            np.ndarray - vector features (shape: [1, feature_dim])
            None - if vector not found
        """
        if not self.is_loaded():
            raise RuntimeError("Index not loaded")
        
        try:
            record_key = str(record_id)
            
            if record_key not in self.metadata:
                log.info(f"Record {record_id} not found in metadata")
                return None
            
            meta = self.metadata[record_key]
            if meta.get("deleted", False):
                log.info(f"Record {record_id} is marked as deleted")
                return None
            
            if hasattr(self.index, 'index') and hasattr(self.index.index, 'direct_map'):
                inner_index = self.index.index
                
                try:
                    inner_id = inner_index.direct_map.get(record_id)
                    if inner_id >= 0:
                        vector = inner_index.reconstruct(inner_id)
                        vector = np.array(vector, dtype=np.float32).reshape(1, -1)
                        log.info(f"Retrieved vector for record {record_id} via direct_map, shape: {vector.shape}")
                        return vector
                except Exception as e:
                    log.error(f"direct_map.get failed: {e}")
            
            try:
                vector = self.index.reconstruct(record_id)
                vector = np.array(vector, dtype=np.float32).reshape(1, -1)
                log.info(f"Retrieved vector for record {record_id} using reconstruct, shape: {vector.shape}")
                return vector
            except Exception as e:
                log.error(f"reconstruct failed: {e}")
            
            log.error(f"Failed to retrieve vector for record {record_id}")
            return None
            
        except Exception as e:
            log.error(f"Error getting vector for record {record_id}: {e}")
            import traceback
            log.error(traceback.format_exc())
            return None
    
    def search_by_vector(self, query_vector: np.ndarray, top_k: int = 10, exclude_ids: List[int] = None) -> list:
        """
        Search similar vectors by query vector
        
        Args:
            query_vector: np.ndarray - query vector (shape: [1, feature_dim])
            top_k: int - number of results to return
            exclude_ids: List[int] - record IDs to exclude from results
        
        Returns:
            list - search results with record_id, similarity, rank
        """
        if not self.is_loaded():
            raise RuntimeError("Index not loaded")
        
        if query_vector.shape[1] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: {query_vector.shape[1]} vs {self.feature_dim}")
        
        query_array = np.array(query_vector, dtype=np.float32)
        
        try:
            if hasattr(self.index, 'index') and hasattr(self.index.index, 'nprobe'):
                self.index.index.nprobe = self.nprobe
            elif hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
        except (AttributeError, TypeError) as e:
            log.error(f"Warning: Failed to set nprobe: {e}")
        
        search_k = top_k + (len(exclude_ids) if exclude_ids else 0)
        
        distances, ids = self.index.search(query_array, search_k)
        
        results = []
        for i in range(len(ids[0])):
            record_id = int(ids[0][i])
            similarity = float(distances[0][i])
            
            if record_id < 0:
                continue
            
            if exclude_ids and record_id in exclude_ids:
                continue
            
            if str(record_id) in self.metadata:
                meta = self.metadata[str(record_id)]
                if not meta.get("deleted", False):
                    results.append({
                        "record_id": record_id,
                        "similarity": similarity,
                        "rank": len(results) + 1
                    })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def delete_index(self) -> bool:
        """
        Delete index files
        
        Returns:
            bool - True if successful, False otherwise
        """
        log.info(f"Deleting index files...")
        
        deleted_files = []
        
        if self.index_file.exists():
            try:
                os.remove(str(self.index_file))
                deleted_files.append(str(self.index_file))
                log.info(f"  - Deleted: {self.index_file}")
            except Exception as e:
                log.error(f"  - Failed to delete {self.index_file}: {e}")
        
        if self.metadata_file.exists():
            try:
                os.remove(str(self.metadata_file))
                deleted_files.append(str(self.metadata_file))
                log.info(f"  - Deleted: {self.metadata_file}")
            except Exception as e:
                log.error(f"  - Failed to delete {self.metadata_file}: {e}")
        
        if self.config_file.exists():
            try:
                os.remove(str(self.config_file))
                deleted_files.append(str(self.config_file))
                log.info(f"  - Deleted: {self.config_file}")
            except Exception as e:
                log.error(f"  - Failed to delete {self.config_file}: {e}")
        
        self.index = None
        self.metadata = {}
        
        log.info(f"Index deleted successfully, {len(deleted_files)} files removed")
        
        return True
    
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