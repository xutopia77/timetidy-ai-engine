"""
Hierarchical Tag Classification Engine

This engine provides:
1. Three-level classification (level1 -> level2 -> attributes)
2. Top-K candidate strategy to avoid error propagation
3. Multiple prompts per category for better stability
4. Configurable hierarchical structure via JSON
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from utils.logger import log
from engines.base import BaseEngine
from engines.clip_encoder import CLIPEncoder
import config


@dataclass
class Category:
    """Category definition
    
    Attributes:
        id: Unique category identifier
        name: Display name
        prompts: List of prompt texts
        negative_prompts: List of negative prompt texts (v3.0)
    """
    id: str
    name: str
    prompts: List[str]
    negative_prompts: List[str] = None
    
    def __post_init__(self):
        if self.negative_prompts is None:
            self.negative_prompts = []


class HierarchicalTagEngine(BaseEngine):
    def __init__(self):
        self._clip_encoder = CLIPEncoder()
        self._faiss_manager = None  # FAISS index manager
        self._config = None
        self._config_path = config.get_tag_config_file()
        self._cache_path = self._get_cache_path()
        self._level1_categories = []
        self._level2_categories = {}
        self._attribute_categories = {}
        
        # Level1 caches
        self._level1_vectors = None
        self._level1_prompt_counts = []
        self._level1_neg_vectors = None
        self._level1_neg_prompt_counts = []
        
        # Level2 caches: {level1_id: (vectors, prompt_counts, neg_vectors, neg_prompt_counts)}
        self._level2_cache = {}
        
        # Attributes caches: {attr_type: (vectors, prompt_counts, neg_vectors, neg_prompt_counts)}
        self._attribute_cache = {}
        
        self._initialized = False  # Flag to avoid re-initialization
        
    def _get_cache_path(self):
        """Get cache file path"""
        cache_dir = config.get_prompt_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "hierarchical_tags_cache.npz"
    
    def _load_config(self):
        if self._config is None:
            if not os.path.exists(self._config_path):
                raise FileNotFoundError(
                    f"Hierarchical tag config not found: {self._config_path}\n"
                    f"Please create the configuration file first."
                )
            
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
            
            self._parse_config()
            
            log.info(
                f"Loaded hierarchical tag config: "
                f"Level1 categories: {len(self._level1_categories)},"
                f"Level2 categories: {sum(len(v) for v in self._level2_categories.values())},"
                f"Attribute categories: {sum(len(v) for v in self._attribute_categories.values())},"
                f"Config path: {self._config_path}"
            )
    
    def _parse_config(self):
        """Parse configuration into Category objects"""
        self._level1_categories = []
        for cat_data in self._config["level1_categories"]:
            self._level1_categories.append(Category(
                id=cat_data["id"],
                name=cat_data["name"],
                prompts=cat_data["prompts"],
                negative_prompts=cat_data.get("negative_prompts", []),
            ))
        
        self._level2_categories = {}
        for level1_id, cat_list in self._config["level2_categories"].items():
            self._level2_categories[level1_id] = []
            for cat_data in cat_list:
                self._level2_categories[level1_id].append(Category(
                    id=cat_data["id"],
                    name=cat_data["name"],
                    prompts=cat_data["prompts"],
                    negative_prompts=cat_data.get("negative_prompts", []),
                ))
        
        self._attribute_categories = {}
        for attr_type, cat_list in self._config["attribute_categories"].items():
            self._attribute_categories[attr_type] = []
            for cat_data in cat_list:
                self._attribute_categories[attr_type].append(Category(
                    id=cat_data["id"],
                    name=cat_data["name"],
                    prompts=cat_data["prompts"],
                    negative_prompts=cat_data.get("negative_prompts", []),
                ))
    
    def _ensure_model(self):
        if self._initialized:
            return
        
        self._clip_encoder._ensure_model()
        self._load_config()
        self._prepare_all_prompts()
        self._initialized = True
    
    def _ensure_faiss(self):
        """Ensure FAISS index is loaded"""
        if self._faiss_manager is None:
            from engines.faiss_index_manager import FAISSIndexManager
            self._faiss_manager = FAISSIndexManager()
        
        if not self._faiss_manager.is_loaded():
            self._faiss_manager.load_index()
    
    def _get_vector_from_faiss(self, record_id: int) -> np.ndarray:
        """Get image vector from FAISS index
        
        Args:
            record_id: Record ID in FAISS index
        
        Returns:
            Image vector (shape: (1024,))
        
        Raises:
            ValueError: If vector not found in FAISS index
        """
        self._ensure_faiss()
        
        vector = self._faiss_manager.get_vector(record_id)
        if vector is None:
            raise ValueError(f"Vector not found in FAISS index for record_id: {record_id}")
        
        return vector.squeeze(0)  # (1, 1024) -> (1024,)
    
    def _is_cache_valid(self) -> bool:
        """Check if cache file is valid and up-to-date
        
        Returns:
            True if cache is valid, False otherwise
        """
        if not os.path.exists(self._cache_path):
            return False
        
        try:
            cache = np.load(self._cache_path, allow_pickle=True)
            
            # 1. Check config file modification time
            config_mtime = os.path.getmtime(self._config_path)
            if float(cache["config_mtime"]) < config_mtime:
                log.info("Cache invalid: config file updated")
                return False
            
            # 2. Check version
            cache_version = str(cache["version"])
            config_version = self._config.get("version", "unknown")
            if cache_version != config_version:
                log.info(f"Cache invalid: version mismatch ({cache_version} vs {config_version})")
                return False
            
            return True
            
        except Exception as e:
            log.warning(f"Cache invalid: failed to load ({e})")
            return False
    
    def _load_prompt_cache(self) -> bool:
        """Load prompt vectors from cache
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            cache = np.load(self._cache_path, allow_pickle=True)
            
            # Load level1 cache
            self._level1_vectors = cache["level1_vectors"]
            self._level1_prompt_counts = list(cache["level1_prompt_counts"])
            self._level1_neg_vectors = cache.get("level1_neg_vectors", None)
            self._level1_neg_prompt_counts = list(cache.get("level1_neg_prompt_counts", []))
            
            # Load level2 cache
            level2_keys = list(cache["level2_keys"])
            for key in level2_keys:
                vectors = cache[f"level2_{key}_vectors"]
                counts = list(cache[f"level2_{key}_counts"])
                neg_vectors = cache.get(f"level2_{key}_neg_vectors", None)
                neg_counts = list(cache.get(f"level2_{key}_neg_counts", []))
                self._level2_cache[key] = (vectors, counts, neg_vectors, neg_counts)
            
            # Load attribute cache
            attr_keys = list(cache["attr_keys"])
            for key in attr_keys:
                vectors = cache[f"attr_{key}_vectors"]
                counts = list(cache[f"attr_{key}_counts"])
                neg_vectors = cache.get(f"attr_{key}_neg_vectors", None)
                neg_counts = list(cache.get(f"attr_{key}_neg_counts", []))
                self._attribute_cache[key] = (vectors, counts, neg_vectors, neg_counts)
            
            log.info(f"Prompt cache loaded from {self._cache_path}")
            return True
            
        except Exception as e:
            log.warning(f"Failed to load cache: {e}")
            return False
    
    def _save_prompt_cache(self):
        """Save prompt vectors to cache"""
        try:
            cache_data = {
                # Metadata
                "config_mtime": os.path.getmtime(self._config_path),
                "version": self._config.get("version", "unknown"),
                # Level1
                "level1_vectors": self._level1_vectors,
                "level1_prompt_counts": np.array(self._level1_prompt_counts),
                "level1_neg_vectors": self._level1_neg_vectors if self._level1_neg_vectors is not None else np.array([]),
                "level1_neg_prompt_counts": np.array(self._level1_neg_prompt_counts),
                # Level2 keys
                "level2_keys": np.array(list(self._level2_cache.keys())),
                # Attribute keys
                "attr_keys": np.array(list(self._attribute_cache.keys())),
            }
            
            # Add level2 cache data
            for key, (vectors, counts, neg_vectors, neg_counts) in self._level2_cache.items():
                cache_data[f"level2_{key}_vectors"] = vectors
                cache_data[f"level2_{key}_counts"] = np.array(counts)
                cache_data[f"level2_{key}_neg_vectors"] = neg_vectors if neg_vectors is not None else np.array([])
                cache_data[f"level2_{key}_neg_counts"] = np.array(neg_counts)
            
            # Add attribute cache data
            for key, (vectors, counts, neg_vectors, neg_counts) in self._attribute_cache.items():
                cache_data[f"attr_{key}_vectors"] = vectors
                cache_data[f"attr_{key}_counts"] = np.array(counts)
                cache_data[f"attr_{key}_neg_vectors"] = neg_vectors if neg_vectors is not None else np.array([])
                cache_data[f"attr_{key}_neg_counts"] = np.array(neg_counts)
            
            np.savez_compressed(self._cache_path, **cache_data)
            log.info(f"Prompt cache saved to {self._cache_path}")
            
        except Exception as e:
            log.warning(f"Failed to save cache: {e}")
    
    def _prepare_level1(self):
        """Pre-encode level1 category prompts and negative prompts"""
        all_prompts = []
        self._level1_prompt_counts = []
        
        for cat in self._level1_categories:
            all_prompts.extend(cat.prompts)
            self._level1_prompt_counts.append(len(cat.prompts))
        
        self._level1_vectors = self._encode_texts_batch(all_prompts)
        
        # Encode negative prompts for level1
        self._level1_neg_vectors = None
        self._level1_neg_prompt_counts = []
        
        all_neg_prompts = []
        for cat in self._level1_categories:
            all_neg_prompts.extend(cat.negative_prompts)
            self._level1_neg_prompt_counts.append(len(cat.negative_prompts))
        
        if all_neg_prompts:
            self._level1_neg_vectors = self._encode_texts_batch(all_neg_prompts)
        
        log.info(
            f"Level1 prompts encoded: {len(all_prompts)} prompts,"
            f" negative prompts: {sum(len(c.negative_prompts) for c in self._level1_categories)}"
        )
    
    def _prepare_all_prompts(self):
        """Pre-encode all prompts at initialization
        
        Uses cache if valid, otherwise encodes and saves to cache.
        """
        import time
        start_time = time.time()
        
        # Try to load from cache first
        if self._is_cache_valid() and self._load_prompt_cache():
            elapsed = int((time.time() - start_time) * 1000)
            total_prompts = (
                sum(len(c.prompts) for c in self._level1_categories) +
                sum(sum(len(c.prompts) for c in cats) for cats in self._level2_categories.values()) +
                sum(sum(len(c.prompts) for c in cats) for cats in self._attribute_categories.values())
            )
            log.info(f"Prompt cache loaded: {total_prompts} prompts in {elapsed}ms")
            return
        
        # Cache invalid or not found, encode prompts
        log.info("Encoding prompts from scratch...")
        
        # Delete old cache file to ensure clean state
        # This prevents loading corrupted cache if encoding fails
        if os.path.exists(self._cache_path):
            os.remove(self._cache_path)
            log.info(f"Deleted old cache file: {self._cache_path}")
        
        # 1. Prepare level1 prompts
        self._prepare_level1()
        
        # 2. Prepare level2 prompts for all level1 categories
        for level1_id, categories in self._level2_categories.items():
            if not categories:
                continue
            
            # Encode positive prompts
            all_prompts = []
            prompt_counts = []
            for cat in categories:
                all_prompts.extend(cat.prompts)
                prompt_counts.append(len(cat.prompts))
            vectors = self._encode_texts_batch(all_prompts)
            
            # Encode negative prompts
            neg_vectors = None
            neg_prompt_counts = []
            all_neg_prompts = []
            for cat in categories:
                all_neg_prompts.extend(cat.negative_prompts)
                neg_prompt_counts.append(len(cat.negative_prompts))
            if all_neg_prompts:
                neg_vectors = self._encode_texts_batch(all_neg_prompts)
            
            self._level2_cache[level1_id] = (vectors, prompt_counts, neg_vectors, neg_prompt_counts)
        
        # 3. Prepare attribute prompts for all attribute types
        for attr_type, categories in self._attribute_categories.items():
            if not categories:
                continue
            
            # Encode positive prompts
            all_prompts = []
            prompt_counts = []
            for cat in categories:
                all_prompts.extend(cat.prompts)
                prompt_counts.append(len(cat.prompts))
            vectors = self._encode_texts_batch(all_prompts)
            
            # Encode negative prompts
            neg_vectors = None
            neg_prompt_counts = []
            all_neg_prompts = []
            for cat in categories:
                all_neg_prompts.extend(cat.negative_prompts)
                neg_prompt_counts.append(len(cat.negative_prompts))
            if all_neg_prompts:
                neg_vectors = self._encode_texts_batch(all_neg_prompts)
            
            self._attribute_cache[attr_type] = (vectors, prompt_counts, neg_vectors, neg_prompt_counts)
        
        elapsed = int((time.time() - start_time) * 1000)
        total_prompts = (
            sum(len(c.prompts) for c in self._level1_categories) +
            sum(sum(len(c.prompts) for c in cats) for cats in self._level2_categories.values()) +
            sum(sum(len(c.prompts) for c in cats) for cats in self._attribute_categories.values())
        )
        log.info(f"All prompts pre-encoded: {total_prompts} prompts in {elapsed}ms")
        
        # Save to cache
        self._save_prompt_cache()
    
    def _encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts in batch
        
        Args:
            texts: List of text strings
        
        Returns:
            Matrix of text embeddings (n, feature_dim)
        """
        embeddings = []
        for text in texts:
            emb = self._clip_encoder.encode_text(text)
            emb = emb.squeeze(0)  # Remove batch dimension: (1, 1024) -> (1024,)
            embeddings.append(emb)
        return np.array(embeddings)
    
    def _encode_category_prompts(self, categories: List[Category]) -> np.ndarray:
        """Encode prompts for a list of categories
        
        Args:
            categories: List of Category objects
        
        Returns:
            Matrix of prompt embeddings
        """
        all_prompts = []
        for cat in categories:
            all_prompts.extend(cat.prompts)
        return self._encode_texts_batch(all_prompts)
    
    def _compute_category_scores(
        self, 
        image_vector: np.ndarray, 
        vectors: np.ndarray, 
        prompt_counts: List[int],
        neg_vectors: Optional[np.ndarray] = None,
        neg_prompt_counts: Optional[List[int]] = None,
    ) -> List[float]:
        """Compute similarity scores for categories with optional negative prompts
        
        When negative prompts are provided, the score is adjusted as:
            adjusted_score = positive_score - neg_weight * negative_score
        
        Args:
            image_vector: Image feature vector
            vectors: Prompt embedding matrix
            prompt_counts: Number of prompts per category
            neg_vectors: Negative prompt embedding matrix (optional)
            neg_prompt_counts: Number of negative prompts per category (optional)
        
        Returns:
            List of adjusted similarity scores per category
        """
        neg_weight = 0.5  # Weight for negative prompt penalty
        
        similarities = vectors @ image_vector
        
        scores = []
        start_idx = 0
        for i, count in enumerate(prompt_counts):
            end_idx = start_idx + count
            avg_score = float(similarities[start_idx:end_idx].mean())
            
            # Apply negative prompt penalty
            if (neg_vectors is not None 
                    and neg_prompt_counts is not None 
                    and neg_prompt_counts[i] > 0):
                neg_start = sum(neg_prompt_counts[:i])
                neg_end = neg_start + neg_prompt_counts[i]
                neg_similarities = neg_vectors[neg_start:neg_end] @ image_vector
                neg_score = float(neg_similarities.mean())
                avg_score = avg_score - neg_weight * neg_score
            
            scores.append(avg_score)
            start_idx = end_idx
        
        return scores
    
    def classify_level1(self, image_vector: np.ndarray) -> List[Dict]:
        """Level1 classification
        
        Args:
            image_vector: Image feature vector
        
        Returns:
            Candidate list with id, name, score (all results)
        """
        scores = self._compute_category_scores(
            image_vector, 
            self._level1_vectors, 
            self._level1_prompt_counts,
            self._level1_neg_vectors,
            self._level1_neg_prompt_counts,
        )
        
        sorted_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in sorted_indices:
            cat = self._level1_categories[idx]
            results.append({
                "id": cat.id,
                "name": cat.name,
                "score": scores[idx],
            })
        
        return results
    
    def classify_level2(
        self, 
        image_vector: np.ndarray, 
        level1_id: str
    ) -> List[Dict]:
        """Level2 classification within a level1 category
        
        Args:
            image_vector: Image feature vector
            level1_id: Level1 category ID
        
        Returns:
            Level2 category list with id, name, score (all results)
        """
        categories = self._level2_categories.get(level1_id, [])
        
        if not categories:
            return []
        
        # Use cached vectors if available
        if level1_id in self._level2_cache:
            vectors, prompt_counts, neg_vectors, neg_prompt_counts = self._level2_cache[level1_id]
        else:
            # Fallback: encode on-the-fly (should not happen if initialized properly)
            vectors = self._encode_category_prompts(categories)
            prompt_counts = [len(cat.prompts) for cat in categories]
            
            neg_vectors = None
            neg_prompt_counts = []
            all_neg_prompts = []
            for cat in categories:
                all_neg_prompts.extend(cat.negative_prompts)
                neg_prompt_counts.append(len(cat.negative_prompts))
            if all_neg_prompts:
                neg_vectors = self._encode_texts_batch(all_neg_prompts)
        
        scores = self._compute_category_scores(
            image_vector, vectors, prompt_counts,
            neg_vectors, neg_prompt_counts,
        )
        
        sorted_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in sorted_indices:
            cat = categories[idx]
            if scores[idx] > 0:
                results.append({
                    "id": cat.id,
                    "name": cat.name,
                    "score": scores[idx],
                })
        
        return results
    
    def classify_attributes(self, image_vector: np.ndarray) -> Dict[str, List[Dict]]:
        """Attribute classification
        
        Args:
            image_vector: Image feature vector
        
        Returns:
            Attribute results for each type (time, weather, season, etc.) (all results)
        """
        results = {}
        
        for attr_type, categories in self._attribute_categories.items():
            # Use cached vectors if available
            if attr_type in self._attribute_cache:
                vectors, prompt_counts, neg_vectors, neg_prompt_counts = self._attribute_cache[attr_type]
            else:
                # Fallback: encode on-the-fly (should not happen if initialized properly)
                vectors = self._encode_category_prompts(categories)
                prompt_counts = [len(cat.prompts) for cat in categories]
                
                neg_vectors = None
                neg_prompt_counts = []
                all_neg_prompts = []
                for cat in categories:
                    all_neg_prompts.extend(cat.negative_prompts)
                    neg_prompt_counts.append(len(cat.negative_prompts))
                if all_neg_prompts:
                    neg_vectors = self._encode_texts_batch(all_neg_prompts)
            
            scores = self._compute_category_scores(
                image_vector, vectors, prompt_counts,
                neg_vectors, neg_prompt_counts,
            )
            
            sorted_indices = np.argsort(scores)[::-1]
            
            attr_results = []
            for idx in sorted_indices:
                cat = categories[idx]
                attr_results.append({
                    "id": cat.id,
                    "name": cat.name,
                    "score": scores[idx],
                })
            
            results[attr_type] = attr_results
        
        return results
    
    def generate_tags(self, file_path: str, record_id: Optional[int] = None, tag_level: str = "all") -> Dict:
        """Generate hierarchical tags for an image
        
        Args:
            file_path: Image file path (used for logging only)
            record_id: Record ID in FAISS index (required)
            tag_level: Tag level to generate ("level1", "level2", "attributes", "all")
        
        Returns:
            Hierarchical tag results with protocol format:
            {
                "data": {
                    "tags": [{"label": "tag_name", "confidence": 0.92}, ...]
                },
                "data_dbg": {
                    "level1_tags": [...],
                    "level2_tags": [...],
                    "attribute_tags": [...],
                    "process_time_ms": 50
                }
            }
        
        Raises:
            ValueError: If record_id is None or vector not found in FAISS
        """
        import time
        start_time = time.time()
        
        # Validate record_id
        if record_id is None:
            raise ValueError("record_id is required for tag generation (FAISS vector lookup)")
        
        settings = self._config.get("settings", {})
        level1_threshold = settings.get("level1_threshold", 0)
        level2_threshold = settings.get("level2_threshold", 0)
        attribute_threshold = settings.get("attribute_threshold", 0)
        log.info(f"Thresholds: level1={level1_threshold}, level2={level2_threshold}, attributes={attribute_threshold}, Generating tags for record_id: {record_id}")
        
        # Get image vector from FAISS (fast lookup)
        image_vector = self._get_vector_from_faiss(record_id)
        log.debug(f"Image vector shape: {image_vector.shape}")
        
        data_dbg = {
            "record_id": record_id,
        }
        
        # Add file_path to debug data if provided
        if file_path:
            data_dbg["file_path"] = str(file_path)
        
        if tag_level in ["all", "level1"]:
            # Get all level1 candidates (no top_k limit for debug data)
            level1_candidates = self.classify_level1(image_vector)
            
            level1_tags = []
            for candidate in level1_candidates:
                if candidate["score"] >= level1_threshold:
                    level1_tags.append({
                        "id": candidate["id"],
                        "name": candidate["name"],
                        "confidence": round(candidate["score"], 4),
                        "level": 1,
                    })
            
            data_dbg["level1_tags"] = level1_tags
            
        if tag_level in ["all", "level2"]:
            if data_dbg.get("level1_tags"):
                parent_id = data_dbg["level1_tags"][0]["id"]
                # Get all level2 candidates (no top_k limit for debug data)
                level2_results = self.classify_level2(image_vector, parent_id)
                level2_tags = []
                for result in level2_results:
                    if result["score"] >= level2_threshold:
                        level2_tags.append({
                            "id": result["id"],
                            "name": result["name"],
                            "confidence": round(result["score"], 4),
                            "level": 2,
                            "parent_id": parent_id,
                        })
                data_dbg["level2_tags"] = level2_tags
            else:
                log.warning("No primary category, skipping level2")
                data_dbg["level2_tags"] = []
        
        if tag_level in ["all", "attributes"]:
            # Get all attribute candidates (no top_k limit for debug data)
            attributes = self.classify_attributes(image_vector)
            attribute_tags = []
            for attr_type, attr_results in attributes.items():
                for attr in attr_results:
                    if attr["score"] >= attribute_threshold:
                        attribute_tags.append({
                            "id": attr["id"],
                            "name": attr["name"],
                            "confidence": round(attr["score"], 4),
                            "level": "attribute",
                            "type": attr_type,
                        })
            
            attribute_tags.sort(key=lambda x: x["confidence"], reverse=True)
            data_dbg["attribute_tags"] = attribute_tags
        
        # Build simplified tags list for data.tags
        # Only keep top-1 for level1 and level2, all for attributes
        all_tags = []
        if data_dbg.get("level1_tags"):
            all_tags.append(data_dbg["level1_tags"][0])  # Only top-1
        if data_dbg.get("level2_tags"):
            all_tags.append(data_dbg["level2_tags"][0])  # Only top-1
        if data_dbg.get("attribute_tags") and False:
            all_tags.extend(data_dbg["attribute_tags"])  # All attributes
        
        all_tags.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Convert to simplified format for data.tags
        tags = [{"label": tag["name"], "confidence": tag["confidence"]} for tag in all_tags]
        
        # Calculate process time
        process_time_ms = int((time.time() - start_time) * 1000)
        data_dbg["process_time_ms"] = process_time_ms
        
        log.info(f"Total tags generated: {len(tags)}, Process time: {process_time_ms}ms")
        
        # Return protocol format
        return {
            "data": {
                "tags": tags
            },
            "data_dbg": data_dbg
        }
        
    def process(self, params: dict) -> dict:
        self._ensure_model()
        cmd = params.get("cmd", "generate_tags")
        if cmd == "generate_tags":
            file_path = params.get("file_path", "")
            record_id = params.get("record_id", None)
            tag_level = params.get("tag_level", "all")
            
            # Validate record_id (required for FAISS lookup)
            if record_id is None:
                raise ValueError("record_id is required for tag generation (FAISS vector lookup)")
            
            return self.generate_tags(file_path, record_id, tag_level)
        else:
            raise ValueError(f"Unknown cmd: {cmd}")
