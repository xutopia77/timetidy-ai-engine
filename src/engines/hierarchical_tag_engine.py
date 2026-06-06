"""
Hierarchical Tag Classification Engine

This engine provides:
1. Three-level classification (level1 -> level2 -> attributes)
2. Top-K candidate strategy to avoid error propagation
3. Multiple prompts per category for better stability
4. Configurable hierarchical structure via JSON

Usage:
    result = engine.process({
        "action": "classify_hierarchical",
        "file_path": "photo.jpg"
    })
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from utils.logger import log
from engines.base import BaseEngine
from engines.search_engine import SearchEngine


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
    def __init__(self, config_path=None, top_k: int = 3):
        self._search_engine = SearchEngine()
        self._config = None
        self._config_path = config_path or self._get_default_config_path()
        self._top_k = top_k
        self._level1_categories = []
        self._level2_categories = {}
        self._attribute_categories = {}
        self._level1_vectors = None
        self._level1_prompt_counts = []
        self._initialized = False  # Flag to avoid re-initialization
        
    def _get_default_config_path(self):
        script_dir = Path(__file__).parent.parent
        return script_dir / "config" / "hierarchical_tags.json"
    
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
        
        self._search_engine._ensure_model()
        self._load_config()
        self._prepare_level1()
        self._initialized = True
    
    def _prepare_level1(self):
        """Pre-encode level1 category prompts and negative prompts"""
        all_prompts = []
        self._level1_prompt_counts = []
        
        for cat in self._level1_categories:
            all_prompts.extend(cat.prompts)
            self._level1_prompt_counts.append(len(cat.prompts))
        
        self._level1_vectors = self._encode_texts_batch(all_prompts)
        
        # Encode negative prompts for level1
        settings = self._config.get("settings", {})
        enable_negative = settings.get("enable_negative_prompts", True)
        
        self._level1_neg_vectors = None
        self._level1_neg_prompt_counts = []
        
        if enable_negative:
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
    
    def _encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts in batch
        
        Args:
            texts: List of text strings
        
        Returns:
            Matrix of text embeddings (n, feature_dim)
        """
        embeddings = []
        for text in texts:
            emb = self._search_engine.encode_text(text)
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
        settings = self._config.get("settings", {})
        enable_negative = settings.get("enable_negative_prompts", True)
        neg_weight = 0.5  # Weight for negative prompt penalty
        
        similarities = vectors @ image_vector
        
        scores = []
        start_idx = 0
        for i, count in enumerate(prompt_counts):
            end_idx = start_idx + count
            avg_score = float(similarities[start_idx:end_idx].mean())
            
            # Apply negative prompt penalty
            if (enable_negative and neg_vectors is not None 
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
        """Level1 classification with Top-K candidates
        
        Args:
            image_vector: Image feature vector
        
        Returns:
            Top-K candidate list with id, name, score
        """
        settings = self._config.get("settings", {})
        top_k = settings.get("top_k_level1", self._top_k)
        
        scores = self._compute_category_scores(
            image_vector, 
            self._level1_vectors, 
            self._level1_prompt_counts,
            self._level1_neg_vectors,
            self._level1_neg_prompt_counts,
        )
        
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        
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
    ) -> Dict:
        """Level2 classification within a level1 category
        
        Args:
            image_vector: Image feature vector
            level1_id: Level1 category ID
        
        Returns:
            Best level2 category with id, name, score
        """
        categories = self._level2_categories.get(level1_id, [])
        
        if not categories:
            return {"id": "unknown", "name": "Unknown", "score": 0.0}
        
        settings = self._config.get("settings", {})
        top_k_level2 = settings.get("top_k_level2", 2)
        enable_negative = settings.get("enable_negative_prompts", True)
        
        vectors = self._encode_category_prompts(categories)
        prompt_counts = [len(cat.prompts) for cat in categories]
        
        # Encode negative prompts for level2
        neg_vectors = None
        neg_prompt_counts = []
        if enable_negative:
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
        
        sorted_indices = np.argsort(scores)[::-1][:top_k_level2]
        
        results = []
        for idx in sorted_indices:
            cat = categories[idx]
            if scores[idx] > 0:
                results.append({
                    "id": cat.id,
                    "name": cat.name,
                    "score": scores[idx],
                })
        
        if not results:
            return {"id": "unknown", "name": "Unknown", "score": 0.0}
        
        return results[0]
    
    def classify_attributes(self, image_vector: np.ndarray) -> Dict[str, List[Dict]]:
        """Attribute classification
        
        Args:
            image_vector: Image feature vector
        
        Returns:
            Attribute results for each type (time, weather, season, etc.)
        """
        settings = self._config.get("settings", {})
        top_k_attrs = settings.get("top_k_attributes", 3)
        enable_negative = settings.get("enable_negative_prompts", True)
        
        results = {}
        
        for attr_type, categories in self._attribute_categories.items():
            vectors = self._encode_category_prompts(categories)
            prompt_counts = [len(cat.prompts) for cat in categories]
            
            # Encode negative prompts for attributes
            neg_vectors = None
            neg_prompt_counts = []
            if enable_negative:
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
            
            sorted_indices = np.argsort(scores)[::-1][:top_k_attrs]
            
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
            file_path: Image file path
            record_id: Optional record ID
            tag_level: Tag level to generate ("level1", "level2", "attributes", "all")
        
        Returns:
            Hierarchical tag results with protocol format:
            {
                "data": {
                    "tags": [{"label": "tag_name", "confidence": 0.92}, ...]
                },
                "data_dbg": {
                    "primary_category": {...},
                    "level1_tags": [...],
                    "level2_tags": [...],
                    "attribute_tags": [...],
                    "process_time_ms": 50
                }
            }
        """
        import time
        start_time = time.time()
        
        settings = self._config.get("settings", {})
        level1_threshold = settings.get("level1_threshold", 0.22)
        level2_threshold = settings.get("level2_threshold", 0.18)
        attribute_threshold = settings.get("attribute_threshold", 0.20)
        
        log.info(f"Generating tags for image: {file_path}")
        log.info(f"Thresholds: level1={level1_threshold}, level2={level2_threshold}, attributes={attribute_threshold}")
        
        image_vector = self._search_engine.encode_image(file_path)
        image_vector = image_vector.squeeze(0)
        
        log.debug(f"Image vector shape: {image_vector.shape}")
        
        data_dbg = {
            "file_path": str(file_path),
            "record_id": record_id,
        }
        
        if tag_level in ["all", "level1"]:
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
            
            if level1_tags:
                data_dbg["primary_category"] = level1_tags[0]
        
        if tag_level in ["all", "level2"]:
            if data_dbg.get("primary_category"):
                parent_id = data_dbg["primary_category"]["id"]
                level2_result = self.classify_level2(image_vector, parent_id)
                
                level2_tags = []
                if level2_result["score"] >= level2_threshold:
                    level2_tags.append({
                        "id": level2_result["id"],
                        "name": level2_result["name"],
                        "confidence": round(level2_result["score"], 4),
                        "level": 2,
                        "parent_id": parent_id,
                    })
                data_dbg["level2_tags"] = level2_tags
            else:
                log.warning("No primary category, skipping level2")
                data_dbg["level2_tags"] = []
        
        if tag_level in ["all", "attributes"]:
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
        all_tags = []
        if data_dbg.get("level1_tags"):
            all_tags.extend(data_dbg["level1_tags"])
        if data_dbg.get("level2_tags"):
            all_tags.extend(data_dbg["level2_tags"])
        if data_dbg.get("attribute_tags"):
            all_tags.extend(data_dbg["attribute_tags"])
        
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
    
    def classify_hierarchical(self, file_path: str) -> Dict:
        """Full hierarchical classification
        
        Strategy:
        1. Level1 classification returns Top-K candidates
        2. Level2 classification within each level1 candidate
        3. Compute combined score = level1_score * level2_score
        4. Select path with highest combined score
        5. Attribute classification
        
        Args:
            file_path: Image file path
        
        Returns:
            Complete classification result
        """
        image_vector = self._search_engine.encode_image(file_path)
        image_vector = image_vector.squeeze(0)  # Remove batch dimension: (1, 1024) -> (1024,)
        
        level1_candidates = self.classify_level1(image_vector)
        
        all_paths = []
        for l1 in level1_candidates:
            l2 = self.classify_level2(image_vector, l1["id"])
            combined_score = l1["score"] * l2["score"]
            all_paths.append({
                "level1": l1,
                "level2": l2,
                "combined_score": combined_score,
            })
        
        best_path = max(all_paths, key=lambda x: x["combined_score"])
        all_paths_sorted = sorted(all_paths, key=lambda x: x["combined_score"], reverse=True)
        
        attributes = self.classify_attributes(image_vector)
        
        return {
            "file_path": str(file_path),
            "level1": best_path["level1"],
            "level2": best_path["level2"],
            "combined_score": best_path["combined_score"],
            "attributes": attributes,
            "all_candidates": all_paths_sorted,
        }
    
    def process(self, params: dict) -> dict:
        self._ensure_model()
        
        cmd = params.get("cmd", "generate_tags")
        
        if cmd == "generate_tags":
            file_path = params.get("file_path", "")
            record_id = params.get("record_id", None)
            tag_level = params.get("tag_level", "all")
            
            if not file_path:
                raise ValueError("file_path is required")
            
            return self.generate_tags(file_path, record_id, tag_level)
        
        elif cmd == "classify_hierarchical":
            file_path = params.get("file_path", "")
            
            if not file_path:
                raise ValueError("file_path is required")
            
            return self.classify_hierarchical(file_path)
        
        elif cmd == "get_config":
            return {
                "config_path": str(self._config_path),
                "version": self._config.get("version", "unknown"),
                "level1_categories": [
                    {"id": c.id, "name": c.name, "negative_prompts": c.negative_prompts}
                    for c in self._level1_categories
                ],
                "level2_categories": {
                    k: [
                        {"id": c.id, "name": c.name, "negative_prompts": c.negative_prompts}
                        for c in v
                    ]
                    for k, v in self._level2_categories.items()
                },
                "attribute_categories": {
                    k: [
                        {"id": c.id, "name": c.name, "negative_prompts": c.negative_prompts}
                        for c in v
                    ]
                    for k, v in self._attribute_categories.items()
                },
                "settings": self._config.get("settings", {}),
            }
        
        elif cmd == "get_available_tags":
            return {
                "level1_tags": [{"id": c.id, "name": c.name} for c in self._level1_categories],
                "level2_tags": {
                    k: [{"id": c.id, "name": c.name} for c in v] 
                    for k, v in self._level2_categories.items()
                },
                "attribute_tags": {
                    k: [{"id": c.id, "name": c.name} for c in v] 
                    for k, v in self._attribute_categories.items()
                },
            }
        
        elif cmd == "reload_config":
            self._config = None
            self._level1_vectors = None
            self._level1_neg_vectors = None
            self._load_config()
            self._prepare_level1()
            return {
                "message": "Configuration reloaded",
                "config_path": str(self._config_path),
            }
        
        else:
            raise ValueError(f"Unknown cmd: {cmd}")