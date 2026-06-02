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
        "image_path": "photo.jpg"
    })
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from engines.base import BaseEngine
from engines.search_engine import SearchEngine


@dataclass
class Category:
    """Category definition
    
    Attributes:
        id: Unique category identifier
        name: Display name
        prompts: List of prompt texts
    """
    id: str
    name: str
    prompts: List[str]


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
            
            print(
                f"Loaded hierarchical tag config:\n"
                f"  - Level1 categories: {len(self._level1_categories)}\n"
                f"  - Level2 categories: {sum(len(v) for v in self._level2_categories.values())}\n"
                f"  - Attribute categories: {sum(len(v) for v in self._attribute_categories.values())}\n"
                f"  - Config path: {self._config_path}",
                file=sys.stderr
            )
    
    def _parse_config(self):
        """Parse configuration into Category objects"""
        self._level1_categories = []
        for cat_data in self._config["level1_categories"]:
            self._level1_categories.append(Category(
                id=cat_data["id"],
                name=cat_data["name"],
                prompts=cat_data["prompts"],
            ))
        
        self._level2_categories = {}
        for level1_id, cat_list in self._config["level2_categories"].items():
            self._level2_categories[level1_id] = []
            for cat_data in cat_list:
                self._level2_categories[level1_id].append(Category(
                    id=cat_data["id"],
                    name=cat_data["name"],
                    prompts=cat_data["prompts"],
                ))
        
        self._attribute_categories = {}
        for attr_type, cat_list in self._config["attribute_categories"].items():
            self._attribute_categories[attr_type] = []
            for cat_data in cat_list:
                self._attribute_categories[attr_type].append(Category(
                    id=cat_data["id"],
                    name=cat_data["name"],
                    prompts=cat_data["prompts"],
                ))
    
    def _ensure_model(self):
        self._search_engine._ensure_model()
        self._load_config()
        self._prepare_level1()
    
    def _prepare_level1(self):
        """Pre-encode level1 category prompts"""
        all_prompts = []
        self._level1_prompt_counts = []
        
        for cat in self._level1_categories:
            all_prompts.extend(cat.prompts)
            self._level1_prompt_counts.append(len(cat.prompts))
        
        self._level1_vectors = self._encode_texts_batch(all_prompts)
        
        print(
            f"Level1 prompts encoded: {len(all_prompts)} prompts",
            file=sys.stderr
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
        prompt_counts: List[int]
    ) -> List[float]:
        """Compute similarity scores for categories
        
        Average similarity across multiple prompts per category.
        
        Args:
            image_vector: Image feature vector
            vectors: Prompt embedding matrix
            prompt_counts: Number of prompts per category
        
        Returns:
            List of average similarity scores per category
        """
        similarities = vectors @ image_vector
        
        scores = []
        start_idx = 0
        for count in prompt_counts:
            end_idx = start_idx + count
            avg_score = float(similarities[start_idx:end_idx].mean())
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
        scores = self._compute_category_scores(
            image_vector, 
            self._level1_vectors, 
            self._level1_prompt_counts
        )
        
        sorted_indices = np.argsort(scores)[::-1][:self._top_k]
        
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
        
        vectors = self._encode_category_prompts(categories)
        prompt_counts = [len(cat.prompts) for cat in categories]
        
        scores = self._compute_category_scores(image_vector, vectors, prompt_counts)
        
        best_idx = np.argmax(scores)
        best_cat = categories[best_idx]
        
        return {
            "id": best_cat.id,
            "name": best_cat.name,
            "score": scores[best_idx],
        }
    
    def classify_attributes(self, image_vector: np.ndarray) -> Dict[str, List[Dict]]:
        """Attribute classification
        
        Args:
            image_vector: Image feature vector
        
        Returns:
            Attribute results for each type (time, weather, season)
        """
        results = {}
        
        for attr_type, categories in self._attribute_categories.items():
            vectors = self._encode_category_prompts(categories)
            prompt_counts = [len(cat.prompts) for cat in categories]
            
            scores = self._compute_category_scores(image_vector, vectors, prompt_counts)
            
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
    
    def generate_tags(self, image_path: str, record_id: Optional[int] = None, tag_level: str = "all") -> Dict:
        """Generate hierarchical tags for an image
        
        Args:
            image_path: Image file path
            record_id: Optional record ID
            tag_level: Tag level to generate ("level1", "level2", "attributes", "all")
        
        Returns:
            Hierarchical tag results
        """
        settings = self._config.get("settings", {})
        level1_threshold = settings.get("level1_threshold", 0.25)
        level2_threshold = settings.get("level2_threshold", 0.20)
        attribute_threshold = settings.get("attribute_threshold", 0.25)
        
        print(f"Generating tags for image: {image_path}", file=sys.stderr)
        print(f"Thresholds: level1={level1_threshold}, level2={level2_threshold}, attributes={attribute_threshold}", file=sys.stderr)
        
        image_vector = self._search_engine.encode_image(image_path)
        image_vector = image_vector.squeeze(0)
        
        print(f"Image vector shape: {image_vector.shape}", file=sys.stderr)
        
        result = {
            "image_path": str(image_path),
            "record_id": record_id,
        }
        
        if tag_level in ["all", "level1"]:
            level1_candidates = self.classify_level1(image_vector)
            
            print(f"Level1 candidates (Top-{self._top_k}):", file=sys.stderr)
            for i, candidate in enumerate(level1_candidates, 1):
                print(f"  {i}. {candidate['name']}: score={candidate['score']:.4f} (threshold={level1_threshold})", file=sys.stderr)
            
            level1_tags = []
            for candidate in level1_candidates:
                if candidate["score"] >= level1_threshold:
                    level1_tags.append({
                        "id": candidate["id"],
                        "name": candidate["name"],
                        "confidence": round(candidate["score"], 4),
                        "level": 1,
                    })
            
            print(f"Level1 tags after filtering: {len(level1_tags)} tags", file=sys.stderr)
            result["level1_tags"] = level1_tags
            
            if level1_tags:
                result["primary_category"] = level1_tags[0]
        
        if tag_level in ["all", "level2"]:
            if result.get("primary_category"):
                parent_id = result["primary_category"]["id"]
                level2_result = self.classify_level2(image_vector, parent_id)
                
                print(f"Level2 result for parent '{parent_id}': {level2_result['name']}, score={level2_result['score']:.4f}", file=sys.stderr)
                
                level2_tags = []
                if level2_result["score"] >= level2_threshold:
                    level2_tags.append({
                        "id": level2_result["id"],
                        "name": level2_result["name"],
                        "confidence": round(level2_result["score"], 4),
                        "level": 2,
                        "parent_id": parent_id,
                    })
                
                print(f"Level2 tags after filtering: {len(level2_tags)} tags", file=sys.stderr)
                result["level2_tags"] = level2_tags
            else:
                print(f"No primary category, skipping level2", file=sys.stderr)
                result["level2_tags"] = []
        
        if tag_level in ["all", "attributes"]:
            attributes = self.classify_attributes(image_vector)
            attribute_tags = []
            
            print(f"Attributes:", file=sys.stderr)
            for attr_type, attr_results in attributes.items():
                print(f"  {attr_type}:", file=sys.stderr)
                for i, attr in enumerate(attr_results, 1):
                    print(f"    {i}. {attr['name']}: score={attr['score']:.4f} (threshold={attribute_threshold})", file=sys.stderr)
                
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
            print(f"Attribute tags after filtering: {len(attribute_tags)} tags", file=sys.stderr)
            result["attribute_tags"] = attribute_tags
        
        if tag_level == "all":
            all_tags = []
            if result.get("level1_tags"):
                all_tags.extend(result["level1_tags"])
            if result.get("level2_tags"):
                all_tags.extend(result["level2_tags"])
            if result.get("attribute_tags"):
                all_tags.extend(result["attribute_tags"])
            
            all_tags.sort(key=lambda x: x["confidence"], reverse=True)
            result["all_tags"] = all_tags
        
        print(f"Total tags generated: {len(result.get('all_tags', []))}", file=sys.stderr)
        
        return result
    
    def classify_hierarchical(self, image_path: str) -> Dict:
        """Full hierarchical classification
        
        Strategy:
        1. Level1 classification returns Top-K candidates
        2. Level2 classification within each level1 candidate
        3. Compute combined score = level1_score * level2_score
        4. Select path with highest combined score
        5. Attribute classification
        
        Args:
            image_path: Image file path
        
        Returns:
            Complete classification result
        """
        image_vector = self._search_engine.encode_image(image_path)
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
            "image_path": str(image_path),
            "level1": best_path["level1"],
            "level2": best_path["level2"],
            "combined_score": best_path["combined_score"],
            "attributes": attributes,
            "all_candidates": all_paths_sorted,
        }
    
    def process(self, params: dict) -> dict:
        self._ensure_model()
        
        action = params.get("action", "generate_tags")
        
        if action == "generate_tags":
            image_path = params.get("image_path", "")
            record_id = params.get("record_id", None)
            tag_level = params.get("tag_level", "all")
            
            if not image_path:
                raise ValueError("image_path is required")
            
            return self.generate_tags(image_path, record_id, tag_level)
        
        elif action == "classify_hierarchical":
            image_path = params.get("image_path", "")
            
            if not image_path:
                raise ValueError("image_path is required")
            
            return self.classify_hierarchical(image_path)
        
        elif action == "get_config":
            return {
                "config_path": str(self._config_path),
                "level1_categories": [{"id": c.id, "name": c.name} for c in self._level1_categories],
                "level2_categories": {
                    k: [{"id": c.id, "name": c.name} for c in v] 
                    for k, v in self._level2_categories.items()
                },
                "attribute_categories": {
                    k: [{"id": c.id, "name": c.name} for c in v] 
                    for k, v in self._attribute_categories.items()
                },
                "settings": self._config.get("settings", {}),
            }
        
        elif action == "get_available_tags":
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
        
        elif action == "reload_config":
            self._config = None
            self._level1_vectors = None
            self._load_config()
            self._prepare_level1()
            return {
                "message": "Configuration reloaded",
                "config_path": str(self._config_path),
            }
        
        else:
            raise ValueError(f"Unknown action: {action}")