"""
Tag Classification Engine using SearchEngine

This engine provides:
1. Image tagging with predefined tags
2. Custom tag list support
3. Open vocabulary tagging

Usage:
    # Use predefined tags
    result = engine.process({
        "action": "tag_image",
        "file_path": "photo.jpg",
        "top_k": 5
    })
    
    # Use custom tags
    result = engine.process({
        "action": "tag_image",
        "file_path": "photo.jpg",
        "tags": ["猫", "狗", "风景", "人物", "美食"],
        "top_k": 3
    })
"""

import os
import numpy as np
from engines.base import BaseEngine
from engines.search_engine import SearchEngine

from utils.logger import log


class TagEngine(BaseEngine):
    DEFAULT_TAGS = [
        "人物", "风景", "建筑", "动物", "植物",
        "美食", "交通工具", "物品", "艺术", "运动",
        "宠物", "猫", "狗", "鸟", "鱼",
        "山川", "湖海", "森林", "草原", "沙漠",
        "城市", "乡村", "公园", "街道", "广场",
        "家庭", "朋友", "聚会", "旅行", "节日",
        "生日", "婚礼", "毕业", "会议", "演出",
        "日出", "日落", "夜晚", "雨天", "雪天",
        "春天", "夏天", "秋天", "冬天",
        "室内", "室外", "白天", "夜晚",
    ]
    
    def __init__(self):
        self._search_engine = SearchEngine()
        self._tag_embeddings_cache = {}
        
    def _ensure_model(self):
        self._search_engine._ensure_model()
    
    def _get_tag_embeddings(self, tags: list) -> dict:
        """
        Get embeddings for tags (with caching)
        
        Args:
            tags: List of tag strings
        
        Returns:
            Dictionary of tag -> embedding
        """
        embeddings = {}
        tags_to_compute = []
        
        for tag in tags:
            if tag in self._tag_embeddings_cache:
                embeddings[tag] = self._tag_embeddings_cache[tag]
            else:
                tags_to_compute.append(tag)
        
        if tags_to_compute:
            log.info(f"Computing embeddings for {len(tags_to_compute)} new tags...")
            for tag in tags_to_compute:
                emb = self._search_engine.encode_text(tag)
                embeddings[tag] = emb
                self._tag_embeddings_cache[tag] = emb
        
        return embeddings
    
    def process(self, params: dict) -> dict:
        self._ensure_model()
        
        cmd = params.get("cmd", "")
        
        if cmd == "tag_image":
            file_path = params.get("file_path", "")
            custom_tags = params.get("tags", None)
            top_k = params.get("top_k", 5)
            threshold = params.get("threshold", 0.2)
            
            if not file_path:
                raise ValueError("file_path is required")
            
            tags = custom_tags if custom_tags else self.DEFAULT_TAGS
            
            image_features = self._search_engine.encode_image(file_path)
            
            tag_embeddings = self._get_tag_embeddings(tags)
            
            results = []
            for tag, emb in tag_embeddings.items():
                similarity = self._search_engine.compute_similarity(image_features, emb)
                if similarity >= threshold:
                    results.append({
                        "label": tag,
                        "confidence": round(similarity, 4),
                    })
            
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "file_path": file_path,
                "tags": results[:top_k],
                "total_tags": len(tags),
            }
        
        elif cmd == "get_available_tags":
            return {
                "default_tags": self.DEFAULT_TAGS,
                "count": len(self.DEFAULT_TAGS),
            }
        
        elif cmd == "add_custom_tags":
            new_tags = params.get("tags", [])
            if not new_tags:
                raise ValueError("tags is required")
            
            self._get_tag_embeddings(new_tags)
            
            return {
                "added_tags": new_tags,
                "cache_size": len(self._tag_embeddings_cache),
            }
        
        else:
            raise ValueError(f"Unknown cmd: {cmd}")