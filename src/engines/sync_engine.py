"""
Data Sync Engine

This engine provides:
1. Event-driven data synchronization
2. Image addition, deletion, modification handling
3. Batch import support

Event types:
- IMAGE_ADDED: Add new image to index
- IMAGE_DELETED: Remove image from index
- IMAGE_MOVED: Update image path (no feature change)
- IMAGE_MODIFIED: Update image features
- BATCH_IMPORTED: Batch add images
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List

from engines.base import BaseEngine
from engines.clip_encoder import CLIPEncoder
from engines.faiss_index_manager import FAISSIndexManager


class SyncEngine(BaseEngine):
    def __init__(self):
        self.clip_encoder = CLIPEncoder()
        self.index_manager = FAISSIndexManager()
        
        print(f"SyncEngine initialized:", file=sys.stderr)
    
    def process(self, params: dict) -> dict:
        event_type = params.get("event_type", "")
        
        if event_type == "IMAGE_ADDED":
            return self._handle_image_added(params)
        
        elif event_type == "IMAGE_DELETED":
            return self._handle_image_deleted(params)
        
        elif event_type == "IMAGE_MOVED":
            return self._handle_image_moved(params)
        
        elif event_type == "IMAGE_MODIFIED":
            return self._handle_image_modified(params)
        
        elif event_type == "BATCH_IMPORTED":
            return self._handle_batch_imported(params)
        
        elif event_type == "get_status":
            return self._get_status()
        
        else:
            raise ValueError(f"Unknown event_type: {event_type}")
    
    def _handle_image_added(self, params: dict) -> dict:
        self._ensure_index_loaded()
        
        record_id = params.get("record_id")
        image_path = params.get("image_path")
        image_hash = params.get("image_hash", "")
        
        if record_id is None or image_path is None:
            raise ValueError("record_id and image_path are required")
        
        start_time = time.time()
        
        features = self.clip_encoder.encode_image(image_path)
        
        success = self.index_manager.add_vector(features, record_id, image_hash)
        
        process_time_ms = (time.time() - start_time) * 1000
        
        if success:
            return {
                "success": True,
                "event_type": "IMAGE_ADDED",
                "record_id": record_id,
                "vector_id": record_id,
                "process_time_ms": round(process_time_ms, 2),
                "message": "Image added to index successfully"
            }
        else:
            return {
                "success": False,
                "event_type": "IMAGE_ADDED",
                "record_id": record_id,
                "error": "Failed to add image to index"
            }
    
    def _handle_image_deleted(self, params: dict) -> dict:
        self._ensure_index_loaded()
        
        record_id = params.get("record_id")
        
        if record_id is None:
            raise ValueError("record_id is required")
        
        start_time = time.time()
        
        success = self.index_manager.remove_vector(record_id)
        
        process_time_ms = (time.time() - start_time) * 1000
        
        if success:
            return {
                "success": True,
                "event_type": "IMAGE_DELETED",
                "record_id": record_id,
                "process_time_ms": round(process_time_ms, 2),
                "message": "Image removed from index successfully"
            }
        else:
            return {
                "success": False,
                "event_type": "IMAGE_DELETED",
                "record_id": record_id,
                "error": "Failed to remove image from index"
            }
    
    def _handle_image_moved(self, params: dict) -> dict:
        self._ensure_index_loaded()
        
        record_id = params.get("record_id")
        new_path = params.get("new_path")
        old_path = params.get("old_path")
        
        if record_id is None:
            raise ValueError("record_id is required")
        
        return {
            "success": True,
            "event_type": "IMAGE_MOVED",
            "record_id": record_id,
            "old_path": old_path,
            "new_path": new_path,
            "message": "Image moved, no feature change required"
        }
    
    def _handle_image_modified(self, params: dict) -> dict:
        self._ensure_index_loaded()
        
        record_id = params.get("record_id")
        image_path = params.get("image_path")
        new_image_hash = params.get("image_hash", "")
        
        if record_id is None or image_path is None:
            raise ValueError("record_id and image_path are required")
        
        start_time = time.time()
        
        self.index_manager.remove_vector(record_id)
        
        features = self.clip_encoder.encode_image(image_path)
        
        success = self.index_manager.add_vector(features, record_id, new_image_hash)
        
        process_time_ms = (time.time() - start_time) * 1000
        
        if success:
            return {
                "success": True,
                "event_type": "IMAGE_MODIFIED",
                "record_id": record_id,
                "vector_id": record_id,
                "process_time_ms": round(process_time_ms, 2),
                "message": "Image features updated successfully"
            }
        else:
            return {
                "success": False,
                "event_type": "IMAGE_MODIFIED",
                "record_id": record_id,
                "error": "Failed to update image features"
            }
    
    def _handle_batch_imported(self, params: dict) -> dict:
        self._ensure_index_loaded()
        
        records = params.get("records", [])
        
        if not records:
            raise ValueError("records list is required")
        
        start_time = time.time()
        
        success_count = 0
        fail_count = 0
        
        for record in records:
            record_id = record.get("record_id")
            image_path = record.get("image_path")
            image_hash = record.get("image_hash", "")
            
            if record_id is None or image_path is None:
                fail_count += 1
                continue
            
            try:
                features = self.clip_encoder.encode_image(image_path)
                success = self.index_manager.add_vector(features, record_id, image_hash)
                
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    
            except Exception as e:
                print(f"Failed to add image {image_path}: {e}", file=sys.stderr)
                fail_count += 1
        
        process_time_ms = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "event_type": "BATCH_IMPORTED",
            "total_count": len(records),
            "success_count": success_count,
            "fail_count": fail_count,
            "process_time_ms": round(process_time_ms, 2),
            "message": f"Batch import completed: {success_count} success, {fail_count} failed"
        }
    
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
    
    def _get_status(self) -> dict:
        return {
            "clip_encoder": self.clip_encoder.get_status(),
            "index_manager": self.index_manager.get_status()
        }