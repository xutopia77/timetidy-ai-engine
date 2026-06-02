"""
Index Initialization Engine

This engine provides:
1. Index initialization from existing images
2. Index rebuilding
3. Batch feature extraction
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict

from engines.base import BaseEngine
from engines.clip_encoder import CLIPEncoder
from engines.faiss_index_manager import FAISSIndexManager


class IndexInitEngine(BaseEngine):
    def __init__(self):
        self.clip_encoder = CLIPEncoder()
        self.index_manager = FAISSIndexManager()
        
        print(f"IndexInitEngine initialized:", file=sys.stderr)
    
    def process(self, params: dict) -> dict:
        action = params.get("action", "init")
        
        if action == "init":
            return self._init_index(params)
        
        elif action == "rebuild":
            return self._rebuild_index(params)
        
        elif action == "delete":
            return self._delete_index(params)
        
        elif action == "get_status":
            return self._get_status()
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _init_index(self, params: dict) -> dict:
        rebuild = params.get("rebuild", False)
        
        if not rebuild and self.index_manager.index_file.exists():
            print(f"Index file exists, loading existing index...", file=sys.stderr)
            if self.index_manager.load_index():
                return {
                    "success": True,
                    "action": "load_existing",
                    "total_images": self.index_manager.index.ntotal,
                    "message": "Existing index loaded successfully"
                }
        
        image_records = params.get("image_records", [])
        
        if not image_records:
            return {
                "success": False,
                "error": "No image_records provided. Please provide list of {record_id, image_path, image_hash}",
                "hint": "Example: [{'record_id': 1, 'image_path': '/path/to/image.jpg', 'image_hash': 'abc123'}]"
            }
        
        return self._build_index_from_records(image_records)
    
    def _rebuild_index(self, params: dict) -> dict:
        image_records = params.get("image_records", [])
        
        if not image_records:
            return {
                "success": False,
                "error": "No image_records provided for rebuild"
            }
        
        return self._build_index_from_records(image_records)
    
    def _delete_index(self, params: dict) -> dict:
        print(f"Deleting index files...", file=sys.stderr)
        
        deleted_files = []
        
        if self.index_manager.index_file.exists():
            try:
                os.remove(str(self.index_manager.index_file))
                deleted_files.append(str(self.index_manager.index_file))
                print(f"  - Deleted: {self.index_manager.index_file}", file=sys.stderr)
            except Exception as e:
                print(f"  - Failed to delete {self.index_manager.index_file}: {e}", file=sys.stderr)
        
        if self.index_manager.metadata_file.exists():
            try:
                os.remove(str(self.index_manager.metadata_file))
                deleted_files.append(str(self.index_manager.metadata_file))
                print(f"  - Deleted: {self.index_manager.metadata_file}", file=sys.stderr)
            except Exception as e:
                print(f"  - Failed to delete {self.index_manager.metadata_file}: {e}", file=sys.stderr)
        
        if self.index_manager.config_file.exists():
            try:
                os.remove(str(self.index_manager.config_file))
                deleted_files.append(str(self.index_manager.config_file))
                print(f"  - Deleted: {self.index_manager.config_file}", file=sys.stderr)
            except Exception as e:
                print(f"  - Failed to delete {self.index_manager.config_file}: {e}", file=sys.stderr)
        
        self.index_manager.index = None
        self.index_manager.metadata = {}
        
        return {
            "success": True,
            "action": "delete",
            "deleted_files": deleted_files,
            "message": f"Index deleted successfully, {len(deleted_files)} files removed"
        }
    
    def _build_index_from_records(self, image_records: List[Dict]) -> dict:
        start_time = time.time()
        
        print(f"Building index from {len(image_records)} image records...", file=sys.stderr)
        
        features_list = []
        record_ids = []
        image_hashes = []
        failed_records = []
        
        batch_size = 32
        total_batches = (len(image_records) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_records))
            batch_records = image_records[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_records)} images)...", file=sys.stderr)
            
            for record in batch_records:
                record_id = record.get("record_id")
                image_path = record.get("image_path")
                image_hash = record.get("image_hash", "")
                
                if record_id is None or image_path is None:
                    failed_records.append(record)
                    continue
                
                try:
                    features = self.clip_encoder.encode_image(image_path)
                    features_list.append(features.flatten())
                    record_ids.append(record_id)
                    image_hashes.append(image_hash)
                    
                except Exception as e:
                    print(f"Failed to encode image {image_path}: {e}", file=sys.stderr)
                    failed_records.append(record)
        
        if not features_list:
            return {
                "success": False,
                "error": "No features extracted successfully",
                "failed_records": len(failed_records)
            }
        
        features_array = np.array(features_list, dtype=np.float32)
        
        success = self.index_manager.build_index(features_array, record_ids, image_hashes)
        
        build_time_seconds = time.time() - start_time
        
        if success:
            return {
                "success": True,
                "action": "build_new",
                "total_images": len(features_list),
                "failed_records": len(failed_records),
                "build_time_seconds": round(build_time_seconds, 2),
                "index_file": str(self.index_manager.index_file),
                "metadata_file": str(self.index_manager.metadata_file),
                "message": "Index built successfully"
            }
        else:
            return {
                "success": False,
                "error": "Failed to build FAISS index",
                "total_images": len(features_list)
            }
    
    def _get_status(self) -> dict:
        return {
            "clip_encoder": self.clip_encoder.get_status(),
            "index_manager": self.index_manager.get_status()
        }