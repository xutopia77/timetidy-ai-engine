"""
Data Sync Engine (File Hash Based)

This engine provides:
1. Batch synchronization based on file_hash comparison
2. Add new files (file_hash not in metadata)
3. Update modified files (file_hash changed)
4. Skip unchanged files (file_hash same)

Key Design:
- Go maintains all data (SQLite, file system, file_hash)
- Python only handles AI features (feature extraction, vector search)
- process_file.json contains all file info with file_hash
- Python compares file_hash to decide whether to update index
"""

import os
import sys
import json
import time
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
        
        print(f"SyncEngine initialized (file_hash based)", file=sys.stderr)
    
    def process(self, params: dict) -> dict:
        operation = params.get("operation", "")
        
        if operation == "sync_all":
            return self._sync_all(params)
        
        elif operation == "get_status":
            return self._get_status()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _sync_all(self, params: dict) -> dict:
        """
        Synchronize all files based on file_hash comparison
        
        Args:
            params: {
                "progress_file": "path/to/process_file.json" (optional)
            }
        
        Returns:
            {
                "success": true,
                "total_files": 10000,
                "processed_files": 50,
                "skipped_files": 9950,
                "failed_files": 0,
                "process_time_ms": 1234.56
            }
        """
        print(f"Starting sync_all operation...", file=sys.stderr)
        
        progress_file_path = params.get("progress_file", "data/progress/process_file.json")
        progress_file = Path(progress_file_path)
        
        if not progress_file.exists():
            raise FileNotFoundError(f"Progress file not found: {progress_file}. Go should create progress file with file_hash info")
        
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Progress file format error: {e}")
        
        files = progress_data.get("files", [])
        
        if not files:
            return {
                "total_files": 0,
                "processed_files": 0,
                "skipped_files": 0,
                "failed_files": 0,
                "message": "No files to sync"
            }
        
        self._ensure_index_loaded()
        
        metadata = self.index_manager.metadata
        
        start_time = time.time()
        
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        
        for file_record in files:
            record_id = file_record.get("record_id")
            file_path = file_record.get("file_path")
            file_hash = file_record.get("file_hash", "")
            
            if record_id is None or file_path is None:
                failed_count += 1
                print(f"Invalid file record: {file_record}", file=sys.stderr)
                continue
            
            record_key = str(record_id)
            
            if record_key in metadata:
                existing_hash = metadata[record_key].get("file_hash", "")
                
                if existing_hash == file_hash:
                    skipped_count += 1
                    continue
                else:
                    print(f"Updating file {record_id} (hash changed: {existing_hash} -> {file_hash})", file=sys.stderr)
                    
                    try:
                        self.index_manager.remove_vector(record_id)
                        
                        features = self.clip_encoder.encode_image(file_path)
                        
                        success = self.index_manager.add_vector(features, record_id, file_hash)
                        
                        if success:
                            processed_count += 1
                        else:
                            failed_count += 1
                            print(f"Failed to update file {record_id}", file=sys.stderr)
                    
                    except Exception as e:
                        failed_count += 1
                        print(f"Error updating file {record_id}: {e}", file=sys.stderr)
            
            else:
                print(f"Adding new file {record_id}", file=sys.stderr)
                
                try:
                    features = self.clip_encoder.encode_image(file_path)
                    
                    success = self.index_manager.add_vector(features, record_id, file_hash)
                    
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
                        print(f"Failed to add file {record_id}", file=sys.stderr)
                
                except Exception as e:
                    failed_count += 1
                    print(f"Error adding file {record_id}: {e}", file=sys.stderr)
        
        self.index_manager.save_index()
        
        process_time_ms = (time.time() - start_time) * 1000
        
        total_files = len(files)
        
        print(f"Sync completed: {processed_count} processed, {skipped_count} skipped, {failed_count} failed", file=sys.stderr)
        
        return {
            "operation": "sync_all",
            "total_files": total_files,
            "processed_files": processed_count,
            "skipped_files": skipped_count,
            "failed_files": failed_count,
            "process_time_ms": round(process_time_ms, 2),
            "message": f"Sync completed: {processed_count} processed, {skipped_count} skipped, {failed_count} failed"
        }
    
    def _ensure_index_loaded(self):
        """Ensure FAISS index is loaded"""
        if not self.index_manager.is_loaded():
            if self.index_manager.index_file.exists():
                self.index_manager.load_index()
            else:
                raise RuntimeError(
                    "FAISS index not found. Please initialize index first:\n"
                    "  action: 'index_init'\n"
                    "  params: {'operation': 'start', 'rebuild': false}"
                )
    
    def _get_status(self) -> dict:
        """Get sync engine status"""
        return {
            "clip_encoder": self.clip_encoder.get_status(),
            "index_manager": self.index_manager.get_status()
        }