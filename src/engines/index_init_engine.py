"""
Index Initialization Engine

This engine provides:
1. Index initialization from existing images
2. Index rebuilding
3. Batch feature extraction
4. Progress file management (start/stop/continue)
"""

import os
import sys
import json
import time
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from engines.base import BaseEngine
from engines.clip_encoder import CLIPEncoder
from engines.faiss_index_manager import FAISSIndexManager
from engines.progress_manager import ProgressManager


class IndexInitEngine(BaseEngine):
    def __init__(self):
        self.clip_encoder = CLIPEncoder()
        self.index_manager = FAISSIndexManager()
        self.progress_manager = ProgressManager()
        
        self.is_building = False
        self.stop_flag = False
        self.build_thread = None
        
        print(f"IndexInitEngine initialized", file=sys.stderr)
    
    def process(self, params: dict) -> dict:
        operation = params.get("operation", "")
        
        if operation == "start":
            rebuild = params.get("rebuild", False)
            return self._start_build(rebuild)
        
        elif operation == "stop":
            return self._stop_build()
        
        elif operation == "continue":
            return self._continue_build()
        
        elif operation == "get_progress":
            return self._get_progress()
        
        elif operation == "init":
            return self._init_index(params)
        
        elif operation == "rebuild":
            return self._rebuild_index(params)
        
        elif operation == "delete":
            return self._delete_index(params)
        
        elif operation == "get_status":
            return self._get_status()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _start_build(self, rebuild: bool) -> dict:
        print(f"Starting index build (rebuild={rebuild})", file=sys.stderr)
        
        if rebuild:
            print(f"Deleting existing index...", file=sys.stderr)
            self.index_manager.delete_index()
            self.progress_manager.reset_progress_file(self.progress_manager.load_progress())
        
        try:
            progress_data = self.progress_manager.load_progress()
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Progress file not found. Please create progress file first.",
                "hint": "Go should create progress file: ai-engine/src/config/process_file.json"
            }
        
        summary = self.progress_manager.get_progress_summary(progress_data)
        
        if summary["status"] == "completed":
            return {
                "success": True,
                "operation": "start",
                "status": "completed",
                "message": "Index already completed, no need to start"
            }
        
        self.is_building = True
        self.stop_flag = False
        
        self.build_thread = threading.Thread(
            target=self._process_files_async,
            args=(progress_data,)
        )
        self.build_thread.start()
        
        return {
            "success": True,
            "operation": "start",
            "total_files": summary["total_files"],
            "processed_files": summary["processed_files"],
            "status": "processing",
            "message": "Index build started"
        }
    
    def _stop_build(self) -> dict:
        print(f"Stopping index build...", file=sys.stderr)
        
        self.stop_flag = True
        self.is_building = False
        
        if self.build_thread and self.build_thread.is_alive():
            self.build_thread.join(timeout=5.0)
        
        progress_data = self.progress_manager.load_progress()
        self.progress_manager.update_progress_stats(progress_data)
        self.progress_manager.save_progress(progress_data)
        self.index_manager.save_index()
        
        summary = self.progress_manager.get_progress_summary(progress_data)
        
        return {
            "success": True,
            "operation": "stop",
            "total_files": summary["total_files"],
            "processed_files": summary["processed_files"],
            "status": "stopped",
            "message": "Index build stopped, progress saved"
        }
    
    def _continue_build(self) -> dict:
        print(f"Continuing index build...", file=sys.stderr)
        
        try:
            progress_data = self.progress_manager.load_progress()
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Progress file not found. Cannot continue build.",
                "hint": "Go should create progress file first"
            }
        
        summary = self.progress_manager.get_progress_summary(progress_data)
        
        if summary["status"] == "completed":
            return {
                "success": True,
                "operation": "continue",
                "status": "completed",
                "message": "Index already completed"
            }
        
        self.is_building = True
        self.stop_flag = False
        
        self.build_thread = threading.Thread(
            target=self._process_files_async,
            args=(progress_data,)
        )
        self.build_thread.start()
        
        return {
            "success": True,
            "operation": "continue",
            "total_files": summary["total_files"],
            "processed_files": summary["processed_files"],
            "status": "processing",
            "message": "Index build continued"
        }
    
    def _get_progress(self) -> dict:
        try:
            progress_data = self.progress_manager.load_progress()
            summary = self.progress_manager.get_progress_summary(progress_data)
            
            if self.is_building:
                summary["status"] = "processing"
            
            return {
                "success": True,
                "operation": "get_progress",
                "data": summary
            }
        
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Progress file not found"
            }
    
    def _process_files_async(self, progress_data: dict):
        batch_size = progress_data.get("batch", 10)
        files = progress_data.get("files", [])
        
        processed_count = 0
        start_time = time.time()
        
        print(f"Processing files (batch_size={batch_size})", file=sys.stderr)
        
        for file_record in files:
            if self.stop_flag:
                print(f"Stop flag detected, stopping processing", file=sys.stderr)
                self.progress_manager.update_progress_stats(progress_data)
                self.progress_manager.save_progress(progress_data)
                self.index_manager.save_index()
                break
            
            if file_record.get("status") != "idle":
                continue
            
            record_id = file_record.get("record_id")
            image_path = file_record.get("image_path")
            image_hash = file_record.get("image_hash", "")
            
            print(f"Processing file {record_id}: {image_path}", file=sys.stderr)
            
            file_record["status"] = "processing"
            
            try:
                features = self.clip_encoder.encode_image(image_path)
                features = features.flatten()
                
                self.index_manager.add_vector(features, record_id, image_hash)
                
                file_record["status"] = "completed"
                processed_count += 1
                
                print(f"File {record_id} processed successfully", file=sys.stderr)
                
            except Exception as e:
                file_record["status"] = "failed"
                print(f"Failed to process {image_path}: {e}", file=sys.stderr)
            
            if processed_count % batch_size == 0:
                self.progress_manager.update_progress_stats(progress_data)
                self.progress_manager.save_progress(progress_data)
                self.index_manager.save_index()
                
                elapsed_time = time.time() - start_time
                avg_time_per_file = elapsed_time / processed_count
                remaining_files = progress_data.get("total_files", 0) - processed_count
                estimated_time = remaining_files * avg_time_per_file
                
                print(f"Progress: {processed_count}/{progress_data.get('total_files', 0)} files, elapsed={elapsed_time:.1f}s, estimated={estimated_time:.1f}s", file=sys.stderr)
        
        if not self.stop_flag:
            self.progress_manager.update_progress_stats(progress_data)
            self.progress_manager.save_progress(progress_data)
            self.index_manager.save_index()
            self.is_building = False
            
            elapsed_time = time.time() - start_time
            print(f"Index build completed: {processed_count} files in {elapsed_time:.1f}s", file=sys.stderr)
    
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