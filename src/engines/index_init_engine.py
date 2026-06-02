"""
Index Initialization Engine (File Hash Based)

This engine provides:
1. Index initialization from process_file.json
2. Index rebuilding (rebuild=true)
3. Batch feature extraction with file_hash comparison
4. Progress monitoring (start/stop/continue/get_progress)

Key Design:
- Go generates process_file.json with all file info and file_hash
- Python reads process_file.json and compares file_hash
- Python processes files with changed file_hash
- Python skips files with unchanged file_hash
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
        
        print(f"IndexInitEngine initialized (file_hash based)", file=sys.stderr)
    
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
        
        elif operation == "get_status":
            return self._get_status()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _start_build(self, rebuild: bool) -> dict:
        """
        Start index building
        
        Args:
            rebuild: bool - whether to rebuild index (delete existing)
        
        Returns:
            {
                "success": true,
                "operation": "start",
                "total_files": 10000,
                "processed_files": 0,
                "status": "processing",
                "message": "Index build started"
            }
        """
        print(f"Starting index build (rebuild={rebuild})", file=sys.stderr)
        
        try:
            progress_data = self.progress_manager.load_progress()
            print(f"Loaded progress file: {len(progress_data.get('files', []))} files", file=sys.stderr)
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Progress file not found. Please create progress file first.",
                "hint": "Go should create progress file: ai-engine/src/config/process_file.json"
            }
        
        if rebuild:
            print(f"Deleting existing index...", file=sys.stderr)
            self.index_manager.delete_index()
            progress_data = self.progress_manager.reset_progress_file(progress_data)
            print(f"Reset progress file: {len(progress_data.get('files', []))} files", file=sys.stderr)
        
        summary = self.progress_manager.get_progress_summary(progress_data)
        print(f"Progress summary: total={summary['total_files']}, processed={summary['processed_files']}, status={summary['status']}", file=sys.stderr)
        
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
        """
        Stop index building and save progress
        
        Returns:
            {
                "success": true,
                "operation": "stop",
                "total_files": 10000,
                "processed_files": 5000,
                "status": "stopped",
                "message": "Index build stopped, progress saved"
            }
        """
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
            "skipped_files": summary.get("skipped_files", 0),
            "failed_files": summary["failed_files"],
            "status": "stopped",
            "message": "Index build stopped, progress saved"
        }
    
    def _continue_build(self) -> dict:
        """
        Continue index building from progress file
        
        Returns:
            {
                "success": true,
                "operation": "continue",
                "total_files": 10000,
                "processed_files": 5000,
                "status": "processing",
                "message": "Index build continued"
            }
        """
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
        """
        Get current build progress
        
        Returns:
            {
                "success": true,
                "operation": "get_progress",
                "data": {
                    "total_files": 10000,
                    "processed_files": 5000,
                    "skipped_files": 4000,
                    "failed_files": 10,
                    "progress_percent": 50.0,
                    "status": "processing",
                    "last_update_time": "2024-01-01 12:00:00"
                }
            }
        """
        try:
            progress_data = self.progress_manager.load_progress()
            summary = self.progress_manager.get_progress_summary(progress_data)
            
            if self.is_building:
                summary["status"] = "processing"
            
            print(f"summary:{summary}", file=sys.stderr)
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
        """
        Process files asynchronously with file_hash comparison
        
        Args:
            progress_data: dict - progress file data
        """
        batch_size = progress_data.get("batch", 10)
        files = progress_data.get("files", [])
        
        if not files:
            print(f"ERROR: No files to process!", file=sys.stderr)
            self.is_building = False
            return
        
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        start_time = time.time()
        
        print(f"Processing files (batch_size={batch_size}, total_files={len(files)})", file=sys.stderr)
        
        if not self.index_manager.is_loaded():
            print(f"Index not loaded, creating new index...", file=sys.stderr)
            self.index_manager.create_empty_index()
            print(f"Index created successfully", file=sys.stderr)
        
        metadata = self.index_manager.metadata
        
        for file_record in files:
            if self.stop_flag:
                print(f"Stop flag detected, stopping processing", file=sys.stderr)
                self.progress_manager.save_progress(progress_data)
                self.index_manager.save_index()
                break
            
            if file_record.get("status") != "pending":
                continue
            
            record_id = file_record.get("record_id")
            image_path = file_record.get("image_path")
            file_hash = file_record.get("file_hash", "")
            
            print(f"Processing file {record_id}: {image_path}", file=sys.stderr)
            
            file_record["status"] = "processing"
            
            record_key = str(record_id)
            
            if record_key in metadata:
                existing_hash = metadata[record_key].get("file_hash", "")
                
                if existing_hash == file_hash:
                    file_record["status"] = "completed"
                    skipped_count += 1
                    print(f"File {record_id} skipped (file_hash unchanged)", file=sys.stderr)
                    continue
                else:
                    print(f"File {record_id} needs update (file_hash changed: {existing_hash} -> {file_hash})", file=sys.stderr)
            
            try:
                if record_key in metadata:
                    self.index_manager.remove_vector(record_id)
                
                features = self.clip_encoder.encode_image(image_path)
                features = features.flatten()
                
                self.index_manager.add_vector(features, record_id, file_hash)
                
                file_record["status"] = "completed"
                processed_count += 1
                
                print(f"File {record_id} processed successfully", file=sys.stderr)
                
            except Exception as e:
                file_record["status"] = "failed"
                failed_count += 1
                print(f"Failed to process {image_path}: {e}", file=sys.stderr)
            
            if (processed_count + skipped_count) % batch_size == 0:
                progress_data["processed_files"] = processed_count
                progress_data["skipped_files"] = skipped_count
                progress_data["failed_files"] = failed_count
                self.progress_manager.update_progress_stats(progress_data)
                self.progress_manager.save_progress(progress_data)
                self.index_manager.save_index()
                
                elapsed_time = time.time() - start_time
                total_processed = processed_count + skipped_count + failed_count
                avg_time_per_file = elapsed_time / total_processed if total_processed > 0 else 0
                remaining_files = progress_data.get("total_files", 0) - total_processed
                estimated_time = remaining_files * avg_time_per_file
                
                print(f"Progress: {total_processed}/{progress_data.get('total_files', 0)} files, elapsed={elapsed_time:.1f}s, estimated={estimated_time:.1f}s", file=sys.stderr)
        
        if not self.stop_flag:
            progress_data["processed_files"] = processed_count
            progress_data["skipped_files"] = skipped_count
            progress_data["failed_files"] = failed_count
            self.progress_manager.update_progress_stats(progress_data)
            self.progress_manager.save_progress(progress_data)
            self.index_manager.save_index()
            self.is_building = False
            
            elapsed_time = time.time() - start_time
            total_processed = processed_count + skipped_count
            print(f"Index build completed: {processed_count} processed, {skipped_count} skipped, {failed_count} failed in {elapsed_time:.1f}s", file=sys.stderr)
    
    def _get_status(self) -> dict:
        """Get index init engine status"""
        return {
            "clip_encoder": self.clip_encoder.get_status(),
            "index_manager": self.index_manager.get_status(),
            "is_building": self.is_building
        }