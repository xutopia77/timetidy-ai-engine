"""
Progress File Manager

Manages the progress file (process_file.json) for FAISS index building.
Provides:
1. Load progress file
2. Save progress file
3. Update progress stats
4. Get pending files
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class ProgressManager:
    def __init__(self, progress_file_path: str = "config/process_file.json"):
        self.progress_file = Path(progress_file_path)
        
        print(f"ProgressManager initialized with file: {self.progress_file}", file=sys.stderr)
    
    def load_progress(self) -> Dict:
        """Load progress file"""
        if not self.progress_file.exists():
            raise FileNotFoundError(f"Progress file not found: {self.progress_file}")
        
        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
            
            print(f"Progress file loaded: {len(progress_data.get('files', []))} files", file=sys.stderr)
            
            return progress_data
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Progress file format error: {e}")
    
    def save_progress(self, progress_data: Dict):
        """Save progress file"""
        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
            
            print(f"Progress file saved: {progress_data.get('processed_files', 0)}/{progress_data.get('total_files', 0)}", file=sys.stderr)
        
        except Exception as e:
            print(f"Failed to save progress file: {e}", file=sys.stderr)
            raise
    
    def update_progress_stats(self, progress_data: Dict) -> Dict:
        """Update progress statistics"""
        files = progress_data.get("files", [])
        
        processed = sum(1 for f in files if f.get("status") == "completed")
        failed = sum(1 for f in files if f.get("status") == "failed")
        processing = sum(1 for f in files if f.get("status") == "processing")
        idle = sum(1 for f in files if f.get("status") == "idle")
        
        progress_data["processed_files"] = processed
        progress_data["failed_files"] = failed
        progress_data["processing_files"] = processing
        progress_data["idle_files"] = idle
        progress_data["last_update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if progress_data.get("total_files", 0) > 0:
            progress_percent = (processed / progress_data["total_files"]) * 100
            progress_data["progress_percent"] = round(progress_percent, 2)
        
        return progress_data
    
    def get_pending_files(self, progress_data: Dict) -> List[Dict]:
        """Get pending files (status='idle')"""
        files = progress_data.get("files", [])
        pending_files = [f for f in files if f.get("status") == "idle"]
        
        print(f"Pending files: {len(pending_files)}", file=sys.stderr)
        
        return pending_files
    
    def get_processing_files(self, progress_data: Dict) -> List[Dict]:
        """Get processing files (status='processing')"""
        files = progress_data.get("files", [])
        processing_files = [f for f in files if f.get("status") == "processing"]
        
        return processing_files
    
    def mark_file_status(self, progress_data: Dict, record_id: int, status: str) -> Dict:
        """Mark a file's status"""
        files = progress_data.get("files", [])
        
        for file_record in files:
            if file_record.get("record_id") == record_id:
                file_record["status"] = status
                print(f"Marked file {record_id} as {status}", file=sys.stderr)
                break
        
        return progress_data
    
    def get_progress_summary(self, progress_data: Dict) -> Dict:
        """Get progress summary"""
        total_files = progress_data.get("total_files", 0)
        processed_files = progress_data.get("processed_files", 0)
        failed_files = progress_data.get("failed_files", 0)
        progress_percent = progress_data.get("progress_percent", 0.0)
        
        status = "idle"
        if processed_files > 0 and processed_files < total_files:
            status = "processing"
        elif processed_files == total_files:
            status = "completed"
        
        return {
            "total_files": total_files,
            "processed_files": processed_files,
            "failed_files": failed_files,
            "progress_percent": progress_percent,
            "status": status,
            "last_update_time": progress_data.get("last_update_time", ""),
            "start_time": progress_data.get("start_time", "")
        }
    
    def create_progress_file(self, image_records: List[Dict], batch: int = 10) -> Dict:
        """Create a new progress file"""
        files = []
        
        for record in image_records:
            files.append({
                "record_id": record.get("record_id"),
                "image_path": record.get("image_path"),
                "image_hash": record.get("image_hash", ""),
                "status": "idle"
            })
        
        progress_data = {
            "version": "1.0.0",
            "batch": batch,
            "files": files,
            "total_files": len(files),
            "processed_files": 0,
            "failed_files": 0,
            "processing_files": 0,
            "idle_files": len(files),
            "progress_percent": 0.0,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.save_progress(progress_data)
        
        return progress_data
    
    def reset_progress_file(self, progress_data: Dict) -> Dict:
        """Reset progress file (mark all files as idle)"""
        files = progress_data.get("files", [])
        
        for file_record in files:
            file_record["status"] = "idle"
        
        progress_data["processed_files"] = 0
        progress_data["failed_files"] = 0
        progress_data["processing_files"] = 0
        progress_data["idle_files"] = len(files)
        progress_data["progress_percent"] = 0.0
        progress_data["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        progress_data["last_update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.save_progress(progress_data)
        
        return progress_data