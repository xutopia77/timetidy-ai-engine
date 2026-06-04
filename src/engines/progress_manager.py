"""
Progress File Manager (File Hash Based)

Manages the progress file (process_file.json) for FAISS index building.
Provides:
1. Load progress file
2. Save progress file
3. Update progress stats
4. Get pending files
5. Reset progress file

Progress File Format:
{
  "version": "1.0.0",
  "batch": 10,
  "files": [
    {
      "record_id": 1001,
      "file_path": "/path/to/image.jpg",
      "file_hash": "abc123",  // Go计算的文件hash
      "status": "pending"     // "pending", "processing", "completed", "failed"
    }
  ],
  "start_time": "2024-01-01 10:00:00",
  "last_update_time": "2024-01-01 12:00:00"
}

Status Description:
- pending: 待处理（未构建索引）
- processing: 正在处理
- completed: 已完成（已构建索引）
- failed: 处理失败

Key Design:
- Go generates process_file.json with all files status="pending"
- Python compares file_hash to decide whether to process
- Python updates status during processing
- Statistics (total_files, processed_files, etc.) are calculated in real-time to avoid data inconsistency
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class ProgressManager:
    def __init__(self, progress_file_path: str = None):
        if progress_file_path is None:
            # project_root is ai-engine directory (parent of src)
            project_root = Path(__file__).parent.parent.parent
            self.progress_file = project_root / "ai-data" / "progress" / "process_file.json"
        else:
            self.progress_file = Path(progress_file_path)
        
        print(f"ProgressManager initialized with file: {self.progress_file}", file=sys.stderr)
    
    def load_progress(self) -> Dict:
        """
        Load progress file
        
        Returns:
            dict - progress file data
        
        Raises:
            FileNotFoundError - if progress file not found
            ValueError - if progress file format error
        """
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
        """
        Save progress file
        
        Args:
            progress_data: dict - progress file data
        """
        files = progress_data.get("files", [])
        total_files = len(files)
        processed_files = sum(1 for f in files if f.get("status") == "completed")
        failed_files = sum(1 for f in files if f.get("status") == "failed")
        pending_files = sum(1 for f in files if f.get("status") == "pending")
        
        try:
            progress_data["last_update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
            
            print(f"Progress file saved: {processed_files} processed, {pending_files} pending, {total_files} total", file=sys.stderr)
        
        except Exception as e:
            print(f"Failed to save progress file: {e}", file=sys.stderr)
            raise
    
    def calculate_stats(self, progress_data: Dict) -> Dict:
        """
        Calculate statistics from files list
        
        Args:
            progress_data: dict - progress file data
        
        Returns:
            dict - calculated statistics
        """
        files = progress_data.get("files", [])
        
        total_files = len(files)
        processed_files = sum(1 for f in files if f.get("status") == "completed")
        failed_files = sum(1 for f in files if f.get("status") == "failed")
        processing_files = sum(1 for f in files if f.get("status") == "processing")
        pending_files = sum(1 for f in files if f.get("status") == "pending")
        
        progress_percent = 0.0
        if total_files > 0:
            progress_percent = round((processed_files / total_files) * 100, 2)
        
        return {
            "total_files": total_files,
            "processed_files": processed_files,
            "failed_files": failed_files,
            "processing_files": processing_files,
            "pending_files": pending_files,
            "progress_percent": progress_percent
        }
    
    def update_progress_stats(self, progress_data: Dict) -> Dict:
        """
        Update progress statistics (deprecated, use calculate_stats instead)
        
        Args:
            progress_data: dict - progress file data
        
        Returns:
            dict - progress data (unchanged, stats are calculated in real-time)
        """
        return progress_data
    
    def get_pending_files(self, progress_data: Dict) -> List[Dict]:
        """
        Get pending files (status='pending')
        
        Args:
            progress_data: dict - progress file data
        
        Returns:
            list - pending files
        """
        files = progress_data.get("files", [])
        pending_files = [f for f in files if f.get("status") == "pending"]
        
        print(f"Pending files: {len(pending_files)}", file=sys.stderr)
        
        return pending_files
    
    def get_processing_files(self, progress_data: Dict) -> List[Dict]:
        """
        Get processing files (status='processing')
        
        Args:
            progress_data: dict - progress file data
        
        Returns:
            list - processing files
        """
        files = progress_data.get("files", [])
        processing_files = [f for f in files if f.get("status") == "processing"]
        
        return processing_files
    
    def mark_file_status(self, progress_data: Dict, record_id: int, status: str) -> Dict:
        """
        Mark a file's status
        
        Args:
            progress_data: dict - progress file data
            record_id: int - record ID
            status: str - new status
        
        Returns:
            dict - updated progress data
        """
        files = progress_data.get("files", [])
        
        for file_record in files:
            if file_record.get("record_id") == record_id:
                file_record["status"] = status
                print(f"Marked file {record_id} as {status}", file=sys.stderr)
                break
        
        return progress_data
    
    def get_progress_summary(self, progress_data: Dict) -> Dict:
        """
        Get progress summary
        
        Args:
            progress_data: dict - progress file data
        
        Returns:
            dict - progress summary
        """
        stats = self.calculate_stats(progress_data)
        
        total_files = stats["total_files"]
        processed_files = stats["processed_files"]
        failed_files = stats["failed_files"]
        pending_files = stats["pending_files"]
        progress_percent = stats["progress_percent"]
        
        status = "pending"
        if processed_files > 0 and processed_files < total_files:
            status = "processing"
        elif processed_files == total_files:
            status = "completed"
        
        return {
            "total_files": total_files,
            "processed_files": processed_files,
            "skipped_files": 0,
            "failed_files": failed_files,
            "pending_files": pending_files,
            "progress_percent": progress_percent,
            "status": status,
            "last_update_time": progress_data.get("last_update_time", ""),
            "start_time": progress_data.get("start_time", "")
        }
    
    def create_progress_file(self, image_records: List[Dict], batch: int = 10) -> Dict:
        """
        Create a new progress file
        
        Args:
            image_records: list - image records with record_id, file_path, file_hash
            batch: int - batch size
        
        Returns:
            dict - progress file data
        """
        files = []
        
        for record in image_records:
            files.append({
                "record_id": record.get("record_id"),
                "file_path": record.get("file_path"),
                "file_hash": record.get("file_hash", ""),
                "status": "pending"
            })
        
        progress_data = {
            "version": "1.0.0",
            "batch": batch,
            "files": files,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.save_progress(progress_data)
        
        return progress_data
    
    def reset_progress_file(self, progress_data: Dict) -> Dict:
        """
        Reset progress file (mark all files as pending)
        
        Args:
            progress_data: dict - progress file data
        
        Returns:
            dict - reset progress data
        """
        files = progress_data.get("files", [])
        
        for file_record in files:
            file_record["status"] = "pending"
        
        progress_data["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        progress_data["last_update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.save_progress(progress_data)
        
        return progress_data