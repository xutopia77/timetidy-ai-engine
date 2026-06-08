"""
Test AI Engine stdio communication (File Hash Based)

This script simulates Go behavior and tests:
1. Ping communication
2. Index initialization (FAISS) with process_file.json
3. Progress monitoring (start/stop/get_progress)
4. Text-to-image search (FAISS)
5. Tag detection
6. Data synchronization (sync_all with file_hash comparison)
7. Generates markdown report with image links

Key Design:
- Go creates process_file.json with file_hash
- Python reads process_file.json and compares file_hash
- Python processes files with changed file_hash
- Python skips files with unchanged file_hash
"""

import sys
import os
import json
import argparse
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add src directory to Python path for utils module
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from python_process import PythonProcess
from report_generator import generate_markdown_report
from utils.logger import log, set_log_file


def find_image_files(root_dir: str, extensions: List[str] = None) -> List[str]:
    """Find all image files in directory"""
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"]
    
    image_files = []
    root_path = Path(root_dir)
    
    for ext in extensions:
        for file_path in root_path.rglob(f"*{ext}"):
            if file_path.is_file():
                image_files.append(str(file_path))
    
    return sorted(image_files)


def compute_file_hash(file_path: str) -> str:
    """Compute file hash (simulate Go behavior)"""
    return hashlib.md5(file_path.encode()).hexdigest()


def create_process_file(image_files: List[str], start_id: int = 1001, output_path: str = None) -> tuple:
    """
    Create process_file.json (simulate Go behavior)
    
    Args:
        image_files: list of image file paths
        start_id: starting record ID
        output_path: output file path
    
    Returns:
        tuple: (process_file_data, record_id_to_path)
    """
    files = []
    record_id_to_path = {}
    
    for i, image_path in enumerate(image_files):
        record_id = start_id + i
        file_hash = compute_file_hash(image_path)
        
        files.append({
            "record_id": record_id,
            "file_path": image_path,
            "file_hash": file_hash,
            "status": "pending"
        })
        
        record_id_to_path[record_id] = image_path
    
    process_file_data = {
        "version": "1.0.0",
        "batch": 10,
        "files": files,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(process_file_data, f, indent=2, ensure_ascii=False)
        log.info(f"✓ Process file created: {output_path}")
    
    return process_file_data, record_id_to_path


def main():
    parser = argparse.ArgumentParser(
        description="Test AI Engine stdio communication (File Hash Based)"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=r"d:\work\56-time-tidy-prj\test-data\data-min-real",
        help="Test data directory path"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=30,
        help="Maximum number of images to test"
    )
    parser.add_argument(
        "--search-query",
        type=str,
        default="一张纯风景的照片",
        help="Search query text"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results"
    )
    
    args = parser.parse_args()
    
    # Setup log file
    log_dir = project_root / "test_results"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-test.log"
    set_log_file(str(log_file))
    
    # Record start time
    start_time = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log.info("\n" + "="*80)
    log.info("AI Engine stdio Communication Test (File Hash Based)")
    log.info("="*80)
    # ===test setup 
    test_data_dir = args.test_dir
    if not os.path.exists(test_data_dir):
        log.info(f"\n✗ Test data directory not found: {test_data_dir}")
        return
    log.info(f"\nTest directory: {test_data_dir}")
    
    image_files = find_image_files(test_data_dir)
    if not image_files:
        log.info(f"\n✗ No image files found in: {test_data_dir}")
        return
    total_found = len(image_files)
    image_files = image_files[:args.max_images]
    log.info(f"Found {total_found} images, using {len(image_files)} for testing")
    # === start python process
    process = PythonProcess()
    if not process.start():
        log.info("\n✗ Failed to start Python process")
        return
    test_results = {}
    log.info("\n" + "-"*80)
    log.info("Test 1: Ping Communication")
    log.info("-"*80)
    # === test ping 
    ping_response = process.send_request("ping", {})
    
    if ping_response and ping_response.get("code") == 0:
        log.info("✓ Ping successful")
        test_results["ping"] = {
            "code": 0,
            "status": "success",
            "message": ping_response.get("data", {}).get("message", "pong")
        }
    else:
        log.info("✗ Ping failed")
        test_results["ping"] = {
            "code": 1,
            "status": str(ping_response) if ping_response else "No response",
        }
    
    # === test cfg_set
    log.info("\n" + "-"*80)
    log.info("Test 2: Set Runtime Configuration")
    log.info("-"*80)
    
    # Prepare config
    config_data = {
        "progress_file": str(project_root / "ai-data" / "progress" / "process_file.json"),
        "index_dir": str(project_root / "ai-data" / "index"),
        "tag_config_file": str(project_root / "src" / "config" / "hierarchical_tags.json"),
        "prompt_cache_dir": str(project_root / "ai-data" / "prompt_cache"),
        "model_dir": str(project_root / "src" / "models"),
        "clip_model_name": "ViT-H-14",
        "log_level": "INFO",
        "log_file": str(log_file),
    }
    
    log.info(f"Sending cfg_set with config:")
    log.info(f"  - progress_file: {config_data['progress_file']}")
    log.info(f"  - index_dir: {config_data['index_dir']}")
    log.info(f"  - tag_config_file: {config_data['tag_config_file']}")
    log.info(f"  - prompt_cache_dir: {config_data['prompt_cache_dir']}")
    log.info(f"  - model_dir: {config_data['model_dir']}")
    log.info(f"  - clip_model_name: {config_data['clip_model_name']}")
    
    cfg_response = process.send_request("cfg_set", config_data)
    
    if cfg_response and cfg_response.get("code") == 0:
        log.info("✓ Configuration set successfully")
        test_results["cfg_set"] = {
            "code": 0,
            "status": "success",
            "message": cfg_response.get("data", {}).get("message", "")
        }
    else:
        log.info("✗ Configuration set failed")
        test_results["cfg_set"] = {
            "code": 1,
            "status": str(cfg_response) if cfg_response else "No response",
        }
    
    # test index init  
    log.info("\n" + "-"*80)
    log.info("Test 3: Index Initialization (File Hash Based)")
    log.info("-"*80)
    log.info("Step 1: Simulate Go creating process_file.json...")
    
    progress_dir = project_root / "ai-data" / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    process_file_path = progress_dir / "process_file.json"
    process_file_data, record_id_to_path = create_process_file(
        image_files,
        start_id=1001,
        output_path=str(process_file_path)
    )
    log.info(f"  - Created process_file.json with {len(image_files)} files,All files status: pending")
    log.info("\nStep 2: Send start request to Python...")
    
    start_params = {
        "cmd": "start",
        "rebuild": True
    }
    start_response = process.send_request("index_init", start_params, timeout=300.0)
    if start_response and start_response.get("code") == 0:
        start_data = start_response.get("data", {})
        log.info(f"✓ Index build started, Total files: {start_data.get('total_files', 0)}, Status: {start_data.get('status', '')}")
        log.info("\nStep 3: Monitor build progress...")
        
        max_wait_time = 300
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            progress_params = {
                "cmd": "get_progress"
            }
            
            progress_response = process.send_request("index_init", progress_params)
            if progress_response and progress_response.get("code") == 0:
                progress_data = progress_response.get("data", {})
                
                status = progress_data.get("status", "")
                processed = progress_data.get("processed_files", 0)
                skipped = progress_data.get("skipped_files", 0)
                total = progress_data.get("total_files", 0)
                percent = progress_data.get("progress_percent", 0.0)
                log.info(f"Progress: {processed + skipped}/{total} ({percent:.1f}%), Processed: {processed}, Skipped: {skipped}")
                if status == "completed":
                    log.info(f"✓ Index build completed")
                    
                    test_results["index_init"] = {
                        "code": 0,
                        "status": "success",
                        "data": {
                            "total_images": total,
                            "processed_files": processed,
                            "skipped_files": progress_data.get("skipped_files", 0),
                            "failed_files": progress_data.get("failed_files", 0),
                            "build_time_seconds": round(time.time() - start_time, 2),
                            "index_file": str(project_root / "ai-data" / "index" / "faiss_index.bin"),
                            "metadata_file": str(project_root / "ai-data" / "index" / "ai_metadata.json")
                        },
                        "record_id_to_path": record_id_to_path
                    }
                    break
                time.sleep(2)
            else:
                log.info("✗ Failed to get progress")
                break
        
        if time.time() - start_time >= max_wait_time:
            log.info("✗ Index build timeout")
            test_results["index_init"] = {
                "code": 1,
                "status": "Index build timeout"
            }
    else:
        log.info("✗ Index initialization failed")
        error = start_response.get("status", "Unknown error") if start_response else "No response"
        log.info(f"  - Error: {error}")
        
        test_results["index_init"] = {
            "code": 1,
            "status": error
        }
    
    # === test search
    log.info("\n" + "-"*80)
    log.info("Test 4: Text-to-Image Search")
    log.info("-"*80)
    if test_results.get("index_init", {}).get("code") == 0:
        search_params = {
            "cmd": "search",
            "query": args.search_query,
            "top_k": 20
        }
        log.info(f"Query: \"{args.search_query}\", Top-K: 20, Searching in indexed images...")
        search_response = process.send_request("search", search_params, timeout=30.0)
        if search_response and search_response.get("code") == 0:
            search_data = search_response.get("data", {})
            results = search_data.get("results", [])
            query_time = search_data.get("query_time_ms", 0)
            
            record_id_to_path = test_results.get("index_init", {}).get("record_id_to_path", {})
            
            log.info(f"✓ Search successful, found {len(results)} results, Query time: {query_time}ms")
            
            for i, result in enumerate(results, 1):
                record_id = result.get("record_id")
                similarity = result.get("similarity")
                rank = result.get("rank")
                image_path = record_id_to_path.get(record_id, "")
                image_name = Path(image_path).name if image_path else f"ID_{record_id}"
                log.info(f"    {i}. rank: {rank}, similarity: {similarity:.4f}, ID: {record_id}, {image_name}")
            
            test_results["search"] = {
                "code": 0,
                "status": "success",
                "query": args.search_query,
                "query_time_ms": query_time,
                "results": results,
                "record_id_to_path": record_id_to_path
            }
        else:
            log.info("✗ Search failed")
            error = search_response.get("status", "Unknown error") if search_response else "No response"
            log.info(f"  - Error: {error}")
            test_results["search"] = {
                "code": 1,
                "status": error,
                "query": args.search_query
            }
    else:
        log.info("✗ Search skipped (index not initialized)")
        test_results["search"] = {
            "code": 1,
            "status": "Index not initialized",
            "query": args.search_query
        }

    # === test tag detection
    log.info("\n" + "-"*80)
    log.info("Test 5: Tag Detection")
    log.info("-"*80)
    
    # Build file_path to record_id mapping from process_file.json
    file_to_record_id = {}
    if process_file_path.exists():
        with open(process_file_path, "r", encoding="utf-8") as f:
            process_file_data = json.load(f)
            for file_info in process_file_data.get("files", []):
                file_to_record_id[file_info["file_path"]] = file_info["record_id"]
        log.info(f"Loaded {len(file_to_record_id)} file-to-record_id mappings")
    
    if image_files:
        tag_results_list = []
        success_count = 0
        total_tags_count = 0
        
        for i, test_image in enumerate(image_files, 1):
            log.info(f"\n[{i}/{len(image_files)}] Testing image: {Path(test_image).name}")
            
            # Get record_id from mapping
            record_id = file_to_record_id.get(test_image)
            if record_id is None:
                log.info(f"  ✗ Skipped (record_id not found in FAISS index)")
                tag_results_list.append({
                    "code": 1,
                    "status": "record_id not found",
                    "file_path": test_image
                })
                continue
            
            tag_params = {
                "cmd": "generate_tags",
                "record_id": record_id,
                "tag_level": "all"
            }
            tag_response = process.send_request("tag_detect", tag_params)
            
            if tag_response and tag_response.get("code") == 0:
                tag_data = tag_response.get("data", {})
                tag_data_dbg = tag_response.get("data_dbg", {})
                
                tags = tag_data.get("tags", [])
                level1_tags = tag_data_dbg.get("level1_tags", [])
                level2_tags = tag_data_dbg.get("level2_tags", [])
                attribute_tags = tag_data_dbg.get("attribute_tags", [])
                process_time_ms = tag_data_dbg.get("process_time_ms", 0)
                
                total_tags = len(tags)
                total_tags_count += total_tags
                success_count += 1
                log.info(f"  ✓ Generated {total_tags} tags in {process_time_ms}ms")
                
                if level1_tags:
                    log.info(f"    Level 1: {', '.join([tag.get('name', '') for tag in level1_tags[:3]])}")
                
                tag_results_list.append({
                    "code": 0,
                    "status": "success",
                    "file_path": test_image,
                    "tags": tags,
                    "level1_tags": level1_tags,
                    "level2_tags": level2_tags,
                    "attribute_tags": attribute_tags,
                    "process_time_ms": process_time_ms
                })
            else:
                log.info(f"  ✗ Tag detection failed")
                tag_results_list.append({
                    "code": 1,
                    "status": "Tag detection failed",
                    "file_path": test_image
                })
        
        log.info(f"\n{'='*60}")
        log.info(f"Tag Detection Summary:")
        log.info(f"  - Total images: {len(image_files)}")
        log.info(f"  - Successful: {success_count}")
        log.info(f"  - Failed: {len(image_files) - success_count}")
        log.info(f"  - Total tags generated: {total_tags_count}")
        log.info(f"{'='*60}")
        
        test_results["tag_detect"] = {
            "code": 0 if success_count > 0 else 1,
            "status": "success" if success_count > 0 else "All tag detections failed",
            "results": tag_results_list,
            "success_count": success_count,
            "total_count": len(image_files),
            "total_tags": total_tags_count
        }
    
    log.info("\n" + "-"*80)
    log.info("Test 6: Find Similar Images")
    log.info("-"*80)
    
    if test_results.get("index_init", {}).get("code") == 0:
        log.info("Step 1: Find similar images by record_ids...")
        # test find similar  
        find_similar_params = {
            "cmd": "find_similar",
            "record_ids": [1001],
            "top_k": args.top_k,
            "min_similarity": 0.5
        }
        
        similar_response = process.send_request("find_similar", find_similar_params, timeout=30.0)
        
        if similar_response and similar_response.get("code") == 0:
            similar_data = similar_response.get("data", {})
            results = similar_data.get("results", [])
            query_time = similar_data.get("total_time_ms", 0)
            batch_size = similar_data.get("batch_size", 0)
            
            log.info(f"✓ Find similar successful, processed {batch_size} reference(s)")
            log.info(f"  - Query time: {query_time}ms")
            
            record_id_to_path = test_results.get("index_init", {}).get("record_id_to_path", {})
            
            for result in results:
                reference_id = result.get("reference_id")
                similar_images = result.get("similar_images", [])
                total_count = result.get("total_count", 0)
                
                log.info(f"  Reference ID {reference_id}: found {total_count} similar images")
                
                for img in similar_images:
                    record_id = img.get("record_id")
                    similarity = img.get("similarity")
                    rank = img.get("rank")
                    image_path = record_id_to_path.get(record_id, "")
                    image_name = Path(image_path).name if image_path else f"ID_{record_id}"
                    log.info(f"    {rank}. {image_name} (ID: {record_id}, similarity: {similarity:.4f})")
            
            test_results["find_similar"] = {
                "code": 0,
                "status": "success",
                "reference_id": 1001,
                "similar_images": similar_images,
                "query_time_ms": query_time,
                "record_id_to_path": record_id_to_path
            }
        else:
            log.info("✗ Find similar failed")
            error = similar_response.get("status", "Unknown error") if similar_response else "No response"
            log.info(f"  - Error: {error}")
            
            test_results["find_similar"] = {
                "code": 1,
                "status": error
            }
    else:
        log.info("✗ Find similar skipped (index not initialized)")
        test_results["find_similar"] = {
            "code": 1,
            "status": "Index not initialized"
        }
    
    log.info("\n" + "-"*80)
    log.info("Test 7: Detect Similar Groups (N-to-N Similarity Detection)")
    log.info("-"*80)
    
    if test_results.get("index_init", {}).get("code") == 0:
        log.info("Step 1: Detect similar groups across all images...")
        # test detect similar groups  
        detect_groups_params = {
            "cmd": "detect_similar_groups",
            "min_similarity": 0.85,
            "scope": "all",
            "top_k_per_image": 10
        }
        
        detect_response = process.send_request("find_similar", detect_groups_params, timeout=60.0)
        
        if detect_response and detect_response.get("code") == 0:
            detect_data = detect_response.get("data", {})
            
            total_images = detect_data.get("total_images", 0)
            total_groups = detect_data.get("total_groups", 0)
            total_pairs = detect_data.get("total_similar_pairs", 0)
            groups = detect_data.get("groups", [])
            query_time = detect_data.get("query_time_ms", 0)
            
            log.info(f"✓ Detect similar groups successful")
            log.info(f"  - Total images: {total_images}")
            log.info(f"  - Total groups: {total_groups}")
            log.info(f"  - Total similar pairs: {total_pairs}")
            log.info(f"  - Query time: {query_time}ms")
            
            record_id_to_path = test_results.get("index_init", {}).get("record_id_to_path", {})
            
            if groups:
                log.info(f"\n  Top groups (sorted by pair_count):")
                for group in groups[:5]:
                    group_id = group.get("group_id")
                    images = group.get("images", [])
                    avg_sim = group.get("avg_similarity", 0)
                    pair_count = group.get("pair_count", 0)
                    
                    log.info(f"\n    Group {group_id} ({len(images)} images, avg_sim: {avg_sim:.4f}, pairs: {pair_count}):")
                    for img in images[:5]:
                        record_id = img.get("record_id")
                        is_rep = img.get("representative", False)
                        sim_to_rep = img.get("similarity_to_representative")
                        image_path = record_id_to_path.get(record_id, "")
                        image_name = Path(image_path).name if image_path else f"ID_{record_id}"
                        
                        if is_rep:
                            log.info(f"      [REP] {image_name} (ID: {record_id})")
                        else:
                            log.info(f"            {image_name} (ID: {record_id}, sim_to_rep: {sim_to_rep:.4f})")
            
            test_results["detect_similar_groups"] = {
                "code": 0,
                "status": "success",
                "total_images": total_images,
                "total_groups": total_groups,
                "total_similar_pairs": total_pairs,
                "groups": groups,
                "query_time_ms": query_time,
                "record_id_to_path": record_id_to_path
            }
        else:
            log.info("✗ Detect similar groups failed")
            error = detect_response.get("status", "Unknown error") if detect_response else "No response"
            log.info(f"  - Error: {error}")
            
            test_results["detect_similar_groups"] = {
                "code": 1,
                "status": error
            }
        
        log.info("\nStep 2: Detect similar groups for specific record_ids...")
        
        record_id_to_path = test_results.get("index_init", {}).get("record_id_to_path", {})
        if len(record_id_to_path) >= 5:
            sample_ids = list(record_id_to_path.keys())[:5]
            
            detect_partial_params = {
                "cmd": "detect_similar_groups",
                "min_similarity": 0.8,
                "scope": "partial",
                "record_ids": sample_ids,
                "top_k_per_image": 5
            }
            
            detect_partial_response = process.send_request("find_similar", detect_partial_params, timeout=30.0)
            
            if detect_partial_response and detect_partial_response.get("code") == 0:
                partial_data = detect_partial_response.get("data", {})
                
                log.info(f"✓ Detect similar groups (partial) successful")
                log.info(f"  - Scope: partial ({len(sample_ids)} images)")
                log.info(f"  - Total groups: {partial_data.get('total_groups', 0)}")
                log.info(f"  - Query time: {partial_data.get('query_time_ms', 0)}ms")
                
                test_results["detect_similar_groups_partial"] = {
                    "code": 0,
                    "status": "success",
                    "scope": "partial",
                    "record_ids": sample_ids,
                    "total_groups": partial_data.get("total_groups", 0),
                    "query_time_ms": partial_data.get("query_time_ms", 0)
                }
            else:
                log.info("✗ Detect similar groups (partial) failed")
                error = detect_partial_response.get("status", "Unknown error") if detect_partial_response else "No response"
                log.info(f"  - Error: {error}")
                
                test_results["detect_similar_groups_partial"] = {
                    "code": 1,
                    "status": error
                }
    else:
        log.info("✗ Detect similar groups skipped (index not initialized)")
        test_results["detect_similar_groups"] = {
            "code": 1,
            "status": "Index not initialized"
        }
    
    process.stop()
    
    output_dir = project_root / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = output_dir / f"{timestamp}-ai-test-report.md"
    
    generate_markdown_report(test_results, str(md_path), test_data_dir)
    
    log.info("\n" + "="*80)
    log.info("Test Summary")
    log.info("="*80)
    
    log.info(f"\n✓ Ping: {test_results['ping'].get('code') == 0}")
    log.info(f"✓ Index Init: {test_results.get('index_init', {}).get('code') == 0}")
    log.info(f"✓ Search: {test_results.get('search', {}).get('code') == 0}")
    log.info(f"✓ Tag Detect: {test_results.get('tag_detect', {}).get('code') == 0}")
    log.info(f"✓ Find Similar: {test_results.get('find_similar', {}).get('code') == 0}")
    log.info(f"✓ Detect Similar Groups: {test_results.get('detect_similar_groups', {}).get('code') == 0}")
    
    detect_result = test_results.get('detect_similar_groups', {})
    if detect_result.get('code') == 0:
        log.info(f"  - Groups: {detect_result.get('total_groups', 0)}, Pairs: {detect_result.get('total_similar_pairs', 0)}")
    
    # Print timing statistics
    end_time = time.time()
    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_seconds = end_time - start_time
    elapsed_minutes = int(elapsed_seconds // 60)
    elapsed_secs = int(elapsed_seconds % 60)
    elapsed_ms = int((elapsed_seconds % 1) * 1000)
    
    log.info(f"\n{'='*80}")
    log.info("Test Timing Statistics:")
    log.info(f"  Start Time:   {start_time_str}")
    log.info(f"  End Time:     {end_time_str}")
    log.info(f"  Total Time:   {elapsed_minutes}m {elapsed_secs}s {elapsed_ms}ms ({elapsed_seconds:.2f}s)")
    log.info(f"{'='*80}")
    
    log.info(f"\nReport generated: {md_path}")
    log.info(f"\nOpen the Markdown report to view test results!")
    log.info("="*80 + "\n")


if __name__ == "__main__":
    main()