"""
Test AI Engine stdio communication (File Hash Based)

This script simulates Go behavior and tests:
1. Ping communication
2. Index initialization (FAISS) with process_file.json
3. Progress monitoring (start/stop/continue/get_progress)
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

from python_process import PythonProcess
from report_generator import generate_markdown_report


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
        print(f"✓ Process file created: {output_path}")
    
    return process_file_data, record_id_to_path


def update_process_file_with_new_hash(process_file_path: str, record_id: int, new_hash: str):
    """
    Update process_file.json with new file_hash (simulate Go behavior)
    
    Args:
        process_file_path: path to process_file.json
        record_id: record ID to update
        new_hash: new file hash
    """
    with open(process_file_path, "r", encoding="utf-8") as f:
        process_data = json.load(f)
    
    for file_record in process_data["files"]:
        if file_record["record_id"] == record_id:
            file_record["file_hash"] = new_hash
            file_record["status"] = "pending"
            break
    
    process_data["last_update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(process_file_path, "w", encoding="utf-8") as f:
        json.dump(process_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Process file updated: record_id={record_id}, new_hash={new_hash}")


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
        default="风景",
        help="Search query text"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("AI Engine stdio Communication Test (File Hash Based)")
    print("="*80)
    
    project_root = Path(__file__).parent.parent
    
    test_data_dir = args.test_dir
    if not os.path.exists(test_data_dir):
        print(f"\n✗ Test data directory not found: {test_data_dir}")
        return
    
    print(f"\nTest directory: {test_data_dir}")
    
    image_files = find_image_files(test_data_dir)
    
    if not image_files:
        print(f"\n✗ No image files found in: {test_data_dir}")
        return
    
    total_found = len(image_files)
    image_files = image_files[:args.max_images]
    
    print(f"Found {total_found} images, using {len(image_files)} for testing")
    
    process = PythonProcess()
    
    if not process.start():
        print("\n✗ Failed to start Python process")
        return
    
    test_results = {}
    
    print("\n" + "-"*80)
    print("Test 1: Ping Communication")
    print("-"*80)
    # test ping =========================================================== 
    ping_response = process.send_request("ping", {})
    
    if ping_response and ping_response.get("code") == 0:
        print("✓ Ping successful")
        test_results["ping"] = {
            "code": 0,
            "status": "success",
            "message": ping_response.get("data", {}).get("message", "pong")
        }
    else:
        print("✗ Ping failed")
        test_results["ping"] = {
            "code": 1,
            "status": str(ping_response) if ping_response else "No response",
        }
    
    print("\n" + "-"*80)
    print("Test 2: Index Initialization (File Hash Based)")
    print("-"*80)

    # test index init =========================================================== 
    print("Step 1: Simulate Go creating process_file.json...")
    
    progress_dir = project_root / "ai-data" / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    process_file_path = progress_dir / "process_file.json"
    
    process_file_data, record_id_to_path = create_process_file(
        image_files,
        start_id=1001,
        output_path=str(process_file_path)
    )
    
    print(f"  - Created process_file.json with {len(image_files)} files")
    print(f"  - All files status: pending")
    print(f"  - Each file has unique file_hash")
    
    print("\nStep 2: Send start request to Python...")
    
    start_params = {
        "cmd": "start",
        "rebuild": True
    }
    
    start_response = process.send_request("index_init", start_params, timeout=300.0)
    
    if start_response and start_response.get("code") == 0:
        start_data = start_response.get("data", {})
        
        print(f"✓ Index build started")
        print(f"  - Total files: {start_data.get('total_files', 0)}")
        print(f"  - Status: {start_data.get('status', '')}")
        
        print("\nStep 3: Monitor build progress...")
        
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

                print(f"Progress: {processed + skipped}/{total} ({percent:.1f}%), Processed: {processed}, Skipped: {skipped}")
                
                if status == "completed":
                    print(f"✓ Index build completed")
                    
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
                print("✗ Failed to get progress")
                break
        
        if time.time() - start_time >= max_wait_time:
            print("✗ Index build timeout")
            test_results["index_init"] = {
                "code": 1,
                "status": "Index build timeout"
            }
    else:
        print("✗ Index initialization failed")
        error = start_response.get("status", "Unknown error") if start_response else "No response"
        print(f"  - Error: {error}")
        
        test_results["index_init"] = {
            "code": 1,
            "status": error
        }
    
    print("\n" + "-"*80)
    print("Test 3: Text-to-Image Search")
    print("-"*80)
    
    if test_results.get("index_init", {}).get("code") == 0:
        # search_params = {
        #     "action": "search",
        #     "query": args.search_query,
        #     "top_k": args.top_k,
        #     "filters": {}
        # }
        
        search_params = {
            "cmd": "search",
            "query": args.search_query,
            "top_k": 20
        }

        print(f"Query: \"{args.search_query}\"")
        print(f"Searching in indexed images...")
        
        # test search =========================================================== 
        search_response = process.send_request("search", search_params, timeout=30.0)
        
        if search_response and search_response.get("code") == 0:
            search_data = search_response.get("data", {})
            results = search_data.get("results", [])
            query_time = search_data.get("query_time_ms", 0)
            
            record_id_to_path = test_results.get("index_init", {}).get("record_id_to_path", {})
            
            print(f"✓ Search successful, found {len(results)} results")
            print(f"  - Query time: {query_time}ms")
            
            for i, result in enumerate(results, 1):
                record_id = result.get("record_id")
                similarity = result.get("similarity")
                rank = result.get("rank")
                image_path = record_id_to_path.get(record_id, "")
                image_name = Path(image_path).name if image_path else f"ID_{record_id}"
                print(f"    {i}. {image_name} (ID: {record_id}, similarity: {similarity:.4f}, rank: {rank})")
            
            test_results["search"] = {
                "code": 0,
                "status": "success",
                "query": args.search_query,
                "query_time_ms": query_time,
                "results": results,
                "record_id_to_path": record_id_to_path
            }
        else:
            print("✗ Search failed")
            error = search_response.get("status", "Unknown error") if search_response else "No response"
            print(f"  - Error: {error}")
            
            test_results["search"] = {
                "code": 1,
                "status": error,
                "query": args.search_query
            }
    else:
        print("✗ Search skipped (index not initialized)")
        test_results["search"] = {
            "code": 1,
            "status": "Index not initialized",
            "query": args.search_query
        }
    
    print("\n" + "-"*80)
    print("Test 4: Tag Detection")
    print("-"*80)
    
    # test tag detect =========================================================== 
    if image_files:
        test_image = image_files[0]
        print(f"Testing image: {Path(test_image).name}")
        
        tag_params = {
            "cmd": "generate_tags",
            "file_path": test_image,
            "tag_level": "all"
        }
        
        tag_response = process.send_request("tag_detect", tag_params)
        
        if tag_response and tag_response.get("code") == 0:
            tag_data = tag_response.get("data", {})
            
            level1_tags = tag_data.get("level1_tags", [])
            level2_tags = tag_data.get("level2_tags", [])
            attribute_tags = tag_data.get("attribute_tags", [])
            all_tags = tag_data.get("all_tags", [])
            
            total_tags = len(all_tags)
            print(f"✓ Tag detection successful, generated {total_tags} tags")
            
            if level1_tags:
                print(f"\n  Level 1 (Primary Category):")
                for tag in level1_tags[:3]:
                    name = tag.get("name", "")
                    confidence = tag.get("confidence", 0.0)
                    print(f"    - {name} (confidence: {confidence:.4f})")
            
            if level2_tags:
                print(f"\n  Level 2 (Sub-category):")
                for tag in level2_tags[:5]:
                    name = tag.get("name", "")
                    confidence = tag.get("confidence", 0.0)
                    parent = tag.get("parent_id", "")
                    print(f"    - {name} (confidence: {confidence:.4f}, parent: {parent})")
            
            if attribute_tags:
                print(f"\n  Attributes:")
                for tag in attribute_tags[:5]:
                    name = tag.get("name", "")
                    confidence = tag.get("confidence", 0.0)
                    attr_type = tag.get("type", "")
                    print(f"    - {name} (confidence: {confidence:.4f}, type: {attr_type})")
            
            test_results["tag_detect"] = {
                "code": 0,
                "status": "success",
                "file_path": test_image,
                "level1_tags": level1_tags,
                "level2_tags": level2_tags,
                "attribute_tags": attribute_tags,
                "all_tags": all_tags
            }
        else:
            print("✗ Tag detection failed")
            test_results["tag_detect"] = {
                "code": 1,
                "status": "Tag detection failed",
                "file_path": test_image
            }
    
    print("\n" + "-"*80)
    print("Test 5: Data Synchronization (File Hash Based)")
    print("-"*80)
    
    if test_results.get("index_init", {}).get("code") == 0 and len(image_files) >= 2:
        print("Step 1: Simulate Go updating process_file.json...")
        
        first_record_id = 1001
        new_hash = "new_hash_" + str(int(time.time()))
        
        update_process_file_with_new_hash(
            str(process_file_path),
            first_record_id,
            new_hash
        )
        
        print(f"  - Updated file_hash for record_id={first_record_id}")
        print(f"  - Old hash: {compute_file_hash(image_files[0])}")
        print(f"  - New hash: {new_hash}")
        
        print("\nStep 2: Send sync_all request...")
        
        sync_params = {
            "cmd": "sync_all"
        }
        
        sync_response = process.send_request("sync", sync_params, timeout=60.0)
        
        if sync_response and sync_response.get("code") == 0:
            sync_data = sync_response.get("data", {})
            
            total_files = sync_data.get("total_files", 0)
            processed_files = sync_data.get("processed_files", 0)
            skipped_files = sync_data.get("skipped_files", 0)
            failed_files = sync_data.get("failed_files", 0)
            process_time = sync_data.get("process_time_ms", 0)
            
            print(f"✓ Sync successful")
            print(f"  - Total files: {total_files}")
            print(f"  - Processed files: {processed_files} (file_hash changed)")
            print(f"  - Skipped files: {skipped_files} (file_hash unchanged)")
            print(f"  - Failed files: {failed_files}")
            print(f"  - Process time: {process_time}ms")
            
            test_results["sync"] = {
                "code": 0,
                "status": "success",
                "data": sync_data
            }
        else:
            print("✗ Sync failed")
            error = sync_response.get("status", "Unknown error") if sync_response else "No response"
            print(f"  - Error: {error}")
            
            test_results["sync"] = {
                "code": 1,
                "status": error
            }
    else:
        print("✗ Sync skipped (index not initialized or insufficient images)")
        test_results["sync"] = {
            "code": 1,
            "status": "Index not initialized or insufficient images"
        }
    
    print("\n" + "-"*80)
    print("Test 6: Find Similar Images")
    print("-"*80)
    
    if test_results.get("index_init", {}).get("code") == 0:
        print("Step 1: Find similar images by record_id...")
        # test find similar =========================================================== 
        find_similar_params = {
            "cmd": "find_similar",
            "record_id": 1001,
            "top_k": args.top_k,
            "min_similarity": 0.5
        }
        
        similar_response = process.send_request("find_similar", find_similar_params, timeout=30.0)
        
        if similar_response and similar_response.get("code") == 0:
            similar_data = similar_response.get("data", {})
            similar_images = similar_data.get("similar_images", [])
            query_time = similar_data.get("query_time_ms", 0)
            
            print(f"✓ Find similar successful, found {len(similar_images)} similar images")
            print(f"  - Query time: {query_time}ms")
            
            record_id_to_path = test_results.get("index_init", {}).get("record_id_to_path", {})
            
            for result in similar_images:
                record_id = result.get("record_id")
                similarity = result.get("similarity")
                rank = result.get("rank")
                image_path = record_id_to_path.get(record_id, "")
                image_name = Path(image_path).name if image_path else f"ID_{record_id}"
                print(f"    {rank}. {image_name} (ID: {record_id}, similarity: {similarity:.4f})")
            
            test_results["find_similar"] = {
                "code": 0,
                "status": "success",
                "reference_id": 1001,
                "similar_images": similar_images,
                "query_time_ms": query_time,
                "record_id_to_path": record_id_to_path
            }
        else:
            print("✗ Find similar failed")
            error = similar_response.get("status", "Unknown error") if similar_response else "No response"
            print(f"  - Error: {error}")
            
            test_results["find_similar"] = {
                "code": 1,
                "status": error
            }
        
        print("\nStep 2: Find similar images by image path...")
        # test find similar by image =========================================================== 
        if image_files:
            test_image = image_files[0]
            
            find_similar_by_image_params = {
                "cmd": "find_similar_by_image",
                "file_path": test_image,
                "top_k": args.top_k,
                "min_similarity": 0.5
            }
            
            similar_response = process.send_request("find_similar", find_similar_by_image_params, timeout=60.0)
            
            if similar_response and similar_response.get("code") == 0:
                similar_data = similar_response.get("data", {})
                similar_images = similar_data.get("similar_images", [])
                query_time = similar_data.get("query_time_ms", 0)
                encode_time = similar_data.get("encode_time_ms", 0)
                search_time = similar_data.get("search_time_ms", 0)
                
                print(f"✓ Find similar by image successful, found {len(similar_images)} similar images")
                print(f"  - Total time: {query_time}ms")
                print(f"  - Encode time: {encode_time}ms")
                print(f"  - Search time: {search_time}ms")
                
                record_id_to_path = test_results.get("index_init", {}).get("record_id_to_path", {})
                
                for result in similar_images:
                    record_id = result.get("record_id")
                    similarity = result.get("similarity")
                    rank = result.get("rank")
                    image_path = record_id_to_path.get(record_id, "")
                    image_name = Path(image_path).name if image_path else f"ID_{record_id}"
                    print(f"    {rank}. {image_name} (ID: {record_id}, similarity: {similarity:.4f})")
                
                test_results["find_similar_by_image"] = {
                    "code": 0,
                    "status": "success",
                    "file_path": test_image,
                    "similar_images": similar_images,
                    "query_time_ms": query_time,
                    "encode_time_ms": encode_time,
                    "search_time_ms": search_time,
                    "record_id_to_path": record_id_to_path
                }
            else:
                print("✗ Find similar by image failed")
                error = similar_response.get("status", "Unknown error") if similar_response else "No response"
                print(f"  - Error: {error}")
                
                test_results["find_similar_by_image"] = {
                    "code": 1,
                    "status": error
                }
    else:
        print("✗ Find similar skipped (index not initialized)")
        test_results["find_similar"] = {
            "code": 1,
            "status": "Index not initialized"
        }
        test_results["find_similar_by_image"] = {
            "code": 1,
            "status": "Index not initialized"
        }
    
    print("\n" + "-"*80)
    print("Test 7: Detect Similar Groups (N-to-N Similarity Detection)")
    print("-"*80)
    
    if test_results.get("index_init", {}).get("code") == 0:
        print("Step 1: Detect similar groups across all images...")
        # test detect similar groups =========================================================== 
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
            
            print(f"✓ Detect similar groups successful")
            print(f"  - Total images: {total_images}")
            print(f"  - Total groups: {total_groups}")
            print(f"  - Total similar pairs: {total_pairs}")
            print(f"  - Query time: {query_time}ms")
            
            record_id_to_path = test_results.get("index_init", {}).get("record_id_to_path", {})
            
            if groups:
                print(f"\n  Top groups (sorted by pair_count):")
                for group in groups[:5]:
                    group_id = group.get("group_id")
                    images = group.get("images", [])
                    avg_sim = group.get("avg_similarity", 0)
                    pair_count = group.get("pair_count", 0)
                    
                    print(f"\n    Group {group_id} ({len(images)} images, avg_sim: {avg_sim:.4f}, pairs: {pair_count}):")
                    for img in images[:5]:
                        record_id = img.get("record_id")
                        is_rep = img.get("representative", False)
                        sim_to_rep = img.get("similarity_to_representative")
                        image_path = record_id_to_path.get(record_id, "")
                        image_name = Path(image_path).name if image_path else f"ID_{record_id}"
                        
                        if is_rep:
                            print(f"      [REP] {image_name} (ID: {record_id})")
                        else:
                            print(f"            {image_name} (ID: {record_id}, sim_to_rep: {sim_to_rep:.4f})")
            
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
            print("✗ Detect similar groups failed")
            error = detect_response.get("status", "Unknown error") if detect_response else "No response"
            print(f"  - Error: {error}")
            
            test_results["detect_similar_groups"] = {
                "code": 1,
                "status": error
            }
        
        print("\nStep 2: Detect similar groups for specific record_ids...")
        
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
                
                print(f"✓ Detect similar groups (partial) successful")
                print(f"  - Scope: partial ({len(sample_ids)} images)")
                print(f"  - Total groups: {partial_data.get('total_groups', 0)}")
                print(f"  - Query time: {partial_data.get('query_time_ms', 0)}ms")
                
                test_results["detect_similar_groups_partial"] = {
                    "code": 0,
                    "status": "success",
                    "scope": "partial",
                    "record_ids": sample_ids,
                    "total_groups": partial_data.get("total_groups", 0),
                    "query_time_ms": partial_data.get("query_time_ms", 0)
                }
            else:
                print("✗ Detect similar groups (partial) failed")
                error = detect_partial_response.get("status", "Unknown error") if detect_partial_response else "No response"
                print(f"  - Error: {error}")
                
                test_results["detect_similar_groups_partial"] = {
                    "code": 1,
                    "status": error
                }
    else:
        print("✗ Detect similar groups skipped (index not initialized)")
        test_results["detect_similar_groups"] = {
            "code": 1,
            "status": "Index not initialized"
        }
    
    process.stop()
    
    output_dir = project_root / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = output_dir / f"ai-test-report-{timestamp}.md"
    
    generate_markdown_report(test_results, str(md_path), test_data_dir)
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    print(f"\n✓ Ping: {test_results['ping'].get('success', False)}")
    print(f"✓ Index Init: {test_results.get('index_init', {}).get('success', False)}")
    print(f"✓ Search: {test_results.get('search', {}).get('success', False)}")
    print(f"✓ Tag Detect: {test_results.get('tag_detect', {}).get('success', False)}")
    print(f"✓ Sync All: {test_results.get('sync', {}).get('success', False)}")
    print(f"✓ Find Similar: {test_results.get('find_similar', {}).get('success', False)}")
    print(f"✓ Find Similar by Image: {test_results.get('find_similar_by_image', {}).get('success', False)}")
    print(f"✓ Detect Similar Groups: {test_results.get('detect_similar_groups', {}).get('success', False)}")
    
    detect_result = test_results.get('detect_similar_groups', {})
    if detect_result.get('success'):
        print(f"  - Groups: {detect_result.get('total_groups', 0)}, Pairs: {detect_result.get('total_similar_pairs', 0)}")
    
    print(f"\nReport generated: {md_path}")
    print(f"\nOpen the Markdown report to view test results!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()