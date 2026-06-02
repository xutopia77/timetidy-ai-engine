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
            "image_path": image_path,
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


def get_relative_path(image_path: Path, base_dir: Path) -> str:
    """Get relative path from base directory"""
    try:
        if not base_dir.is_dir():
            base_dir = base_dir.parent
        rel_path = os.path.relpath(str(image_path), str(base_dir))
        return rel_path.replace("\\", "/")
    except ValueError:
        return str(image_path.name)


def generate_markdown_report(test_results: Dict, output_path: str, test_data_dir: str):
    """Generate markdown test report"""
    lines = []
    
    lines.append("# AI Engine stdio Communication Test Report (File Hash Based)")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Test Data**: `{test_data_dir}`")
    lines.append("")
    
    lines.append("## Test Summary")
    lines.append("")
    
    ping_result = test_results.get("ping", {})
    index_init_result = test_results.get("index_init", {})
    search_results = test_results.get("search", {})
    tag_results = test_results.get("tag_detect", {})
    sync_result = test_results.get("sync", {})
    
    lines.append("| Test | Status | Details |")
    lines.append("|------|--------|---------|")
    
    ping_status = "✓ Pass" if ping_result.get("success") else "✗ Fail"
    lines.append(f"| Ping | {ping_status} | {ping_result.get('message', '')} |")
    
    index_init_status = "✓ Pass" if index_init_result.get("success") else "✗ Fail"
    total_images = index_init_result.get("data", {}).get("total_images", 0)
    build_time = index_init_result.get("data", {}).get("build_time_seconds", 0)
    processed = index_init_result.get("data", {}).get("processed_files", 0)
    skipped = index_init_result.get("data", {}).get("skipped_files", 0)
    lines.append(f"| Index Init | {index_init_status} | {processed} processed, {skipped} skipped, {total_images} total |")
    
    search_status = "✓ Pass" if search_results.get("success") else "✗ Fail"
    search_count = len(search_results.get("results", []))
    query_time = search_results.get("query_time_ms", 0)
    lines.append(f"| Search | {search_status} | {search_count} results in {query_time}ms |")
    
    tag_status = "✓ Pass" if tag_results.get("success") else "✗ Fail"
    all_tags = tag_results.get("all_tags", [])
    tag_count = len(all_tags)
    lines.append(f"| Tag Detect | {tag_status} | Generated {tag_count} tags |")
    
    sync_status = "✓ Pass" if sync_result.get("success") else "✗ Fail"
    processed_files = sync_result.get("data", {}).get("processed_files", 0)
    skipped_files = sync_result.get("data", {}).get("skipped_files", 0)
    total_files = sync_result.get("data", {}).get("total_files", 0)
    lines.append(f"| Sync All | {sync_status} | {processed_files} processed, {skipped_files} skipped, {total_files} total |")
    
    lines.append("")
    
    if index_init_result.get("success"):
        lines.append("## Index Initialization (File Hash Based)")
        lines.append("")
        
        data = index_init_result.get("data", {})
        lines.append(f"- **Total files**: {data.get('total_images', 0)}")
        lines.append(f"- **Processed files**: {data.get('processed_files', 0)}")
        lines.append(f"- **Skipped files**: {data.get('skipped_files', 0)}")
        lines.append(f"- **Failed files**: {data.get('failed_files', 0)}")
        lines.append(f"- **Build time**: {data.get('build_time_seconds', 0)}s")
        lines.append(f"- **Index file**: `{data.get('index_file', '')}`")
        lines.append(f"- **Metadata file**: `{data.get('metadata_file', '')}`")
        lines.append("")

    elif index_init_result.get("error"):
        lines.append("## Index Initialization")
        lines.append("")
        lines.append(f"- **Error**: {index_init_result.get('error', 'Unknown error')}")
        lines.append("")
    
    if search_results.get("success") and search_results.get("results"):
        lines.append("## Search Results")
        lines.append("")
        
        query = search_results.get("query", "风景")
        results = search_results.get("results", [])
        record_id_to_path = search_results.get("record_id_to_path", {})
        
        lines.append(f"### Query: \"{query}\"")
        lines.append("")
        lines.append(f"**Query time**: {query_time}ms")
        lines.append("")
        lines.append(f"**Found {len(results)} images**")
        lines.append("")
        
        lines.append("| Rank | Image | Record ID | Similarity |")
        lines.append("|------|-------|-----------|------------|")
        
        for result in results:
            rank = result.get("rank", 0)
            record_id = result.get("record_id", 0)
            similarity = result.get("similarity", 0)
            image_path = record_id_to_path.get(record_id, "")
            image_name = Path(image_path).name if image_path else f"ID_{record_id}"
            lines.append(f"| {rank} | {image_name} | {record_id} | {similarity:.4f} |")
        
        lines.append("")
        
        if results and record_id_to_path:
            lines.append("### Image Preview")
            lines.append("")
            
            md_file_dir = Path(output_path).parent
            
            for i, result in enumerate(results[:5], 1):
                record_id = result.get("record_id", 0)
                similarity = result.get("similarity", 0)
                image_path = record_id_to_path.get(record_id, "")
                
                if image_path:
                    rel_path = get_relative_path(Path(image_path), md_file_dir)
                    image_name = Path(image_path).name
                    
                    lines.append(f"**{i}. {image_name}** (similarity: {similarity:.4f})")
                    lines.append("")
                    lines.append(f"![{image_name}]({rel_path})")
                    lines.append("")
                    lines.append(f"*Path: `{rel_path}`*")
                    lines.append("")
                    lines.append("---")
                    lines.append("")
        
        lines.append("---")
        lines.append("")
    
    if tag_results.get("success"):
        lines.append("## Tag Detection Results")
        lines.append("")
        
        image_path = tag_results.get("image_path", "")
        level1_tags = tag_results.get("level1_tags", [])
        level2_tags = tag_results.get("level2_tags", [])
        attribute_tags = tag_results.get("attribute_tags", [])
        all_tags = tag_results.get("all_tags", [])
        
        if image_path:
            lines.append(f"### Image: {Path(image_path).name}")
            lines.append("")
            
            md_file_dir = Path(output_path).parent
            rel_path = get_relative_path(Path(image_path), md_file_dir)
            
            lines.append(f"![{Path(image_path).name}]({rel_path})")
            lines.append("")
            lines.append(f"**Path**: `{rel_path}`")
            lines.append("")
        
        total_tags = len(all_tags)
        lines.append(f"**Generated {total_tags} tags**:")
        lines.append("")
        
        if level1_tags:
            lines.append("#### Level 1 - Primary Category")
            lines.append("")
            lines.append("| Category | Confidence |")
            lines.append("|----------|------------|")
            
            for tag in level1_tags[:3]:
                name = tag.get("name", "")
                confidence = tag.get("confidence", 0.0)
                lines.append(f"| {name} | {confidence:.4f} |")
            
            lines.append("")
        
        if level2_tags:
            lines.append("#### Level 2 - Sub-category")
            lines.append("")
            lines.append("| Sub-category | Confidence | Parent |")
            lines.append("|--------------|------------|--------|")
            
            for tag in level2_tags[:5]:
                name = tag.get("name", "")
                confidence = tag.get("confidence", 0.0)
                parent = tag.get("parent_id", "")
                lines.append(f"| {name} | {confidence:.4f} | {parent} |")
            
            lines.append("")
        
        if attribute_tags:
            lines.append("#### Attributes")
            lines.append("")
            lines.append("| Attribute | Confidence | Type |")
            lines.append("|-----------|------------|------|")
            
            for tag in attribute_tags[:5]:
                name = tag.get("name", "")
                confidence = tag.get("confidence", 0.0)
                attr_type = tag.get("type", "")
                lines.append(f"| {name} | {confidence:.4f} | {attr_type} |")
            
            lines.append("")
        
        if all_tags:
            lines.append("#### All Tags (Sorted by Confidence)")
            lines.append("")
            lines.append("| Tag | Level | Confidence |")
            lines.append("|-----|-------|------------|")
            
            for tag in all_tags[:10]:
                name = tag.get("name", "")
                level = tag.get("level", "")
                confidence = tag.get("confidence", 0.0)
                lines.append(f"| {name} | {level} | {confidence:.4f} |")
            
            lines.append("")
    
    if sync_result.get("success"):
        lines.append("## Data Synchronization (File Hash Based)")
        lines.append("")
        
        data = sync_result.get("data", {})
        
        lines.append("### Sync All Results")
        lines.append("")
        lines.append(f"- **Total files**: {data.get('total_files', 0)}")
        lines.append(f"- **Processed files**: {data.get('processed_files', 0)}")
        lines.append(f"- **Skipped files**: {data.get('skipped_files', 0)}")
        lines.append(f"- **Failed files**: {data.get('failed_files', 0)}")
        lines.append(f"- **Process time**: {data.get('process_time_ms', 0)}ms")
        lines.append(f"- **Message**: {data.get('message', '')}")
        lines.append("")
        
        lines.append("### File Hash Comparison Logic")
        lines.append("")
        lines.append("```")
        lines.append("Python compares file_hash from process_file.json:")
        lines.append("")
        lines.append("1. New file: file_hash not in metadata")
        lines.append("   -> Add new vector")
        lines.append("")
        lines.append("2. Updated file: file_hash changed")
        lines.append("   -> Remove old vector, add new vector")
        lines.append("")
        lines.append("3. Unchanged file: file_hash same")
        lines.append("   -> Skip (no processing)")
        lines.append("```")
        lines.append("")
        
        lines.append("### Sync Flow")
        lines.append("")
        lines.append("```")
        lines.append("Go generates process_file.json")
        lines.append("  -> Python reads process_file.json")
        lines.append("  -> Python compares file_hash")
        lines.append("  -> Python processes changed files")
        lines.append("  -> Python skips unchanged files")
        lines.append("  -> Python returns sync results")
        lines.append("```")
        lines.append("")
    elif sync_result.get("error"):
        lines.append("## Data Synchronization")
        lines.append("")
        lines.append(f"- **Error**: {sync_result.get('error', 'Unknown error')}")
        lines.append("")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\n✓ Markdown report generated: {output_path}")


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
    
    ping_response = process.send_request("ping", {})
    
    if ping_response and ping_response.get("success"):
        print("✓ Ping successful")
        test_results["ping"] = {
            "success": True,
            "message": ping_response.get("data", {}).get("message", "pong")
        }
    else:
        print("✗ Ping failed")
        test_results["ping"] = {
            "success": False,
            "message": str(ping_response)
        }
    
    print("\n" + "-"*80)
    print("Test 2: Index Initialization (File Hash Based)")
    print("-"*80)
    
    print("Step 1: Simulate Go creating process_file.json...")
    
    config_dir = project_root / "src" / "config"
    config_dir.mkdir(exist_ok=True)
    process_file_path = config_dir / "process_file.json"
    
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
        "operation": "start",
        "rebuild": True
    }
    
    start_response = process.send_request("index_init", start_params, timeout=300.0)
    
    if start_response and start_response.get("success"):
        start_data = start_response.get("data", {})
        
        print(f"✓ Index build started")
        print(f"  - Total files: {start_data.get('total_files', 0)}")
        print(f"  - Status: {start_data.get('status', '')}")
        
        print("\nStep 3: Monitor build progress...")
        
        max_wait_time = 300
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            progress_params = {
                "operation": "get_progress"
            }
            
            progress_response = process.send_request("index_init", progress_params)
            
            if progress_response and progress_response.get("success"):
                progress_data1 = progress_response.get("data", {})
                progress_data = progress_data1.get("data", {})
                
                status = progress_data.get("status", "")
                processed = progress_data.get("processed_files", 0)
                skipped = progress_data.get("skipped_files", 0)
                total = progress_data.get("total_files", 0)
                percent = progress_data.get("progress_percent", 0.0)

                print(f"Progress: {processed + skipped}/{total} ({percent:.1f}%), Processed: {processed}, Skipped: {skipped}")
                
                if status == "completed":
                    print(f"✓ Index build completed")
                    
                    test_results["index_init"] = {
                        "success": True,
                        "data": {
                            "total_images": total,
                            "processed_files": processed,
                            "skipped_files": skipped,
                            "failed_files": progress_data.get("failed_files", 0),
                            "build_time_seconds": round(time.time() - start_time, 2),
                            "index_file": str(project_root / "src" / "data" / "index" / "faiss_index.bin"),
                            "metadata_file": str(project_root / "src" / "data" / "index" / "ai_metadata.json")
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
                "success": False,
                "error": "Index build timeout"
            }
    else:
        print("✗ Index initialization failed")
        error = start_response.get("error", "Unknown error") if start_response else "No response"
        print(f"  - Error: {error}")
        
        test_results["index_init"] = {
            "success": False,
            "error": error
        }
    
    print("\n" + "-"*80)
    print("Test 3: Text-to-Image Search")
    print("-"*80)
    
    if test_results.get("index_init", {}).get("success"):
        search_params = {
            "action": "search",
            "query": args.search_query,
            "top_k": args.top_k,
            "filters": {}
        }
        
        print(f"Query: \"{args.search_query}\"")
        print(f"Searching in indexed images...")
        
        search_response = process.send_request("search", search_params, timeout=30.0)
        
        if search_response and search_response.get("success"):
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
                "success": True,
                "query": args.search_query,
                "query_time_ms": query_time,
                "results": results,
                "record_id_to_path": record_id_to_path
            }
        else:
            print("✗ Search failed")
            error = search_response.get("error", "Unknown error") if search_response else "No response"
            print(f"  - Error: {error}")
            
            test_results["search"] = {
                "success": False,
                "query": args.search_query,
                "error": error
            }
    else:
        print("✗ Search skipped (index not initialized)")
        test_results["search"] = {
            "success": False,
            "query": args.search_query,
            "error": "Index not initialized"
        }
    
    print("\n" + "-"*80)
    print("Test 4: Tag Detection")
    print("-"*80)
    
    if image_files:
        test_image = image_files[0]
        print(f"Testing image: {Path(test_image).name}")
        
        tag_params = {
            "action": "generate_tags",
            "image_path": test_image,
            "tag_level": "all"
        }
        
        tag_response = process.send_request("tag_detect", tag_params)
        
        if tag_response and tag_response.get("success"):
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
                "success": True,
                "image_path": test_image,
                "level1_tags": level1_tags,
                "level2_tags": level2_tags,
                "attribute_tags": attribute_tags,
                "all_tags": all_tags
            }
        else:
            print("✗ Tag detection failed")
            test_results["tag_detect"] = {
                "success": False,
                "image_path": test_image,
                "tags": []
            }
    
    print("\n" + "-"*80)
    print("Test 5: Data Synchronization (File Hash Based)")
    print("-"*80)
    
    if test_results.get("index_init", {}).get("success") and len(image_files) >= 2:
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
            "operation": "sync_all"
        }
        
        sync_response = process.send_request("sync", sync_params, timeout=60.0)
        
        if sync_response and sync_response.get("success"):
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
                "success": True,
                "data": sync_data
            }
        else:
            print("✗ Sync failed")
            error = sync_response.get("error", "Unknown error") if sync_response else "No response"
            print(f"  - Error: {error}")
            
            test_results["sync"] = {
                "success": False,
                "error": error
            }
    else:
        print("✗ Sync skipped (index not initialized or insufficient images)")
        test_results["sync"] = {
            "success": False,
            "error": "Index not initialized or insufficient images"
        }
    
    process.stop()
    
    output_dir = project_root / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = output_dir / f"stdio_communication_test_{timestamp}.md"
    
    generate_markdown_report(test_results, str(md_path), test_data_dir)
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    print(f"\n✓ Ping: {test_results['ping'].get('success', False)}")
    print(f"✓ Index Init: {test_results.get('index_init', {}).get('success', False)}")
    print(f"✓ Search: {test_results.get('search', {}).get('success', False)}")
    print(f"✓ Tag Detect: {test_results.get('tag_detect', {}).get('success', False)}")
    print(f"✓ Sync All: {test_results.get('sync', {}).get('success', False)}")
    
    print(f"\nReport generated: {md_path}")
    print(f"\nOpen the Markdown report to view test results!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()