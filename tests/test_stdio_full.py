"""
Test AI Engine stdio communication and generate markdown report

This script tests:
1. Ping communication
2. Index initialization (FAISS)
3. Text-to-image search (FAISS)
4. Tag detection
5. Data synchronization events
6. Generates markdown report with image links
"""

import sys
import os
import json
import argparse
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


def generate_image_records(image_files: List[str], start_id: int = 1001) -> tuple:
    """Generate image records for index initialization and record_id mapping"""
    records = []
    record_id_to_path = {}
    
    for i, image_path in enumerate(image_files):
        record_id = start_id + i
        records.append({
            "record_id": record_id,
            "image_path": image_path,
            "image_hash": f"hash_{record_id}"
        })
        record_id_to_path[record_id] = image_path
    
    return records, record_id_to_path


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
    
    lines.append("# AI Engine stdio Communication Test Report")
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
    lines.append(f"| Index Init | {index_init_status} | {total_images} images in {build_time}s |")
    
    search_status = "✓ Pass" if search_results.get("success") else "✗ Fail"
    search_count = len(search_results.get("results", []))
    query_time = search_results.get("query_time_ms", 0)
    lines.append(f"| Search | {search_status} | {search_count} results in {query_time}ms |")
    
    tag_status = "✓ Pass" if tag_results.get("success") else "✗ Fail"
    all_tags = tag_results.get("all_tags", [])
    tag_count = len(all_tags)
    lines.append(f"| Tag Detect | {tag_status} | Generated {tag_count} tags |")
    
    sync_status = "✓ Pass" if sync_result.get("success") else "✗ Fail"
    event_type = sync_result.get("event_type", "")
    process_time = sync_result.get("data", {}).get("process_time_ms", 0)
    lines.append(f"| Sync Event | {sync_status} | {event_type} in {process_time}ms |")
    
    lines.append("")
    
    if index_init_result.get("success"):
        lines.append("## Index Initialization")
        lines.append("")
        
        data = index_init_result.get("data", {})
        lines.append(f"- **Total images indexed**: {data.get('total_images', 0)}")
        lines.append(f"- **Build time**: {data.get('build_time_seconds', 0)}s")
        lines.append(f"- **Failed records**: {data.get('failed_records', 0)}")
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
            
            for tag in level1_tags:
                name = tag.get("name", "")
                confidence = tag.get("confidence", 0.0)
                lines.append(f"| {name} | {confidence:.4f} |")
            
            lines.append("")
        
        if level2_tags:
            lines.append("#### Level 2 - Sub-category")
            lines.append("")
            lines.append("| Sub-category | Confidence | Parent |")
            lines.append("|--------------|------------|--------|")
            
            for tag in level2_tags:
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
            
            for tag in attribute_tags:
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
        lines.append("## Sync Event Results")
        lines.append("")
        
        event_type = sync_result.get("event_type", "")
        data = sync_result.get("data", {})
        
        lines.append(f"### Event: {event_type}")
        lines.append("")
        lines.append(f"- **Process time**: {data.get('process_time_ms', 0)}ms")
        lines.append(f"- **Message**: {data.get('message', '')}")
        lines.append("")
    elif sync_result.get("error"):
        lines.append("## Sync Event Results")
        lines.append("")
        lines.append(f"- **Error**: {sync_result.get('error', 'Unknown error')}")
        lines.append("")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\n✓ Markdown report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test AI Engine stdio communication"
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
    print("AI Engine stdio Communication Test")
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
    
    # =============================================================
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
    print("Test 2: Index Initialization (FAISS)")
    print("-"*80)
    
    print("Deleting old index files...")
    
    # =============================================================
    delete_params = {
        "action": "delete"
    }
    delete_response = process.send_request("index_init", delete_params)
    if delete_response and delete_response.get("success"):
        deleted_files = delete_response.get("data", {}).get("deleted_files", [])
        print(f"✓ Old index deleted: {len(deleted_files)} files removed")
    else:
        print("✗ Old index deletion failed (or no old index exists)")
    
    image_records, record_id_to_path = generate_image_records(image_files, start_id=1001)
    print(f"Initializing index with {len(image_records)} images...")
    index_init_params = {
        "action": "init",
        "rebuild": True,
        "image_records": image_records
    }
    index_init_response = process.send_request("index_init", index_init_params, timeout=300.0)
    if index_init_response and index_init_response.get("success"):
        index_init_data = index_init_response.get("data", {})
        
        print(f"✓ Index initialized successfully")
        print(f"  - Total images: {index_init_data.get('total_images', 0)}")
        print(f"  - Build time: {index_init_data.get('build_time_seconds', 0)}s")
        print(f"  - Failed records: {index_init_data.get('failed_records', 0)}")
        
        test_results["index_init"] = {
            "success": True,
            "data": index_init_data,
            "record_id_to_path": record_id_to_path
        }
    else:
        print("✗ Index initialization failed")
        error = index_init_response.get("error", "Unknown error") if index_init_response else "No response"
        print(f"  - Error: {error}")
        
        test_results["index_init"] = {
            "success": False,
            "error": error
        }
    
    print("\n" + "-"*80)
    print("Test 3: Text-to-Image Search (FAISS)")
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
        print(f"tag_detect resp data {tag_response}")
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
    print("Test 5: Data Synchronization Event")
    print("-"*80)
    
    if test_results.get("search", {}).get("success") and test_results.get("search", {}).get("results"):
        first_result = test_results["search"]["results"][0]
        record_id = first_result.get("record_id")
        
        print(f"Testing IMAGE_DELETED event for record_id: {record_id}")
        
        sync_params = {
            "event_type": "IMAGE_DELETED",
            "record_id": record_id
        }
        
        sync_response = process.send_request("sync", sync_params)
        
        if sync_response and sync_response.get("success"):
            sync_data = sync_response.get("data", {})
            
            print(f"✓ Sync event successful")
            print(f"  - Event type: IMAGE_DELETED")
            print(f"  - Process time: {sync_data.get('process_time_ms', 0)}ms")
            
            test_results["sync"] = {
                "success": True,
                "event_type": "IMAGE_DELETED",
                "data": sync_data
            }
        else:
            print("✗ Sync event failed")
            error = sync_response.get("error", "Unknown error") if sync_response else "No response"
            print(f"  - Error: {error}")
            
            test_results["sync"] = {
                "success": False,
                "event_type": "IMAGE_DELETED",
                "error": error
            }
    else:
        print("✗ Sync event skipped (no search results)")
        test_results["sync"] = {
            "success": False,
            "event_type": "IMAGE_DELETED",
            "error": "No search results available"
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
    print(f"✓ Sync Event: {test_results.get('sync', {}).get('success', False)}")
    
    print(f"\nReport generated: {md_path}")
    print(f"\nOpen the Markdown report to view test results!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()