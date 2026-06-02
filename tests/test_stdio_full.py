"""
Test AI Engine stdio communication and generate markdown report

This script tests:
1. Ping communication
2. Text-to-image search
3. Tag detection
4. Generates markdown report with image links
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
    search_results = test_results.get("search", {})
    tag_results = test_results.get("tag_detect", {})
    
    lines.append("| Test | Status | Details |")
    lines.append("|------|--------|---------|")
    
    ping_status = "✓ Pass" if ping_result.get("success") else "✗ Fail"
    lines.append(f"| Ping | {ping_status} | {ping_result.get('message', '')} |")
    
    search_status = "✓ Pass" if search_results.get("success") else "✗ Fail"
    search_count = len(search_results.get("results", []))
    lines.append(f"| Search | {search_status} | Found {search_count} images |")
    
    tag_status = "✓ Pass" if tag_results.get("success") else "✗ Fail"
    tag_count = len(tag_results.get("tags", []))
    lines.append(f"| Tag Detect | {tag_status} | Generated {tag_count} tags |")
    
    lines.append("")
    
    if search_results.get("success") and search_results.get("results"):
        lines.append("## Search Results")
        lines.append("")
        
        query = search_results.get("query", "风景")
        results = search_results.get("results", [])
        
        lines.append(f"### Query: \"{query}\"")
        lines.append("")
        lines.append(f"**Found {len(results)} images**")
        lines.append("")
        
        lines.append("| Rank | Image | Similarity | Preview |")
        lines.append("|------|-------|------------|---------|")
        
        md_file_dir = Path(output_path).parent
        
        for i, result in enumerate(results, 1):
            image_path = Path(result["image_path"])
            image_name = image_path.name
            rel_path = get_relative_path(image_path, md_file_dir)
            similarity = result["similarity"]
            
            lines.append(f"| {i} | [{image_name}]({rel_path}) | {similarity:.4f} | ![]({rel_path}) |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        lines.append("### Detailed Search Results")
        lines.append("")
        
        for i, result in enumerate(results, 1):
            image_path = Path(result["image_path"])
            md_file_dir = Path(output_path).parent
            rel_path = get_relative_path(image_path, md_file_dir)
            
            lines.append(f"#### {i}. {image_path.name}")
            lines.append("")
            lines.append(f"![{image_path.name}]({rel_path})")
            lines.append("")
            lines.append(f"**Path**: `{rel_path}`")
            lines.append("")
            lines.append(f"**Similarity**: {result['similarity']:.4f}")
            lines.append("")
            lines.append("---")
            lines.append("")
    
    if tag_results.get("success") and tag_results.get("tags"):
        lines.append("## Tag Detection Results")
        lines.append("")
        
        image_path = tag_results.get("image_path", "")
        tags = tag_results.get("tags", [])
        
        if image_path:
            lines.append(f"### Image: {Path(image_path).name}")
            lines.append("")
            
            md_file_dir = Path(output_path).parent
            rel_path = get_relative_path(Path(image_path), md_file_dir)
            
            lines.append(f"![{Path(image_path).name}]({rel_path})")
            lines.append("")
            lines.append(f"**Path**: `{rel_path}`")
            lines.append("")
        
        lines.append(f"**Generated {len(tags)} tags**:")
        lines.append("")
        
        lines.append("| Tag | Confidence |")
        lines.append("|-----|------------|")
        
        for tag in tags:
            label = tag.get("label", "")
            confidence = tag.get("confidence", 0.0)
            lines.append(f"| {label} | {confidence:.4f} |")
        
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
        default=10,
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
    print("Test 2: Text-to-Image Search")
    print("-"*80)
    
    search_params = {
        "action": "search_by_text",
        "text": args.search_query,
        "image_paths": image_files,
        "top_k": args.top_k
    }
    
    print(f"Query: \"{args.search_query}\"")
    print(f"Searching in {len(image_files)} images...")
    
    search_response = process.send_request("search", search_params)
    
    if search_response and search_response.get("success"):
        search_data = search_response.get("data", {})
        results = search_data.get("results", [])
        
        print(f"✓ Search successful, found {len(results)} results")
        
        for i, result in enumerate(results, 1):
            image_name = Path(result["image_path"]).name
            similarity = result["similarity"]
            print(f"  {i}. {image_name} (similarity: {similarity:.4f})")
        
        test_results["search"] = {
            "success": True,
            "query": args.search_query,
            "results": results
        }
    else:
        print("✗ Search failed")
        test_results["search"] = {
            "success": False,
            "query": args.search_query,
            "results": []
        }
    
    print("\n" + "-"*80)
    print("Test 3: Tag Detection")
    print("-"*80)
    
    if image_files:
        test_image = image_files[0]
        print(f"Testing image: {Path(test_image).name}")
        
        tag_params = {
            "action": "tag_image",
            "image_path": test_image,
            "top_k": 10,
            "threshold": 0.2
        }
        
        tag_response = process.send_request("tag_detect", tag_params)
        
        if tag_response and tag_response.get("success"):
            tag_data = tag_response.get("data", {})
            tags = tag_data.get("tags", [])
            
            print(f"✓ Tag detection successful, generated {len(tags)} tags")
            
            for i, tag in enumerate(tags[:5], 1):
                label = tag.get("label", "")
                confidence = tag.get("confidence", 0.0)
                print(f"  {i}. {label} (confidence: {confidence:.4f})")
            
            test_results["tag_detect"] = {
                "success": True,
                "image_path": test_image,
                "tags": tags
            }
        else:
            print("✗ Tag detection failed")
            test_results["tag_detect"] = {
                "success": False,
                "image_path": test_image,
                "tags": []
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
    print(f"✓ Search: {test_results['search'].get('success', False)}")
    print(f"✓ Tag Detect: {test_results['tag_detect'].get('success', False)}")
    
    print(f"\nReport generated: {md_path}")
    print(f"\nOpen the Markdown report to view test results and image previews!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()