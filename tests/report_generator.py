"""
Markdown Report Generator for AI Engine Tests

This module generates markdown test reports with image previews.
"""

import os
from pathlib import Path
from typing import Dict


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
    lines.append(f"**Test Data**: `{test_data_dir}`")
    lines.append("")
    # Test Summary
    lines.append("## Test Summary")
    lines.append("")
    
    ping_result = test_results.get("ping", {})
    index_init_result = test_results.get("index_init", {})
    search_results = test_results.get("search", {})
    tag_results = test_results.get("tag_detect", {})
    sync_result = test_results.get("sync", {})
    find_similar_result = test_results.get("find_similar", {})
    detect_groups_result = test_results.get("detect_similar_groups", {})
    
    lines.append("| Test | Status | Details |")
    lines.append("|------|--------|---------|")
    
    ping_status = "✓ Pass" if ping_result.get("code") == 0 else "✗ Fail"
    lines.append(f"| Ping | {ping_status} | {ping_result.get('message', '')} |")
    
    index_init_status = "✓ Pass" if index_init_result.get("code") == 0 else "✗ Fail"
    total_images = index_init_result.get("data", {}).get("total_images", 0)
    build_time = index_init_result.get("data", {}).get("build_time_seconds", 0)
    processed = index_init_result.get("data", {}).get("processed_files", 0)
    skipped = index_init_result.get("data", {}).get("skipped_files", 0)
    lines.append(f"| Index Init | {index_init_status} | {processed} processed, {skipped} skipped, {total_images} total |")
    
    search_status = "✓ Pass" if search_results.get("code") == 0 else "✗ Fail"
    search_count = len(search_results.get("results", []))
    query_time = search_results.get("query_time_ms", 0)
    lines.append(f"| Search | {search_status} | {search_count} results in {query_time}ms |")
    
    tag_status = "✓ Pass" if tag_results.get("code") == 0 else "✗ Fail"
    success_count = tag_results.get("success_count", 0)
    total_count = tag_results.get("total_count", 0)
    total_tags = tag_results.get("total_tags", 0)
    lines.append(f"| Tag Detect | {tag_status} | {success_count}/{total_count} images, {total_tags} tags |")
    
    sync_status = "✓ Pass" if sync_result.get("code") == 0 else "✗ Fail"
    processed_files = sync_result.get("data", {}).get("processed_files", 0)
    skipped_files = sync_result.get("data", {}).get("skipped_files", 0)
    total_files = sync_result.get("data", {}).get("total_files", 0)
    lines.append(f"| Sync All | {sync_status} | {processed_files} processed, {skipped_files} skipped, {total_files} total |")
    
    find_similar_status = "✓ Pass" if find_similar_result.get("code") == 0 else "✗ Fail"
    similar_count = len(find_similar_result.get("similar_images", []))
    similar_time = find_similar_result.get("query_time_ms", 0)
    lines.append(f"| Find Similar | {find_similar_status} | {similar_count} similar images in {similar_time}ms |")
    
    detect_groups_status = "✓ Pass" if detect_groups_result.get("code") == 0 else "✗ Fail"
    total_groups = detect_groups_result.get("total_groups", 0)
    total_pairs = detect_groups_result.get("total_similar_pairs", 0)
    detect_time = detect_groups_result.get("query_time_ms", 0)
    lines.append(f"| Detect Similar Groups | {detect_groups_status} | {total_groups} groups, {total_pairs} pairs in {detect_time}ms |")
    
    lines.append("")
    # Index Initialization
    if index_init_result.get("code") == 0:
        lines.append("## Index Initialization")
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

    elif index_init_result.get("code", 1) != 0:
        lines.append("## Index Initialization")
        lines.append("")
        lines.append(f"- **Error**: {index_init_result.get('status', 'Unknown error')}")
        lines.append("")
        
    # Search Results
    if search_results.get("code") == 0 and search_results.get("results"):
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
        
        if results and record_id_to_path:
            md_file_dir = Path(output_path).parent
            
            lines.append("| Info | Image |")
            lines.append("|------|-------|")
            
            for i, result in enumerate(results, 1):
                record_id = result.get("record_id", 0)
                similarity = result.get("similarity", 0)
                rank = result.get("rank", 0)
                image_path = record_id_to_path.get(record_id, "")
                
                if image_path:
                    rel_path = get_relative_path(Path(image_path), md_file_dir)
                    image_name = Path(image_path).name
                    
                    info_lines = [
                        f"**Rank**: {rank}",
                        f"**Image**: {image_name}",
                        f"**Record ID**: {record_id}",
                        f"**Similarity**: {similarity:.4f}"
                    ]
                    info_text = "<br>".join(info_lines)
                    
                    lines.append(f"| {info_text} | ![{image_name}]({rel_path}) |")
            
            if len(results) > 10:
                lines.append(f"| ... and {len(results) - 10} more results | |")
            
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    if tag_results.get("code") == 0:
        lines.append("## Tag Detection Results")
        lines.append("")
        
        success_count = tag_results.get("success_count", 0)
        total_count = tag_results.get("total_count", 0)
        total_tags = tag_results.get("total_tags", 0)
        results_list = tag_results.get("results", [])
        
        lines.append(f"**Summary**: {success_count}/{total_count} images processed, {total_tags} tags generated")
        lines.append("")
        
        md_file_dir = Path(output_path).parent
        
        for idx, result in enumerate(results_list, 1):
            file_path = result.get("file_path", "")
            result_code = result.get("code", 1)
            
            if result_code == 0:
                tags = result.get("tags", [])
                level1_tags = result.get("level1_tags", [])
                level2_tags = result.get("level2_tags", [])
                attribute_tags = result.get("attribute_tags", [])
                process_time_ms = result.get("process_time_ms", 0)
                
                lines.append(f"### Image {idx}: {Path(file_path).name}")
                lines.append("")
                
                if file_path:
                    rel_path = get_relative_path(Path(file_path), md_file_dir)
                    lines.append(f"<img src=\"{rel_path}\" width=\"25%\">")
                    lines.append("")
                
                tag_count = len(tags)
                lines.append("```")
                lines.append(f"**Generated {tag_count} tags in {process_time_ms}ms**:")
                lines.append("")
                
                # Show simplified tags
                if tags:
                    lines.append("**Tags**")
                    lines.append("")
                    for tag in tags[:10]:
                        label = tag.get("label", "")
                        confidence = tag.get("confidence", 0.0)
                        lines.append(f"- {label} (confidence: {confidence:.4f})")
                
                if level1_tags:
                    lines.append("**Level 1 - Primary Category**")
                    for tag in level1_tags[:3]:
                        name = tag.get("name", "")
                        confidence = tag.get("confidence", 0.0)
                        lines.append(f"- confidence:{confidence:.4f}, Category:{name}")
                    lines.append("")
                
                if level2_tags:
                    lines.append("**Level 2 - Sub-category**")
                    for tag in level2_tags[:5]:
                        name = tag.get("name", "")
                        confidence = tag.get("confidence", 0.0)
                        parent = tag.get("parent_id", "")
                        lines.append(f"- Confidence:{confidence:.4f}, Parent:{parent}, Sub-category:{name}")
                    lines.append("")
                
                if attribute_tags:
                    lines.append("**Attributes**")
                    for tag in attribute_tags[:5]:
                        name = tag.get("name", "")
                        confidence = tag.get("confidence", 0.0)
                        attr_type = tag.get("type", "")
                        lines.append(f"- Confidence:{confidence:.4f}, Type:{attr_type}, Attribute:{name}")
                lines.append("```")
                lines.append("")
            else:
                lines.append(f"### Image {idx}: {Path(file_path).name}")
                lines.append("")
                lines.append(f"**Status**: ✗ Failed - {result.get('status', 'Unknown error')}")
                lines.append("")
                lines.append("---")
                lines.append("")
    
    if sync_result.get("code") == 0:
        lines.append("## Data Synchronization")
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
    elif sync_result.get("code", 1) != 0:
        lines.append("## Data Synchronization")
        lines.append("")
        lines.append(f"- **Error**: {sync_result.get('status', 'Unknown error')}")
        lines.append("")
    
    if find_similar_result.get("code") == 0:
        lines.append("## Find Similar Images (1-to-N)")
        lines.append("")
        
        reference_id = find_similar_result.get("reference_id", 0)
        similar_images = find_similar_result.get("similar_images", [])
        query_time = find_similar_result.get("query_time_ms", 0)
        record_id_to_path = find_similar_result.get("record_id_to_path", {})
        
        lines.append(f"### Reference Image: ID {reference_id}")
        lines.append("")
        lines.append(f"**Query time**: {query_time}ms")
        lines.append("")
        lines.append(f"**Found {len(similar_images)} similar images**:")
        lines.append("")
        
        if similar_images:
            lines.append("| Rank | Image | Record ID | Similarity |")
            lines.append("|------|-------|-----------|------------|")
            
            for result in similar_images:
                rank = result.get("rank", 0)
                record_id = result.get("record_id", 0)
                similarity = result.get("similarity", 0)
                image_path = record_id_to_path.get(record_id, "")
                image_name = Path(image_path).name if image_path else f"ID_{record_id}"
                lines.append(f"| {rank} | {image_name} | {record_id} | {similarity:.4f} |")
            
            lines.append("")
        else:
            lines.append("No similar images found.")
            lines.append("")
    
    elif find_similar_result.get("code", 1) != 0:
        lines.append("## Find Similar Images")
        lines.append("")
        lines.append(f"- **Error**: {find_similar_result.get('status', 'Unknown error')}")
        lines.append("")
    
    if detect_groups_result.get("code") == 0:
        lines.append("## Detect Similar Groups (N-to-N)")
        lines.append("")
        
        total_images = detect_groups_result.get("total_images", 0)
        total_groups = detect_groups_result.get("total_groups", 0)
        total_pairs = detect_groups_result.get("total_similar_pairs", 0)
        groups = detect_groups_result.get("groups", [])
        query_time = detect_groups_result.get("query_time_ms", 0)
        record_id_to_path = detect_groups_result.get("record_id_to_path", {})
        
        lines.append("### Summary")
        lines.append("")
        lines.append(f"- **Total images**: {total_images}")
        lines.append(f"- **Total groups**: {total_groups}")
        lines.append(f"- **Total similar pairs**: {total_pairs}")
        lines.append(f"- **Query time**: {query_time}ms")
        lines.append("")
        
        lines.append("### Algorithm")
        lines.append("")
        lines.append("```")
        lines.append("1. Retrieve all vectors from FAISS index")
        lines.append("2. Batch search: each vector searches top-K neighbors")
        lines.append("3. Filter: keep pairs with similarity >= min_similarity")
        lines.append("4. Deduplicate: keep only pairs where id_A < id_B")
        lines.append("5. Group: use Union-Find for transitive grouping")
        lines.append("   - A≈B, B≈C → A,B,C same group")
        lines.append("```")
        lines.append("")
        
        if groups:
            lines.append("### Similar Groups (Sorted by Pair Count)")
            lines.append("")
            
            for group in groups[:10]:
                group_id = group.get("group_id", 0)
                images = group.get("images", [])
                avg_sim = group.get("avg_similarity", 0)
                pair_count = group.get("pair_count", 0)
                
                lines.append(f"#### Group {group_id}")
                lines.append("")
                lines.append(f"- **Images**: {len(images)}")
                lines.append(f"- **Average similarity**: {avg_sim:.4f}")
                lines.append(f"- **Pair count**: {pair_count}")
                lines.append("")
                
                lines.append("| Info | Image |")
                lines.append("|------|-------|")
                
                for img in images[:10]:
                    record_id = img.get("record_id", 0)
                    is_rep = img.get("representative", False)
                    sim_to_rep = img.get("similarity_to_representative", 1.0)
                    image_path = record_id_to_path.get(record_id, "")
                    image_name = Path(image_path).name if image_path else f"ID_{record_id}"
                    
                    role = "Representative" if is_rep else "Member"
                    sim_str = "-" if is_rep else f"{sim_to_rep:.4f}"
                    
                    info_lines = [
                        f"**Role**: {role}",
                        f"**Image**: {image_name}",
                        f"**Record ID**: {record_id}",
                        f"**Similarity to Rep**: {sim_str}"
                    ]
                    info_text = "<br>".join(info_lines)
                    
                    if image_path:
                        rel_path = get_relative_path(Path(image_path), md_file_dir)
                        lines.append(f"| {info_text} | ![{image_name}]({rel_path}) |")
                    else:
                        lines.append(f"| {info_text} | - |")
                
                lines.append("")
            
            if len(groups) > 10:
                lines.append(f"*... and {len(groups) - 10} more groups*")
                lines.append("")
        
        else:
            lines.append("### No Similar Groups Found")
            lines.append("")
            lines.append("No groups with 2+ similar images were detected.")
            lines.append("")
    
    elif detect_groups_result.get("code", 1) != 0:
        lines.append("## Detect Similar Groups")
        lines.append("")
        lines.append(f"- **Error**: {detect_groups_result.get('error', 'Unknown error')}")
        lines.append("")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\n✓ Markdown report generated: {output_path}")
