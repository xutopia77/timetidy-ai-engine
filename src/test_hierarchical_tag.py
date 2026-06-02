"""
Test Hierarchical Tag Engine with real data and generate markdown report
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engines.hierarchical_tag_engine import HierarchicalTagEngine


def find_image_files(root_dir, extensions=[".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"]):
    """
    Find all image files in directory
    
    Args:
        root_dir: Root directory to search
        extensions: Image file extensions
    
    Returns:
        List of image file paths
    """
    image_files = []
    root_path = Path(root_dir)
    
    for ext in extensions:
        for file_path in root_path.rglob(f"*{ext}"):
            if file_path.is_file():
                image_files.append(str(file_path))
    
    return sorted(image_files)


def get_relative_path(image_path: Path, base_dir: Path) -> str:
    """Get relative path from base directory
    
    Args:
        image_path: Image file path
        base_dir: Base directory path
    
    Returns:
        Relative path string (can include .. for parent directories)
    """
    try:
        if not base_dir.is_dir():
            base_dir = base_dir.parent
        rel_path = os.path.relpath(str(image_path), str(base_dir))
        return rel_path.replace("\\", "/")
    except ValueError:
        return str(image_path.name)


def generate_markdown_report(results, output_path, test_data_dir, project_root):
    """
    Generate markdown report with detailed classification results
    
    Args:
        results: Classification results
        output_path: Output markdown file path
        test_data_dir: Test data directory path
        project_root: Project root directory path
    """
    lines = []
    
    lines.append("# 层次化分类测试报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**测试图片数量**: {len(results)}")
    lines.append(f"**测试数据目录**: `{test_data_dir}`")
    lines.append("")
    
    lines.append("## 目录")
    lines.append("")
    lines.append("- [分类统计](#分类统计)")
    lines.append("- [详细分类结果](#详细分类结果)")
    lines.append("")
    
    lines.append("## 分类统计")
    lines.append("")
    
    level1_counts = Counter([r["level1"]["name"] for r in results])
    lines.append("### 一级分类分布")
    lines.append("")
    lines.append("| 类别 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for name, count in level1_counts.most_common():
        percentage = count / len(results) * 100
        lines.append(f"| {name} | {count} | {percentage:.1f}% |")
    lines.append("")
    
    level2_counts = Counter([r["level2"]["name"] for r in results])
    lines.append("### 二级分类分布")
    lines.append("")
    lines.append("| 类别 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for name, count in level2_counts.most_common():
        percentage = count / len(results) * 100
        lines.append(f"| {name} | {count} | {percentage:.1f}% |")
    lines.append("")
    
    lines.append("### 属性分布")
    lines.append("")
    
    for attr_type in ["time", "weather", "season"]:
        attr_counts = Counter([r["attributes"][attr_type][0]["name"] for r in results])
        lines.append(f"**{attr_type}**:")
        lines.append("")
        lines.append("| 类别 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        for name, count in attr_counts.most_common():
            percentage = count / len(results) * 100
            lines.append(f"| {name} | {count} | {percentage:.1f}% |")
        lines.append("")
    
    avg_l1_score = sum(r["level1"]["score"] for r in results) / len(results)
    avg_l2_score = sum(r["level2"]["score"] for r in results) / len(results)
    avg_combined = sum(r["combined_score"] for r in results) / len(results)
    
    lines.append("### 平均置信度")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|------|------|")
    lines.append(f"| 一级分类 | {avg_l1_score:.4f} |")
    lines.append(f"| 二级分类 | {avg_l2_score:.4f} |")
    lines.append(f"| 综合评分 | {avg_combined:.4f} |")
    lines.append("")
    
    lines.append("## 详细分类结果")
    lines.append("")
    
    for i, result in enumerate(results, 1):
        image_path = Path(result["image_path"])
        
        md_file_dir = Path(output_path).parent
        rel_path_from_md = get_relative_path(image_path, md_file_dir)
        
        lines.append(f"### {i}. {image_path.name}")
        lines.append("")
        lines.append(f"![{image_path.name}]({rel_path_from_md})")
        lines.append("")
        lines.append(f"**路径**: `{rel_path_from_md}`")
        lines.append("")
        
        l1 = result["level1"]
        lines.append(f"**一级分类**: {l1['name']} (相似度: {l1['score']:.4f})")
        lines.append("")
        
        l2 = result["level2"]
        lines.append(f"**二级分类**: {l2['name']} (相似度: {l2['score']:.4f})")
        lines.append("")
        
        lines.append(f"**综合评分**: {result['combined_score']:.4f}")
        lines.append("")
        
        lines.append("**属性标签**:")
        lines.append("")
        lines.append("| 类型 | 标签 | 相似度 |")
        lines.append("|------|------|--------|")
        for attr_type, attrs in result["attributes"].items():
            best = attrs[0]
            lines.append(f"| {attr_type} | {best['name']} | {best['score']:.4f} |")
        lines.append("")
        
        if len(result["all_candidates"]) > 1:
            lines.append("**候选路径分析**:")
            lines.append("")
            lines.append("| 排名 | 一级分类 | 二级分类 | 综合评分 |")
            lines.append("|------|----------|----------|----------|")
            for j, path in enumerate(result["all_candidates"], 1):
                l1 = path["level1"]
                l2 = path["level2"]
                score = path["combined_score"]
                marker = " <-- 最佳" if j == 1 else ""
                lines.append(f"| {j} | {l1['name']} | {l2['name']} | {score:.4f}{marker} |")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\n✓ Markdown report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Hierarchical Tag Engine with real images"
    )
    parser.add_argument(
        "--num-images", 
        type=int, 
        default=5,
        help="Number of images to process (default: 5)"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=r"d:\work\56-time-tidy-prj\test-data\data-min-real",
        help="Test data directory path"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Hierarchical Tag Engine Test")
    print("="*80)
    
    test_data_dir = args.test_dir
    project_root = Path(__file__).parent.parent
    
    if not os.path.exists(test_data_dir):
        print(f"\n✗ Test data directory not found: {test_data_dir}")
        return
    
    print(f"\nTest data directory: {test_data_dir}")
    print(f"Number of images to process: {args.num_images}")
    
    image_files = find_image_files(test_data_dir)
    
    if not image_files:
        print(f"\n✗ No image files found in {test_data_dir}")
        return
    
    total_found = len(image_files)
    image_files = image_files[:args.num_images]
    
    print(f"\nFound {total_found} image files, processing {len(image_files)} images")
    
    engine = HierarchicalTagEngine(top_k=3)
    
    results = []
    errors = []
    
    print(f"\nProcessing images...")
    
    for idx, image_path in enumerate(image_files, 1):
        try:
            print(f"  [{idx}/{len(image_files)}] Processing: {Path(image_path).name}")
            
            result = engine.process({
                "action": "classify_hierarchical",
                "image_path": image_path,
            })
            
            results.append(result)
            
        except Exception as e:
            print(f"  ✗ Error processing {image_path}: {e}")
            errors.append({
                "image_path": image_path,
                "error": str(e),
            })
    
    print(f"\n✓ Processed {len(results)} images successfully")
    if errors:
        print(f"✗ {len(errors)} images failed")
    
    output_dir = project_root / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = output_dir / f"hierarchical_tag_test_{timestamp}.md"
    
    generate_markdown_report(results, md_path, test_data_dir, project_root)
    
    json_path = output_dir / f"hierarchical_tag_test_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "test_data_dir": test_data_dir,
            "total_images": len(results),
            "results": results,
            "errors": errors,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON results saved: {json_path}")
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"\n✓ Total images: {len(image_files)}")
    print(f"✓ Successfully processed: {len(results)}")
    print(f"✗ Failed: {len(errors)}")
    print(f"\nReports generated:")
    print(f"  - Markdown: {md_path}")
    print(f"  - JSON: {json_path}")
    print(f"\nOpen the markdown report to view results with image previews!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()