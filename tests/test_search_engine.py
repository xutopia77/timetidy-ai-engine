"""
Test SearchEngine text-to-image search functionality and generate markdown report
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from engines.search_engine import SearchEngine


def get_relative_path(image_path: Path, base_dir: Path) -> str:
    """Get relative path from base directory
    
    Args:
        image_path: Image file path
        base_dir: Base directory path
    
    Returns:
        Relative path string with forward slashes
    """
    try:
        if not base_dir.is_dir():
            base_dir = base_dir.parent
        rel_path = os.path.relpath(str(image_path), str(base_dir))
        return rel_path.replace("\\", "/")
    except ValueError:
        return str(image_path.name)


def find_image_files(root_dir, extensions=[".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"]):
    """Find all image files in directory
    
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


def load_search_queries(config_path: str) -> dict:
    """Load search queries from JSON config file
    
    Args:
        config_path: Path to search_queries.json
    
    Returns:
        Dictionary with queries and settings
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_markdown_report(all_results, output_path, test_data_dir, queries):
    """Generate markdown report for text-to-image search test
    
    Args:
        all_results: Dictionary of search results for each query
        output_path: Output markdown file path
        test_data_dir: Test data directory path
        queries: List of search queries
    """
    lines = []
    
    lines.append("# 文搜图测试报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**测试目录**: `{test_data_dir}`")
    lines.append(f"**搜索查询数**: {len(queries)}")
    lines.append("")
    
    lines.append("## 搜索配置")
    lines.append("")
    lines.append("| 查询文本 | 描述 | Top-K |")
    lines.append("|----------|------|-------|")
    for query in queries:
        lines.append(f"| {query['text']} | {query.get('description', '-')} | {query.get('top_k', 5)} |")
    lines.append("")
    
    lines.append("## 搜索结果")
    lines.append("")
    
    for query_text, results in all_results.items():
        query_config = next((q for q in queries if q['text'] == query_text), None)
        description = query_config.get('description', '') if query_config else ''
        
        lines.append(f"### 查询: \"{query_text}\"")
        lines.append("")
        if description:
            lines.append(f"**描述**: {description}")
            lines.append("")
        
        if not results:
            lines.append("*未找到匹配的图像*")
            lines.append("")
            continue
        
        lines.append(f"**找到 {len(results)} 个结果**")
        lines.append("")
        
        lines.append("| 排名 | 图像 | 相似度 | 预览 |")
        lines.append("|------|------|--------|------|")
        
        for i, result in enumerate(results, 1):
            image_name = Path(result["file_path"]).name
            md_file_dir = Path(output_path).parent
            rel_path = get_relative_path(Path(result["file_path"]), md_file_dir)
            similarity = result["similarity"]
            
            lines.append(f"| {i} | [{image_name}]({rel_path}) | {similarity:.4f} | ![]({rel_path}) |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    lines.append("## 详细结果")
    lines.append("")
    
    for query_text, results in all_results.items():
        if not results:
            continue
        
        lines.append(f"### 查询: \"{query_text}\"")
        lines.append("")
        
        for i, result in enumerate(results, 1):
            image_path = Path(result["file_path"])
            md_file_dir = Path(output_path).parent
            rel_path = get_relative_path(image_path, md_file_dir)
            
            lines.append(f"#### {i}. {image_path.name}")
            lines.append("")
            lines.append(f"![{image_path.name}]({rel_path})")
            lines.append("")
            lines.append(f"**路径**: `{rel_path}`")
            lines.append("")
            lines.append(f"**相似度**: {result['similarity']:.4f}")
            lines.append("")
            lines.append("---")
            lines.append("")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\n✓ Markdown 报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="测试搜索引擎的文搜图功能"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="搜索查询配置文件路径 (默认: tests/search_queries.json)"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=r"d:\work\56-time-tidy-prj\test-data\data-min-real",
        help="测试图像目录路径"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="最大搜索图像数量 (覆盖配置文件设置)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("搜索引擎文搜图测试")
    print("="*80)
    
    project_root = Path(__file__).parent.parent
    tests_dir = Path(__file__).parent
    
    config_path = args.config
    if config_path is None:
        config_path = tests_dir / "search_queries.json"
    
    if not os.path.exists(config_path):
        print(f"\n✗ 配置文件不存在: {config_path}")
        print(f"请创建配置文件，示例格式:")
        print(json.dumps({
            "queries": [
                {"text": "风景", "description": "搜索风景图片", "top_k": 5}
            ],
            "settings": {"max_images": 100}
        }, ensure_ascii=False, indent=2))
        return
    
    config = load_search_queries(config_path)
    queries = config.get("queries", [])
    
    if not queries:
        print(f"\n✗ 配置文件中没有搜索查询")
        return
    
    test_data_dir = args.test_dir
    if not os.path.exists(test_data_dir):
        print(f"\n✗ 测试图像目录不存在: {test_data_dir}")
        return
    
    print(f"\n配置文件: {config_path}")
    print(f"测试目录: {test_data_dir}")
    print(f"搜索查询数: {len(queries)}")
    
    max_images = args.max_images
    if max_images is None:
        max_images = config.get("settings", {}).get("max_images", 100)
    print(f"最大图像数: {max_images}")
    
    image_files = find_image_files(test_data_dir)
    
    if not image_files:
        print(f"\n✗ 未找到图像文件: {test_data_dir}")
        return
    
    total_found = len(image_files)
    image_files = image_files[:max_images]
    
    print(f"\n找到 {total_found} 个图像文件，使用 {len(image_files)} 个进行搜索")
    
    engine = SearchEngine()
    
    all_results = {}
    
    print(f"\n开始搜索...")
    
    for query in queries:
        query_text = query["text"]
        top_k = query.get("top_k", 5)
        
        print(f"\n查询: \"{query_text}\" (Top-{top_k})")
        
        try:
            result = engine.process({
                "action": "search_by_text",
                "text": query_text,
                "file_paths": image_files,
                "top_k": top_k,
            })
            
            search_results = result.get("results", [])
            all_results[query_text] = search_results
            
            print(f"  找到 {len(search_results)} 个结果")
            for i, res in enumerate(search_results, 1):
                image_name = Path(res["file_path"]).name
                similarity = res["similarity"]
                print(f"    {i}. {image_name} (相似度: {similarity:.4f})")
            
        except Exception as e:
            print(f"  ✗ 搜索失败: {e}")
            all_results[query_text] = []
    
    output_dir = project_root / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = output_dir / f"search_engine_test_{timestamp}.md"
    
    generate_markdown_report(all_results, md_path, test_data_dir, queries)
    
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"\n✓ 搜索查询数: {len(queries)}")
    for query_text, results in all_results.items():
        print(f"✓ \"{query_text}\": {len(results)} 个结果")
    print(f"\n报告已生成:")
    print(f"  - Markdown: {md_path}")
    print(f"\n打开 Markdown 报告查看搜索结果和图像预览！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()