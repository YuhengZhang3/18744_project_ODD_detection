import json
from collections import Counter
import os
from pathlib import Path

def count_categories(json_paths):
    """
    统计给定JSON文件中所有物体的category_id出现次数。
    json_paths: 列表 包含一个或多个JSON文件路径。
    """
    category_counter = Counter()
    
    for json_path in json_paths:
        print(f"Processing {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 假设JSON是一个列表，每个元素是一张图像的标注
        # 如果JSON是字典且包含特定键，请根据实际情况调整
        if isinstance(data, list):
            images = data
        elif isinstance(data, dict) and 'images' in data:
            images = data['images']  # 有些COCO格式会有'images'键
        else:
            # 如果JSON是单个对象，则包装成列表
            images = [data]
        
        for img in images:
            objects = img.get('objects', [])
            for obj in objects:
                cat = obj.get('category_id')
                if cat:
                    category_counter[cat] += 1
    
    return category_counter

if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    # 项目根目录（scripts的上一级）
    project_root = script_dir.parent
    # 构建JSON文件路径
    train_json = project_root / "data" / "roadwork_traj" / "traj_annotations" / "trajectories_train_equidistant.json"
    val_json = project_root / "data" / "roadwork_traj" / "traj_annotations" / "trajectories_val_equidistant.json"
    
    # 检查文件是否存在
    json_files = []
    for path in [train_json, val_json]:
        if os.path.exists(path):
            json_files.append(path)
        else:
            print(f"Warning: {path} not found.")
    
    if not json_files:
        print("No JSON files found. Please check the paths.")
    else:
        counter = count_categories(json_files)
        
        print("\n===== Category counts =====")
        # 按出现次数降序输出
        for cat, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            print(f"{cat}: {count}")
        
        # 可选：保存到文件
        with open("category_counts.txt", "w", encoding='utf-8') as f:
            for cat, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{cat}: {count}\n")
        print("\nResults saved to category_counts.txt")