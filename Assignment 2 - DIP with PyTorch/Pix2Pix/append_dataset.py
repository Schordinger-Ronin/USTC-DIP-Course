import os

def process_split(split_dir, list_file):
    """
    处理单个数据集划分（train / val）的追加
    """
    # 如果该文件夹不存在，直接跳过并返回 0
    if not os.path.exists(split_dir):
        return 0

    # 1. 读取现有列表（使用 utf-8 编码匹配终端生成的文件）
    existing_files = set()
    if os.path.exists(list_file):
        with open(list_file, 'r', encoding='utf-8') as f:
            existing_files = set(line.strip() for line in f)

    added_count = 0
    # 2. 追加新的图片路径
    with open(list_file, 'a', encoding='utf-8') as f:
        for filename in sorted(os.listdir(split_dir)):
            # 【Mac 专属优化】：除了确认是 .jpg 结尾，还要排除 macOS 特有的 '._' 隐藏文件
            if filename.endswith('.jpg') and not filename.startswith('._'):
                # 构造相对路径，与原有的 txt 文件格式保持一致
                rel_path = f"{split_dir}/{filename}".replace('//', '/')
                
                # 去重判定：只有当文件路径不在现有列表中时才写入
                if rel_path not in existing_files:
                    f.write(rel_path + '\n')
                    added_count += 1
                    
    return added_count

def append_dataset_lists(dataset_path):
    """
    将新数据集的图片路径追加到现有的训练和验证列表中
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    print(f"正在扫描 {dataset_path} ...")
    
    # 仅处理 train 和 val
    train_added = process_split(f"{dataset_path}/train", './train_list.txt')
    val_added = process_split(f"{dataset_path}/val", './val_list.txt')
    
    print(f"追加完成！新增 Train: {train_added} 张, Val: {val_added} 张。")

def print_dataset_statistics():
    """
    打印数据集最终的统计信息
    """
    lists = {
        'Training': './train_list.txt',
        'Validation': './val_list.txt'
    }
    
    print("\n=== 更新后的数据集统计 ===")
    for name, filepath in lists.items():
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                files = [line.strip() for line in f]
            print(f"\n{name} 集合:")
            print(f"总图片数: {len(files)}")
            print("末尾 3 张样本路径 (用来确认新数据已成功加入):")
            for path in files[-3:]:
                print(f"  {path}")
        else:
            print(f"\n找不到清单文件: {filepath}")

if __name__ == '__main__':
    # 你新下载的数据集路径
    new_dataset_path = './datasets/cityscapes'
    
    try:
        append_dataset_lists(new_dataset_path)
        print_dataset_statistics()
    except Exception as e:
        print(f"Error: {str(e)}")