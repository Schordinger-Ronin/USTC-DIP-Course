import os

def process_split(split_dir, list_file):
    """
    Process the appending for a single dataset split (train / val)
    """
    if not os.path.exists(split_dir):
        return 0

    # Read the existing list
    existing_files = set()
    if os.path.exists(list_file):
        with open(list_file, 'r', encoding='utf-8') as f:
            existing_files = set(line.strip() for line in f)

    added_count = 0
    # Append new image paths
    with open(list_file, 'a', encoding='utf-8') as f:
        for filename in sorted(os.listdir(split_dir)):
            if filename.endswith('.jpg') and not filename.startswith('._'):
                rel_path = f"{split_dir}/{filename}".replace('//', '/')
                if rel_path not in existing_files:
                    f.write(rel_path + '\n')
                    added_count += 1
    return added_count

def append_dataset_lists(dataset_path):
    """
    Append image paths of the new dataset to the existing training and validation lists
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
    train_added = process_split(f"{dataset_path}/train", './train_list.txt')
    val_added = process_split(f"{dataset_path}/val", './val_list.txt')

def print_dataset_statistics():
    """
    Print the final dataset statistics
    """
    lists = {
        'Training': './train_list.txt',
        'Validation': './val_list.txt'
    }
    
    for name, filepath in lists.items():
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                files = [line.strip() for line in f]
                for path in files[-3:]:
                    print(f" {path}")
        else:
            print(f"\nList file not found: {filepath}")

if __name__ == '__main__':
    # Path to your newly downloaded dataset
    new_dataset_path = './datasets/cityscapes'
    
    try:
        append_dataset_lists(new_dataset_path)
        print_dataset_statistics()
    except Exception as e:
        print(f"Error: {str(e)}")