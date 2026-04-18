import argparse
import pandas as pd
import os

# 配置输入文件路径
INPUT_FILE = '/root/workspace/bishe/mgpusim-4.1.1/amd/samples/minerva/mem.csv'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process rows for a specific epoch and calculate page_vnum offsets.'
    )
    parser.add_argument(
        '--input',
        default=INPUT_FILE,
        help='Path to input CSV file.'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=2,
        help='Target epoch value to process.'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Path to output CSV file. Defaults to epoch{epoch}_processed.csv.'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    input_file = args.input
    target_epoch = args.epoch
    output_file = args.output or f'epoch{target_epoch}_processed.csv'

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return

    print(f"Reading {input_file}...")
    
    # 读取CSV文件
    # 假设文件有表头，如果没有需调整header参数
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 检查必要的列是否存在
    required_columns = ['epoch', 'page_vnum']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV must contain columns: {required_columns}")
        return

    # 筛选指定 epoch 的数据
    target_epoch_df = df[df['epoch'] == target_epoch].copy()

    if target_epoch_df.empty:
        print(f"No data found for epoch = {target_epoch}.")
        return

    print(f"Found {len(target_epoch_df)} rows for epoch {target_epoch}.")

    #获取第一行的 page_vnum 作为基准
    # 注意：这里假设通过 epoch 筛选后的数据顺序即为原始顺序
    # 如果需要按时间或其他字段排序，请先排序：target_epoch_df = target_epoch_df.sort_values(by='time')
    
    first_row_vnum_hex = target_epoch_df.iloc[0]['page_vnum']
    
    try:
        base_vnum = int(first_row_vnum_hex, 16)
        print(f"Base page_num (index 0): {first_row_vnum_hex} ({base_vnum})")
    except ValueError:
        print(f"Error: Invalid hex format for base page_num: {first_row_vnum_hex}")
        return

    def calculate_index(hex_vnum):
        try:
            current_vnum = int(hex_vnum, 16)
            # 计算序号: (current - base)
            idx = (current_vnum - base_vnum)
            return idx
        except ValueError:
            return None

    # 应用计算
    target_epoch_df['calculated_index'] = target_epoch_df['page_vnum'].apply(calculate_index)

    # 检查是否有计算失败的行
    if target_epoch_df['calculated_index'].isnull().any():
        print("Warning: Some rows contains invalid page_num and index could not be calculated.")

    # 将结果保存到 CSV
    # 只需要 output page_vnum and calculated_index
    output_df = target_epoch_df[['page_vnum', 'calculated_index']]
    
    output_df.to_csv(output_file, index=False)
    print(f"Processing complete. Result saved to {output_file}")

if __name__ == "__main__":
    main()
