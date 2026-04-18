import pandas as pd

def count_epochs_pandas(file_path, chunk_size=500000, range_start=1, range_end=10):
    target_values = [0, 1, 2]
    counts = {val: 0 for val in target_values}

    if range_start < 1 or range_end < range_start:
        raise ValueError("range_start/range_end 参数非法，要求 1 <= range_start <= range_end")

    # 分别保存每个 epoch 的前 N 行
    preview_chunks = {val: [] for val in target_values}
    preview_counts = {val: 0 for val in target_values}
    preview_limit = range_end

    try:
        # 读取完整行，便于输出指定区间的内容
        reader = pd.read_csv(file_path, chunksize=chunk_size)

        for i, chunk in enumerate(reader):
            if 'epoch' not in chunk.columns:
                raise KeyError

            matched = chunk[chunk['epoch'].isin(target_values)]
            val_counts = matched['epoch'].value_counts()

            for val in target_values:
                counts[val] += int(val_counts.get(val, 0))

                if preview_counts[val] < preview_limit:
                    need = preview_limit - preview_counts[val]
                    part = chunk[chunk['epoch'] == val].head(need)
                    if not part.empty:
                        preview_chunks[val].append(part)
                        preview_counts[val] += len(part)

            print(f"已处理第 {i + 1} 个数据块...")

        total_rows = sum(counts.values())
        print("\n最终统计结果：")
        print(f"epoch 为 0/1/2 的行总数: {total_rows}")
        for val in target_values:
            print(f"epoch {val}: {counts[val]} 行")

        print(f"\n分别显示 epoch 为 0/1/2 的第 {range_start}-{range_end} 行内容：")
        for val in target_values:
            print(f"\nepoch {val} 第 {range_start}-{range_end} 行：")
            if preview_chunks[val]:
                preview_df = pd.concat(preview_chunks[val], ignore_index=True)
                if len(preview_df) >= range_start:
                    ranged_df = preview_df.iloc[range_start - 1:range_end]
                    print(ranged_df.to_string(index=False))
                else:
                    print(f"epoch {val} 的匹配行不足 {range_start} 行，实际只有 {len(preview_df)} 行。")
            else:
                print(f"未找到 epoch 为 {val} 的行。")

    except FileNotFoundError:
        print("文件未找到，请检查路径。")
    except KeyError:
        print("CSV 文件中不存在列名 'epoch'。")

# 使用示例
count_epochs_pandas('/root/workspace/bishe/mgpusim-4.1.1/amd/samples/minerva/mem.csv')