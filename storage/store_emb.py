import torch
import sys
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader, Subset

# 确保从任意工作目录或动态导入时都能找到项目模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data_provider.data_loader_save import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from gen_prompt_emb import GenPromptEmb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--output_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--l_layers", type=int, default=12)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--divide", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=min(10, os.cpu_count()))
    parser.add_argument("--embed_version", type=str, default="original",
                        help="嵌入版本标识，用于区分不同版本生成的嵌入（如 'original', 'wavelet', 'gpt2'）")
    parser.add_argument("--start_idx", type=int, default=0, help="当前分片起始样本索引（含）")
    parser.add_argument("--end_idx", type=int, default=-1, help="当前分片结束样本索引（含），-1 表示到最后")
    parser.add_argument("--skip_existing", action="store_true", help="目标文件已存在时跳过")
    parser.add_argument("--indices_file", type=str, default="", help="按文件指定精确样本索引，每行一个整数")
    return parser.parse_args()

def get_dataset(data_path, flag, input_len, output_len):
    datasets = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    dataset_class = datasets.get(data_path, Dataset_Custom)
    return dataset_class(flag=flag, size=[input_len, 0, output_len], data_path=data_path)

def save_embeddings(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_set = get_dataset(args.data_path, 'train', args.input_len, args.output_len)
    test_set = get_dataset(args.data_path, 'test', args.input_len, args.output_len)
    val_set = get_dataset(args.data_path, 'val', args.input_len, args.output_len)

    datasets = {
        'train': train_set,
        'test': test_set,
        'val': val_set
    }
    full_dataset = datasets[args.divide]
    total_samples = len(full_dataset)

    if args.indices_file:
        with open(args.indices_file, "r", encoding="utf-8") as f:
            subset_indices = [int(line.strip()) for line in f if line.strip()]
        if not subset_indices:
            print(f"No indices found in {args.indices_file}, nothing to do.")
            return
        start_idx = subset_indices[0]
        end_idx = subset_indices[-1]
    else:
        start_idx = max(0, args.start_idx)
        end_idx = total_samples - 1 if args.end_idx < 0 else min(args.end_idx, total_samples - 1)
        if start_idx > end_idx:
            raise ValueError(
                f"Invalid shard range for {args.divide}: start_idx={start_idx}, end_idx={end_idx}, total={total_samples}"
            )
        subset_indices = list(range(start_idx, end_idx + 1))

    data_subset = Subset(full_dataset, subset_indices)

    data_loader = DataLoader(
        data_subset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    gen_prompt_emb = GenPromptEmb(
        device=device, # type: ignore
        input_len=args.input_len,
        data_path=args.data_path,
        model_name=args.model_name,
        d_model=args.d_model,
        layer=args.l_layers,
        divide=args.divide
    ).to(device)

    # 创建保存目录（添加版本标识）
    save_path = f"./Embeddings/{args.data_path}/{args.embed_version}/{args.divide}/"
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving embeddings to: {save_path}")
    print(f"Embedding version: {args.embed_version}")
    print(f"Shard range: [{start_idx}, {end_idx}] / {total_samples}")

    emb_time_path = f"./Results/emb_logs/"
    os.makedirs(emb_time_path, exist_ok=True)

    for i, (x, y, x_mark, y_mark) in enumerate(data_loader):
        global_idx = subset_indices[i]
        embeddings = gen_prompt_emb.generate_embeddings(x.to(device), x_mark.to(device))

        file_path = f"{save_path}{global_idx}.h5"
        if args.skip_existing and os.path.exists(file_path):
            continue
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('embeddings', data = embeddings.cpu().numpy())

        # # Save and visualize the first sample
        # if i >= 0:
        #     break
    
if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    save_embeddings(args)
    t2 = time.time()
    print(f"Total time spent: {(t2 - t1)/60:.4f} minutes")
