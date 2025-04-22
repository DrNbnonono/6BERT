import ipaddress
from typing import List, Dict, Tuple, Set
import numpy as np
import pandas as pd
import logging
import os
import json
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import time

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_address_chunk(addresses: List[str], position: int = 0) -> Tuple[Set[str], List[str], Dict]:
    """处理一批IPv6地址，返回词汇集合、词序列和统计信息"""
    vocab = set()
    word_sequences = []
    stats = {"地址": [], "长度": [], "前缀": []}
    
    # 使用与原论文一致的位置编码
    location_alpha = '0123456789abcdefghijklmnopqrstuv'
    
    for addr in addresses:
        try:
            # 直接处理地址，移除冒号
            ip = ipaddress.ip_address(addr.strip())
            exploded = ip.exploded.replace(":", "")
            
            # 确保地址长度为32个十六进制数字
            if len(exploded) != 32:
                exploded = exploded.zfill(32)
            
            # 构建词汇 - 使用原论文的方式
            words = []
            for pos, nybble in enumerate(exploded):
                if pos < len(location_alpha):  # 确保位置在有效范围内
                    word = f"{nybble}{location_alpha[pos]}"
                    vocab.add(word)
                    words.append(word)
            
            # 为BERT模型添加特殊标记
            words = ["[CLS]"] + words + ["[SEP]"]
            word_sequences.append(" ".join(words) + "\n")
            
            # 收集统计信息
            stats["地址"].append(addr)
            stats["长度"].append(len(exploded))
            stats["前缀"].append(exploded[:8])
        except Exception as e:
            logging.warning(f"处理地址时出错: {addr}, 错误: {e}")
    
    return vocab, word_sequences, stats

def build_vocabulary(all_vocabs: List[Set[str]]) -> Dict[str, int]:
    """从多个词汇集合构建统一的词汇表"""
    # 合并所有词汇集合
    combined_vocab = set()
    for vocab_set in all_vocabs:
        combined_vocab.update(vocab_set)
    
    # 添加BERT特殊标记
    special_tokens = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    word2id = {word: i+len(special_tokens) for i, word in enumerate(sorted(combined_vocab))}
    word2id.update(special_tokens)
    
    logging.info(f"构建词汇表完成，大小: {len(word2id)}")
    return word2id

def process_to_words(input_path: str, output_path: str, stats_path: str = None, 
                     chunk_size: int = 100000, num_workers: int = None) -> Tuple[Dict[str, int], pd.DataFrame]:
    """将原始地址转换为词序列文件，返回词汇表和统计信息，使用并行处理"""
    start_time = time.time()
    logging.info(f"从 {input_path} 读取地址...")
    
    # 确定CPU核心数
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # 保留一个核心给系统
    
    logging.info(f"使用 {num_workers} 个工作进程进行并行处理")
    
    # 计算文件总行数以初始化进度条
    total_lines = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    logging.info(f"文件包含 {total_lines} 行数据")
    
    # 创建进程池
    pool = mp.Pool(processes=num_workers)
    
    # 准备结果收集器
    all_vocabs = []
    all_word_sequences = []
    all_stats = {"地址": [], "长度": [], "前缀": []}
    
    # 分块读取文件并并行处理
    with open(input_path, 'r', encoding='utf-8') as f:
        # 使用tqdm创建总体进度条
        pbar = tqdm(total=total_lines, desc="处理IPv6地址", unit="地址")
        
        chunk = []
        chunk_position = 0
        
        for line in f:
            chunk.append(line.strip())
            
            if len(chunk) >= chunk_size:
                # 提交当前块进行处理
                result = pool.apply_async(
                    process_address_chunk, 
                    args=(chunk, chunk_position),
                    callback=lambda _: pbar.update(len(chunk))
                )
                
                # 收集结果
                vocab, word_sequences, stats = result.get()
                all_vocabs.append(vocab)
                all_word_sequences.extend(word_sequences)
                
                for key in all_stats:
                    all_stats[key].extend(stats[key])
                
                # 重置块
                chunk = []
                chunk_position += 1
        
        # 处理最后一个不完整的块
        if chunk:
            result = pool.apply_async(
                process_address_chunk, 
                args=(chunk, chunk_position),
                callback=lambda _: pbar.update(len(chunk))
            )
            
            vocab, word_sequences, stats = result.get()
            all_vocabs.append(vocab)
            all_word_sequences.extend(word_sequences)
            
            for key in all_stats:
                all_stats[key].extend(stats[key])
    
    # 关闭进程池和进度条
    pool.close()
    pool.join()
    pbar.close()
    
    # 构建词汇表
    word2id = build_vocabulary(all_vocabs)
    
    # 将词序列写入文件
    logging.info(f"将 {len(all_word_sequences)} 个词序列写入 {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(all_word_sequences)
    
    # 创建统计数据框
    stats_df = pd.DataFrame(all_stats)
    
    # 保存统计信息
    if stats_path:
        stats_df.to_csv(stats_path, index=False)
        logging.info(f"统计信息已保存到 {stats_path}")
    
    elapsed_time = time.time() - start_time
    logging.info(f"处理完成，耗时: {elapsed_time:.2f} 秒")
    
    return word2id, stats_df

def parallel_process_to_words(input_path: str, output_path: str, stats_path: str = None, 
                             chunk_size: int = 100000, num_workers: int = None) -> Tuple[Dict[str, int], pd.DataFrame]:
    """使用更高效的并行处理方法处理大文件"""
    start_time = time.time()
    logging.info(f"从 {input_path} 读取地址...")
    
    # 确定CPU核心数
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # 保留一个核心给系统
    
    logging.info(f"使用 {num_workers} 个工作进程进行并行处理")
    
    # 计算文件总行数以初始化进度条
    total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
    logging.info(f"文件包含 {total_lines} 行数据")
    
    # 分块读取文件
    chunks = []
    addresses = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="读取文件", unit="行"):
            addresses.append(line.strip())
            if len(addresses) >= chunk_size:
                chunks.append(addresses)
                addresses = []
    
    # 添加最后一个不完整的块
    if addresses:
        chunks.append(addresses)
    
    logging.info(f"文件已分为 {len(chunks)} 个块进行处理")
    
    # 创建进程池
    pool = mp.Pool(processes=num_workers)
    
    # 并行处理所有块
    results = []
    for i, chunk in enumerate(chunks):
        result = pool.apply_async(process_address_chunk, args=(chunk, i))
        results.append(result)
    
    # 收集结果
    all_vocabs = []
    all_word_sequences = []
    all_stats = {"地址": [], "长度": [], "前缀": []}
    
    for result in tqdm(results, desc="收集结果", unit="块"):
        vocab, word_sequences, stats = result.get()
        all_vocabs.append(vocab)
        all_word_sequences.extend(word_sequences)
        
        for key in all_stats:
            all_stats[key].extend(stats[key])
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 构建词汇表
    word2id = build_vocabulary(all_vocabs)
    
    # 将词序列写入文件
    logging.info(f"将 {len(all_word_sequences)} 个词序列写入 {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(all_word_sequences)
    
    # 创建统计数据框
    stats_df = pd.DataFrame(all_stats)
    
    # 保存统计信息
    if stats_path:
        stats_df.to_csv(stats_path, index=False)
        logging.info(f"统计信息已保存到 {stats_path}")
    
    elapsed_time = time.time() - start_time
    logging.info(f"处理完成，耗时: {elapsed_time:.2f} 秒")
    
    return word2id, stats_df

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="处理IPv6地址数据")
    parser.add_argument("--input", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/public_database/03.txt", help="输入文件路径")
    parser.add_argument("--output_dir", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/processed/03", help="输出目录")
    parser.add_argument("--chunk_size", type=int, default=100000, help="处理块大小")
    parser.add_argument("--workers", type=int, default=None, help="工作进程数")
    parser.add_argument("--method", type=str, choices=["standard", "parallel"], default="parallel", help="处理方法")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置输出文件路径
    output_path = os.path.join(args.output_dir, "word_sequences.txt")
    stats_path = os.path.join(args.output_dir, "address_stats.csv")
    vocab_path = os.path.join(args.output_dir, "vocabulary.json")
    
    # 处理数据
    if args.method == "standard":
        word2id, _ = process_to_words(
            args.input, 
            output_path, 
            stats_path, 
            args.chunk_size, 
            args.workers
        )
    else:
        word2id, _ = parallel_process_to_words(
            args.input, 
            output_path, 
            stats_path, 
            args.chunk_size, 
            args.workers
        )
    
    # 保存词汇表
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(word2id, f, ensure_ascii=False, indent=2)
    
    logging.info(f"词汇表已保存到 {vocab_path}")

    # 读取全部词序列
    with open(output_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    # 设置随机种子确保可重复性
    np.random.seed(42)
    np.random.shuffle(all_lines)
    
    # 按照8:1:1的比例划分数据集
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    # 计算分割点
    train_split_idx = int(len(all_lines) * train_ratio)
    val_split_idx = train_split_idx + int(len(all_lines) * val_ratio)
    
    # 划分数据集
    train_lines = all_lines[:train_split_idx]
    val_lines = all_lines[train_split_idx:val_split_idx]
    test_lines = all_lines[val_split_idx:]
    
    # 设置输出文件路径
    train_path = os.path.join(args.output_dir, "train_masked_sequences.txt")
    val_path = os.path.join(args.output_dir, "val_masked_sequences.txt")
    test_path = os.path.join(args.output_dir, "test_masked_sequences.txt")
    
    # 保存数据集
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(val_path, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
    
    # 输出数据集信息
    logging.info(f"数据集划分完成，比例为 8:1:1")
    logging.info(f"训练集: {len(train_lines)} 条记录，已保存到 {train_path}")
    logging.info(f"验证集: {len(val_lines)} 条记录，已保存到 {val_path}")
    logging.info(f"测试集: {len(test_lines)} 条记录，已保存到 {test_path}")
    
if __name__ == "__main__":
    main()