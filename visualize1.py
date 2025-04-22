import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from transformers import BertConfig, BertModel
from collections import defaultdict
from matplotlib.patches import Patch

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_vocab(model_path, vocab_path):
    """加载训练好的模型和词汇表"""
    # 加载词汇表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    
    # 创建模型配置 - 根据实际模型调整参数
    bert_config = BertConfig(
        vocab_size=len(word2id),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=34,
        type_vocab_size=1,
    )
    
    # 创建基础BERT模型并加载权重
    model = BertModel(bert_config)
    state_dict = torch.load(model_path, map_location='cpu')
    
    # 调整权重名称以匹配HuggingFace格式
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('bert.'):
            new_state_dict[k[5:]] = v  # 移除'bert.'前缀
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    return model, word2id

def extract_ipv6_embeddings(model, word2id):
    """提取IPv6地址词嵌入并进行分类"""
    # 提取词嵌入矩阵
    embeddings = model.embeddings.word_embeddings.weight.data.cpu().numpy()
    
    # 分类IPv6地址词
    nybble_words = defaultdict(list)
    position_words = defaultdict(list)
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    
    for word, idx in word2id.items():
        if word in special_tokens:
            continue
        
        # 分类nybble和position
        if len(word) >= 1:
            nybble = word[0]
            position = word[1:] if len(word) > 1 else ''
            
            nybble_words[nybble].append(embeddings[idx])
            position_words[position].append(embeddings[idx])
    
    return embeddings, nybble_words, position_words

def visualize_ipv6_structure(embeddings, word2id, output_dir):
    """专门针对IPv6结构的可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    words = list(word2id.keys())
    vectors = np.array([embeddings[word2id[word]] for word in words])
    
    # 过滤特殊标记
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    mask = [word not in special_tokens for word in words]
    filtered_words = [word for i, word in enumerate(words) if mask[i]]
    filtered_vectors = vectors[mask]
    
    # 提取IPv6特征
    nybbles = [word[0] if len(word) > 0 else '' for word in filtered_words]
    positions = [word[1:] if len(word) > 1 else '' for word in filtered_words]
    
    # 1. 使用UMAP进行降维 - 更适合保留全局结构
    logging.info("应用UMAP降维...")
    reducer = UMAP(n_components=2, random_state=42, 
                  n_neighbors=15, min_dist=0.1, metric='cosine')
    umap_vectors = reducer.fit_transform(filtered_vectors)
    
    # 2. 按nybble可视化
    plot_ipv6_features(umap_vectors, nybbles, positions, output_dir, 'umap')
    
    # 3. 使用PCA+TSNE组合
    logging.info("应用PCA+TSNE降维...")
    pca = PCA(n_components=50, random_state=42)
    pca_vectors = pca.fit_transform(filtered_vectors)
    
    tsne = TSNE(n_components=2, random_state=42, 
                perplexity=30, early_exaggeration=12, 
                learning_rate=200, n_iter=1000)
    tsne_vectors = tsne.fit_transform(pca_vectors)
    
    # 4. 按nybble可视化
    plot_ipv6_features(tsne_vectors, nybbles, positions, output_dir, 'tsne')
    
    # 5. 专门分析前缀和后缀结构
    analyze_prefix_suffix_structure(filtered_words, filtered_vectors, output_dir)

def plot_ipv6_features(vectors, nybbles, positions, output_dir, method):
    """绘制IPv6特征可视化图"""
    df = pd.DataFrame({
        'x': vectors[:, 0],
        'y': vectors[:, 1],
        'nybble': nybbles,
        'position': positions
    })
    
    # 按nybble可视化
    plt.figure(figsize=(16, 12))
    unique_nybbles = sorted(list(set(nybbles)))
    palette = sns.color_palette("hsv", len(unique_nybbles))
    
    for i, nybble in enumerate(unique_nybbles):
        mask = df['nybble'] == nybble
        plt.scatter(df[mask]['x'], df[mask]['y'], 
                   color=palette[i], label=nybble, alpha=0.7, s=30)
    
    plt.title(f'IPv6 Word Embeddings ({method.upper()} - by Nybble)')
    plt.xlabel(f'{method.upper()} dimension 1')
    plt.ylabel(f'{method.upper()} dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    
    nybble_path = os.path.join(output_dir, f'{method}_nybble.png')
    plt.savefig(nybble_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"保存nybble可视化到 {nybble_path}")
    
    # 按position可视化 (只显示前16个位置)
    plt.figure(figsize=(16, 12))
    position_counts = df['position'].value_counts()
    top_positions = position_counts[:16].index.tolist()
    
    palette = sns.color_palette("tab20", len(top_positions))
    
    for i, pos in enumerate(top_positions):
        mask = df['position'] == pos
        plt.scatter(df[mask]['x'], df[mask]['y'], 
                   color=palette[i], label=pos, alpha=0.7, s=30)
    
    # 其他位置
    mask = ~df['position'].isin(top_positions)
    if mask.any():
        plt.scatter(df[mask]['x'], df[mask]['y'], 
                   color='gray', label='Other', alpha=0.5, s=20)
    
    plt.title(f'IPv6 Word Embeddings ({method.upper()} - by Position)')
    plt.xlabel(f'{method.upper()} dimension 1')
    plt.ylabel(f'{method.upper()} dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    
    position_path = os.path.join(output_dir, f'{method}_position.png')
    plt.savefig(position_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"保存position可视化到 {position_path}")

def analyze_prefix_suffix_structure(words, vectors, output_dir):
    """专门分析IPv6前缀和后缀结构"""
    # 提取前缀(前8个nybble)和后缀(后24个nybble)
    prefixes = []
    suffixes = []
    prefix_vectors = []
    suffix_vectors = []
    
    for i, word in enumerate(words):
        if len(word) >= 2:  # 确保有nybble和position
            position = word[1:]
            if position.isdigit():  # 确保position是数字
                pos_num = int(position)
                if pos_num < 8:  # 前缀部分
                    prefixes.append((word, pos_num))
                    prefix_vectors.append(vectors[i])
                else:  # 后缀部分
                    suffixes.append((word, pos_num))
                    suffix_vectors.append(vectors[i])
    
    # 前缀分析
    if prefix_vectors and len(prefix_vectors) > 10:
        analyze_region_structure(prefix_vectors, prefixes, output_dir, 'prefix')
    
    # 后缀分析
    if suffix_vectors and len(suffix_vectors) > 10:
        analyze_region_structure(suffix_vectors, suffixes, output_dir, 'suffix')

def analyze_region_structure(vectors, words_with_pos, output_dir, region_name):
    """分析特定区域的结构"""
    words, positions = zip(*words_with_pos)
    positions = np.array(positions)
    vectors = np.array(vectors)
    
    # 标准化数据
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(vectors)
    
    # 使用PCA降维
    pca = PCA(n_components=2, random_state=42)
    pca_vectors = pca.fit_transform(scaled_vectors)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'x': pca_vectors[:, 0],
        'y': pca_vectors[:, 1],
        'position': positions,
        'nybble': [word[0] for word in words]
    })
    
    # 可视化位置分布
    plt.figure(figsize=(14, 10))
    sns.scatterplot(x='x', y='y', hue='position', data=df, 
                   palette='viridis', alpha=0.7, s=50)
    plt.title(f'IPv6 {region_name.capitalize()} Structure by Position')
    plt.xlabel('PCA dimension 1')
    plt.ylabel('PCA dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    pos_path = os.path.join(output_dir, f'pca_{region_name}_position.png')
    plt.savefig(pos_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"保存{region_name}位置分析到 {pos_path}")
    
    # 可视化nybble分布
    plt.figure(figsize=(14, 10))
    unique_nybbles = sorted(df['nybble'].unique())
    palette = sns.color_palette("hsv", len(unique_nybbles))
    
    for i, nybble in enumerate(unique_nybbles):
        mask = df['nybble'] == nybble
        plt.scatter(df[mask]['x'], df[mask]['y'], 
                   color=palette[i], label=nybble, alpha=0.7, s=50)
    
    plt.title(f'IPv6 {region_name.capitalize()} Structure by Nybble')
    plt.xlabel('PCA dimension 1')
    plt.ylabel('PCA dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    
    nybble_path = os.path.join(output_dir, f'pca_{region_name}_nybble.png')
    plt.savefig(nybble_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"保存{region_name} nybble分析到 {nybble_path}")

if __name__ == "__main__":
    # 配置路径
    model_path = "d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/models/best_bert_model.pt"
    vocab_path = "d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/processed/vocabulary.json"
    output_dir = "d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/visualizations"
    
    # 加载模型和词汇表
    logging.info("加载模型和词汇表...")
    model, word2id = load_model_and_vocab(model_path, vocab_path)
    
    # 提取词嵌入
    logging.info("提取词嵌入...")
    embeddings, nybble_words, position_words = extract_ipv6_embeddings(model, word2id)
    
    # 可视化IPv6结构
    logging.info("开始可视化IPv6结构...")
    visualize_ipv6_structure(embeddings, word2id, output_dir)
    
    logging.info("可视化完成！所有结果已保存到 %s", output_dir)