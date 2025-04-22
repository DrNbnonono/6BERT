import torch
import numpy as np
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import json
import os
import logging
from transformers import BertConfig, BertForMaskedLM, BertModel
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_vocab(model_path, vocab_path):
    """加载训练好的模型和词汇表"""
    # 加载词汇表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    
    # 加载模型配置 - 需要与训练时的配置一致
    bert_config = BertConfig(
        vocab_size=len(word2id),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=34,  # 与训练时保持一致
        type_vocab_size=1,  # 与训练时保持一致
    )
    
    # 创建模型
    model = BertForMaskedLM(bert_config)
    
    # 加载预训练权重
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 调整权重名称以匹配HuggingFace格式
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('bert.'):
            new_state_dict[k[5:]] = v  # 移除'bert.'前缀
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    return model, word2id

def get_contextual_embeddings(model, word2id, sample_size=1000):
    """获取上下文嵌入"""
    model.eval()
    device = next(model.parameters()).device
    
    # 创建示例输入
    words = [w for w in word2id.keys() if w not in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]]
    sample_words = np.random.choice(words, min(sample_size, len(words)), replace=False)
    
    # 准备输入数据
    embeddings = {}
    for word in tqdm(sample_words, desc="提取上下文嵌入"):
        # 创建包含目标词的简单序列
        input_ids = torch.tensor([[word2id["[CLS]"], word2id[word], word2id["[SEP]"]]]).to(device)
        
        with torch.no_grad():
            outputs = model.bert(input_ids)
            # 取目标词位置的隐藏状态
            embeddings[word] = outputs.last_hidden_state[0, 1].cpu().numpy()
    
    return embeddings

def visualize_embeddings(embeddings, word2id, output_dir):
    """改进的可视化函数，支持多种降维方法和3D可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    words = list(embeddings.keys())
    vectors = np.array([embeddings[word] for word in words])
    
    # 提取nybble和位置信息
    nybbles = [word[0] if len(word) > 0 else '' for word in words]
    positions = [word[1:] if len(word) > 1 else '' for word in words]
    
    # 1. 使用PCA降维到3D
    pca_3d = PCA(n_components=3)
    pca_3d_vectors = pca_3d.fit_transform(vectors)
    
    # 3D可视化
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    df_3d = pd.DataFrame({
        'x': pca_3d_vectors[:, 0],
        'y': pca_3d_vectors[:, 1],
        'z': pca_3d_vectors[:, 2],
        'word': words,
        'nybble': nybbles
    })
    
    unique_nybbles = sorted(list(set(nybbles)))
    palette = sns.color_palette("hsv", len(unique_nybbles))
    
    for i, nybble in enumerate(unique_nybbles):
        mask = df_3d['nybble'] == nybble
        ax.scatter(df_3d[mask]['x'], df_3d[mask]['y'], df_3d[mask]['z'], 
                  color=palette[i], label=nybble, alpha=0.7, s=30)
    
    ax.set_title('PCA 3D Visualization of BERT Embeddings (by nybble)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(output_dir, 'pca_3d_nybble.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. 使用UMAP降维
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_vectors = reducer.fit_transform(vectors)
    
    df_umap = pd.DataFrame({
        'x': umap_vectors[:, 0],
        'y': umap_vectors[:, 1],
        'word': words,
        'nybble': nybbles
    })
    
    plt.figure(figsize=(16, 12))
    sns.scatterplot(x='x', y='y', hue='nybble', data=df_umap, palette='tab20', alpha=0.7, s=50)
    plt.title('UMAP Visualization of BERT Embeddings (by nybble)')
    plt.savefig(os.path.join(output_dir, 'umap_nybble.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. 保留原有的t-SNE可视化
    pca = PCA(n_components=50)
    pca_vectors = pca.fit_transform(vectors)
    
    tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=24, 
                learning_rate=500, n_iter=2000, random_state=42)
    reduced_vectors = tsne.fit_transform(pca_vectors)
    
    df = pd.DataFrame({
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1],
        'word': words,
        'nybble': nybbles,
        'position': positions
    })
    
if __name__ == "__main__":
    # 配置路径
    model_path = "d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/models/best_bert_model.pt"
    vocab_path = "d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/processed/vocabulary.json"
    output_dir = "d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/visualizations/visualize0"
    
    # 加载模型和词汇表
    model, word2id = load_model_and_vocab(model_path, vocab_path)
    
    # 获取上下文嵌入
    contextual_embeddings = get_contextual_embeddings(model, word2id)
    
    # 可视化
    visualize_embeddings(contextual_embeddings, word2id, output_dir)