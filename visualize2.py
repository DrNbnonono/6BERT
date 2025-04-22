import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from umap import UMAP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
from bertviz import head_view
from transformers import BertTokenizer, BertModel, BertConfig
import ipaddress
import torch
import os
import json

class IPv6BERTVisualizer:
    def __init__(self, model_path, vocab_path):
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]  # 先初始化special_tokens
        self.model, self.word2id = self._load_model(model_path, vocab_path)
        self.embeddings = self._extract_embeddings()
        
    def _load_model(self, model_path, vocab_path):
        """加载模型和词汇表"""
        with open(vocab_path, 'r') as f:
            word2id = json.load(f)
        
        config = BertConfig(
            vocab_size=len(word2id),
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12
        )
        
        model = BertModel(config)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)  # 添加weights_only=True参数
        model.load_state_dict(state_dict, strict=False)
        return model, word2id

    def _extract_embeddings(self):
        """提取词嵌入并分类"""
        embeddings = self.model.embeddings.word_embeddings.weight.data.cpu().numpy()
        
        # 分类IPv6地址词
        self.nybble_data = []
        self.position_data = []
        self.words = []
        
        for word, idx in self.word2id.items():
            if word not in self.special_tokens and len(word) >= 2:
                nybble = word[0]
                position = word[1:]
                
                if position.isdigit():
                    self.nybble_data.append((nybble, embeddings[idx]))
                    self.position_data.append((int(position), embeddings[idx]))
                    self.words.append(word)
        
        return embeddings
    
    def visualize_3d_structure(self, method='umap', output_dir='visualizations'):
        """三维可视化IPv6地址结构"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据
        vectors = np.array([vec for (_, vec) in self.nybble_data])
        nybbles = [nyb for (nyb, _) in self.nybble_data]
        positions = [pos for (pos, _) in self.position_data]
        
        # 降维到3D
        if method == 'umap':
            reducer = UMAP(n_components=3, random_state=42, 
                          n_neighbors=15, min_dist=0.1, metric='cosine')
        elif method == 'pca':
            reducer = PCA(n_components=3, random_state=42)
        else:  # tsne
            reducer = TSNE(n_components=3, random_state=42, 
                          perplexity=30, learning_rate=200)
        
        reduced_vectors = reducer.fit_transform(vectors)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'x': reduced_vectors[:, 0],
            'y': reduced_vectors[:, 1],
            'z': reduced_vectors[:, 2],
            'nybble': nybbles,
            'position': positions,
            'word': self.words
        })
        
        # 3D交互式可视化 - 按nybble
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='nybble',
                           hover_data=['word', 'position'],
                           title=f'IPv6 Address Structure (3D {method.upper()} - by Nybble)')
        fig.write_html(os.path.join(output_dir, f'3d_{method}_nybble.html'))
        
        # 3D交互式可视化 - 按position
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='position',
                           hover_data=['word', 'nybble'],
                           title=f'IPv6 Address Structure (3D {method.upper()} - by Position)')
        fig.write_html(os.path.join(output_dir, f'3d_{method}_position.html'))
        
        # 静态3D可视化
        self._plot_static_3d(df, method, output_dir)
        
        # 聚类分析
        self._cluster_analysis(reduced_vectors, df, method, output_dir)

    def _plot_static_3d(self, df, method, output_dir):
        """生成静态3D图"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 按nybble着色
        unique_nybbles = sorted(df['nybble'].unique())
        colors = sns.color_palette("hsv", len(unique_nybbles))
        
        for nybble, color in zip(unique_nybbles, colors):
            mask = df['nybble'] == nybble
            ax.scatter(df[mask]['x'], df[mask]['y'], df[mask]['z'], 
                      c=[color], label=nybble, alpha=0.7, s=30)
        
        ax.set_title(f'IPv6 Address Structure (3D {method.upper()} - by Nybble)')
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.set_zlabel(f'{method.upper()} 3')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'3d_{method}_nybble_static.png'), dpi=300)
        plt.close()

    def _cluster_analysis(self, vectors, df, method, output_dir):
        """聚类分析"""
        # K-means聚类
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)  # 添加n_init参数
        clusters = kmeans.fit_predict(vectors)
        df['cluster'] = clusters
        
        # 3D聚类可视化
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster',
                           hover_data=['word', 'nybble', 'position'],
                           title=f'IPv6 Address Clusters (3D {method.upper()})')
        fig.write_html(os.path.join(output_dir, f'3d_{method}_clusters.html'))
        
        # 分析每个簇的特征
        cluster_stats = df.groupby('cluster').agg({
            'nybble': lambda x: x.value_counts().index[0],
            'position': ['min', 'max', lambda x: x.value_counts().index[0]]
        })
        cluster_stats.to_csv(os.path.join(output_dir, f'cluster_stats_{method}.csv'))

    def visualize_attention(self, sample_address, output_dir):
        """可视化注意力机制"""
        try:
            # 将IPv6地址转换为模型输入格式
            exploded = ipaddress.IPv6Address(sample_address).exploded.replace(":", "")
            words = ["[CLS]"] + [f"{nybble}{i}" for i, nybble in enumerate(exploded)] + ["[SEP]"]
            
            # 转换为模型输入
            input_ids = torch.tensor([[self.word2id.get(w, self.word2id["[UNK]"]) for w in words]])
            
            # 获取注意力权重
            outputs = self.model(input_ids, output_attentions=True)
            attention = outputs.attentions  # 所有层的注意力权重
            
            # 手动可视化注意力权重
            if attention is not None and len(attention) > 0:
                # 选择最后一层的注意力
                last_layer_attention = attention[-1][0].detach().cpu().numpy()  # [num_heads, seq_len, seq_len]
                
                # 计算平均注意力
                avg_attention = last_layer_attention.mean(axis=0)  # [seq_len, seq_len]
                
                # 绘制热力图
                plt.figure(figsize=(12, 10))
                sns.heatmap(avg_attention, xticklabels=words, yticklabels=words, cmap="YlGnBu")
                plt.title(f'IPv6地址注意力热力图 ({sample_address})')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'attention_heatmap.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 为每个注意力头创建单独的热力图
                num_heads = last_layer_attention.shape[0]
                rows = int(np.ceil(num_heads / 2))
                fig, axes = plt.subplots(rows, 2, figsize=(20, 5*rows))
                axes = axes.flatten()
                
                for h in range(num_heads):
                    if h < len(axes):
                        sns.heatmap(last_layer_attention[h], xticklabels=words if h >= num_heads-2 else [], 
                                yticklabels=words if h % 2 == 0 else [], cmap="YlGnBu", ax=axes[h])
                        axes[h].set_title(f'注意力头 #{h+1}')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'attention_heads.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"注意力可视化已保存到 {output_dir}")
            else:
                print("无法获取注意力权重")
            
        except Exception as e:
            print(f"Error visualizing attention: {e}")

    def visualize_hexbin(self, output_dir, method='pca'):
        """Hexbin密度可视化"""
        # 降维到2D
        vectors = np.array([vec for (_, vec) in self.nybble_data])
        
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # umap
            reducer = UMAP(n_components=2, random_state=42)
            
        reduced = reducer.fit_transform(vectors)
        
        # Hexbin图
        plt.figure(figsize=(12, 10))
        plt.hexbin(reduced[:, 0], reduced[:, 1], gridsize=50, cmap='Blues')
        plt.colorbar(label='Count in bin')
        plt.title(f'IPv6 Address Density ({method.upper()})')
        plt.savefig(os.path.join(output_dir, f'hexbin_{method}.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    visualizer = IPv6BERTVisualizer(
        model_path="D:\\bigchuang\\ipv6地址论文\\10-6VecLM\\BERT\\models\\best_bert_model.pt",
        vocab_path="D:\\bigchuang\\ipv6地址论文\\10-6VecLM\\BERT\\data\\processed\\vocabulary.json"
    )
    
    # 多种可视化方法
    visualizer.visualize_3d_structure(method='umap')
    visualizer.visualize_3d_structure(method='pca')
    
    # 注意力可视化(示例地址)
    visualizer.visualize_attention("2001:db8::1", output_dir="visualizations")
    
    # 密度可视化
    visualizer.visualize_hexbin(output_dir="visualizations")