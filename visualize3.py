import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import logging
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM, BertModel
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
import ipaddress
import random
from scipy.spatial import distance
import matplotlib as mpl

try:
    # Windows系统
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    mpl.rcParams['font.family'] = 'sans-serif'
except:
    # 如果上述方法失败，可以尝试使用matplotlib自带的字体
    logging.warning("无法设置中文字体，图表中的中文可能无法正确显示")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IPv6BERTVisualizer:
    """IPv6 BERT模型可视化与分析工具"""
    
    def __init__(self, model_path, vocab_path, output_dir="visualizations"):
        """
        初始化可视化器
        
        Args:
            model_path: BERT模型路径
            vocab_path: 词汇表路径
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
        
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = len(self.word2id)
        
        # 加载模型配置 - 使用实际的词汇表大小和与训练时一致的参数
        self.config = BertConfig(
            vocab_size=self.vocab_size,  # 使用实际的词汇表大小
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            output_attentions=True,  # 启用注意力输出
            output_hidden_states=True,  # 启用隐藏状态输出
            max_position_embeddings=34,  # 修改为与训练时一致的值
            type_vocab_size=1  # 修改为与训练时一致的值
        )
        
        # 加载模型 - 使用torch.load直接加载状态字典，并添加weights_only=True参数
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # 创建模型并加载状态字典
        self.model = BertForMaskedLM(self.config)
        self.model.load_state_dict(state_dict)
        
        # 提取基础模型
        self.base_model = self.model.bert
        
        # 提取词嵌入
        self.embeddings = self.model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
        
        logging.info(f"Model loaded, vocabulary size: {self.vocab_size}, embedding dimension: {self.embeddings.shape[1]}")
        logging.info(f"Number of attention heads: {self.config.num_attention_heads}, number of hidden layers: {self.config.num_hidden_layers}")
    
    def analyze_embedding_space(self):
        """分析词嵌入空间的基本特性"""
        # 过滤掉特殊标记
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        special_ids = [self.word2id[token] for token in special_tokens]
        
        # 提取非特殊标记的嵌入
        filtered_embeddings = np.array([self.embeddings[i] for i in range(self.vocab_size) if i not in special_ids])
        
        # 计算向量范数
        norms = np.linalg.norm(filtered_embeddings, axis=1)
        avg_norm = np.mean(norms)
        
        # 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(filtered_embeddings)
        
        # 提取上三角部分（不包括对角线）
        mask = np.triu_indices(similarity_matrix.shape[0], k=1)
        similarities = similarity_matrix[mask]
        
        # 计算统计量
        avg_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        
        # 绘制相似度分布直方图
        plt.figure(figsize=(10, 6))
        sns.histplot(similarities, bins=50, kde=True)
        plt.axvline(avg_sim, color='r', linestyle='--', label=f'平均值: {avg_sim:.2f}')
        plt.title('词向量余弦相似度分布')
        plt.xlabel('余弦相似度')
        plt.ylabel('频率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "similarity-distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存统计结果
        stats_df = pd.DataFrame({
            '指标': ['平均相似度', '相似度标准差', '最小相似度', '最大相似度', '平均向量范数'],
            '值': [f"{avg_sim:.2f}±{std_sim:.2f}", f"{std_sim:.2f}", f"{min_sim:.2f}", f"{max_sim:.2f}", f"{avg_norm:.2f}"]
        })
        
        stats_df.to_csv(os.path.join(self.output_dir, "embedding_stats.csv"), index=False)
        
        logging.info(f"Embedding space analysis completed. Average similarity: {avg_sim:.2f}±{std_sim:.2f}, range: [{min_sim:.2f}, {max_sim:.2f}], average norm: {avg_norm:.2f}")
        
        return {
            'avg_sim': avg_sim,
            'std_sim': std_sim,
            'min_sim': min_sim,
            'max_sim': max_sim,
            'avg_norm': avg_norm,
            'filtered_embeddings': filtered_embeddings
        }
    
    def cluster_analysis(self, filtered_embeddings=None, k=8):
        """对嵌入空间进行聚类分析"""
        if filtered_embeddings is None:
            # 过滤掉特殊标记
            special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            special_ids = [self.word2id[token] for token in special_tokens]
            filtered_embeddings = np.array([self.embeddings[i] for i in range(self.vocab_size) if i not in special_ids])
        
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(filtered_embeddings)
        
        # 计算聚类评估指标
        silhouette = silhouette_score(filtered_embeddings, clusters)
        db_score = davies_bouldin_score(filtered_embeddings, clusters)
        ch_score = calinski_harabasz_score(filtered_embeddings, clusters)
        
        # 使用PCA降维到2维进行可视化
        pca = PCA(n_components=2, random_state=42)
        reduced_embeddings = pca.fit_transform(filtered_embeddings)
        
        # 绘制聚类结果
        plt.figure(figsize=(12, 10))
        
        # 为每个聚类创建不同的颜色
        palette = sns.color_palette("hsv", k)
        
        # 绘制散点图
        for i in range(k):
            mask = clusters == i
            plt.scatter(
                reduced_embeddings[mask, 0], 
                reduced_embeddings[mask, 1],
                color=palette[i],
                label=f'Cluster {i}',
                alpha=0.7,
                s=50
            )
        
        # 绘制聚类中心
        centers = pca.transform(kmeans.cluster_centers_)
        plt.scatter(
            centers[:, 0], 
            centers[:, 1],
            c='black',
            marker='X',
            s=200,
            alpha=1,
            label='Centroids'
        )
        
        plt.title(f'K-means聚类结果 (k={k})\nSilhouette: {silhouette:.2f}, Davies-Bouldin: {db_score:.2f}, Calinski-Harabasz: {ch_score:.0f}')
        plt.xlabel('PCA维度1')
        plt.ylabel('PCA维度2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "cluster_visualization.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存聚类评估指标
        cluster_stats = pd.DataFrame({
            '指标': ['轮廓系数', 'Davies-Bouldin指数', 'Calinski-Harabasz指数'],
            '值': [f"{silhouette:.2f}", f"{db_score:.2f}", f"{ch_score:.0f}"]
        })
        
        cluster_stats.to_csv(os.path.join(self.output_dir, "cluster_stats.csv"), index=False)
        
        logging.info(f"Cluster analysis completed. Silhouette: {silhouette:.2f}, Davies-Bouldin: {db_score:.2f}, Calinski-Harabasz: {ch_score:.0f}")
        
        return {
            'silhouette': silhouette,
            'db_score': db_score,
            'ch_score': ch_score,
            'clusters': clusters
        }
    
    def position_entropy_analysis(self, test_data_path):
        """分析各位置的熵值分布"""
        # 读取测试数据
        with open(test_data_path, 'r', encoding='utf-8') as f:
            sequences = [line.strip().split() for line in f]
        
        # 提取地址部分（去掉[CLS]和[SEP]）
        address_sequences = [seq[1:-1] for seq in sequences]
        
        # 计算每个位置的熵
        position_entropy = []
        
        for pos in range(32):  # IPv6地址有32个十六进制数字
            # 收集该位置的所有值
            values = [seq[pos] if pos < len(seq) else None for seq in address_sequences]
            values = [v for v in values if v is not None]
            
            # 计算频率分布
            value_counts = {}
            for v in values:
                if v in value_counts:
                    value_counts[v] += 1
                else:
                    value_counts[v] = 1
            
            # 计算熵
            total = len(values)
            entropy = 0
            for count in value_counts.values():
                p = count / total
                entropy -= p * np.log2(p)
            
            position_entropy.append(entropy)
        
        # 计算前缀和后缀的平均熵（假设前16位是前缀，后16位是后缀）
        prefix_entropy = np.mean(position_entropy[:16])  # 前16个位置（前缀）
        suffix_entropy = np.mean(position_entropy[16:])  # 后16个位置（后缀）
        
        # 绘制熵值分布
        plt.figure(figsize=(12, 6))
        
        # 为不同部分使用不同颜色
        colors = ['#3498db'] * 16 + ['#e74c3c'] * 16
        
        plt.bar(range(32), position_entropy, color=colors)
        
        # 添加部分区域标记
        plt.axvline(x=15.5, color='black', linestyle='--', alpha=0.5)
        
        # 添加区域标签
        plt.text(8, max(position_entropy) * 1.05, f'网络前缀\n平均熵: {prefix_entropy:.2f}', ha='center')
        plt.text(24, max(position_entropy) * 1.05, f'接口标识符\n平均熵: {suffix_entropy:.2f}', ha='center')
        
        plt.title('IPv6地址各位置的熵值分布')
        plt.xlabel('位置')
        plt.ylabel('熵 (bits)')
        plt.xticks(range(32))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "position_entropy.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存熵值数据
        entropy_df = pd.DataFrame({
            '位置': range(32),
            '熵值': position_entropy,
            '区域': ['网络前缀'] * 16 + ['接口标识符'] * 16
        })
        
        entropy_df.to_csv(os.path.join(self.output_dir, "position_entropy.csv"), index=False)
        
        logging.info(f"Position entropy analysis completed. Prefix entropy: {prefix_entropy:.2f}, Suffix entropy: {suffix_entropy:.2f}")
        
        return {
            'position_entropy': position_entropy,
            'prefix_entropy': prefix_entropy,
            'suffix_entropy': suffix_entropy
        }

    def position_correlation_analysis(self, test_data_path):
        """分析位置间的相关性"""
        # 读取测试数据
        with open(test_data_path, 'r', encoding='utf-8') as f:
            sequences = [line.strip().split() for line in f]
        
        # 提取地址部分（去掉[CLS]和[SEP]）
        address_sequences = [seq[1:-1] for seq in sequences]
        
        # 将字符转换为数值（十六进制）
        numeric_sequences = []
        for seq in address_sequences:
            numeric_seq = []
            for nybble in seq:
                try:
                    numeric_seq.append(int(nybble, 16))
                except ValueError:
                    # 如果不是有效的十六进制数字，使用0
                    numeric_seq.append(0)
            
            # 确保长度为32
            if len(numeric_seq) < 32:
                numeric_seq.extend([0] * (32 - len(numeric_seq)))
            elif len(numeric_seq) > 32:
                numeric_seq = numeric_seq[:32]
            
            numeric_sequences.append(numeric_seq)
        
        # 转换为NumPy数组
        data = np.array(numeric_sequences)
        
        # 计算相关性矩阵
        correlation_matrix = np.zeros((32, 32))
        
        for i in range(32):
            for j in range(32):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # 使用Spearman相关系数，对非正态分布数据更适用
                    correlation, _ = stats.spearmanr(data[:, i], data[:, j])
                    correlation_matrix[i, j] = correlation
        
        # 计算相邻位置和非相邻位置的平均相关系数
        adjacent_correlations = []
        for i in range(31):
            adjacent_correlations.append(correlation_matrix[i, i+1])
        
        non_adjacent_correlations = []
        for i in range(32):
            for j in range(32):
                if i != j and abs(i - j) > 1:
                    non_adjacent_correlations.append(correlation_matrix[i, j])
        
        avg_adjacent_corr = np.mean(adjacent_correlations)
        avg_non_adjacent_corr = np.mean(non_adjacent_correlations)
        
        # 找出显著相关的位置对
        significant_pairs = []
        for i in range(32):
            for j in range(i+1, 32):
                if abs(correlation_matrix[i, j]) > 0.5:  # 相关系数绝对值大于0.5视为显著
                    significant_pairs.append((i, j, correlation_matrix[i, j]))
        
        # 按相关系数绝对值排序
        significant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # 绘制相关性热图
        plt.figure(figsize=(12, 10))
        mask = np.zeros_like(correlation_matrix)
        mask[np.triu_indices_from(mask)] = True  # 只显示下三角
        
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=.5,
            annot=False,
            fmt='.2f',
            cbar_kws={'shrink': .8, 'label': 'Spearman相关系数'}
        )
        
        plt.title('IPv6地址位置间相关性矩阵')
        plt.xlabel('位置')
        plt.ylabel('位置')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "position_correlation.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存相关性数据
        correlation_df = pd.DataFrame(correlation_matrix)
        correlation_df.columns = [f'位置{i}' for i in range(32)]
        correlation_df.index = [f'位置{i}' for i in range(32)]
        correlation_df.to_csv(os.path.join(self.output_dir, "position_correlation.csv"))
        
        # 保存显著相关对
        if significant_pairs:
            sig_pairs_df = pd.DataFrame(significant_pairs, columns=['位置1', '位置2', '相关系数'])
            sig_pairs_df.to_csv(os.path.join(self.output_dir, "significant_correlations.csv"), index=False)
        
        logging.info(f"Position correlation analysis completed. Average adjacent correlation: {avg_adjacent_corr:.2f}, Average non-adjacent correlation: {avg_non_adjacent_corr:.2f}")
        logging.info(f"Found {len(significant_pairs)} significantly correlated position pairs")
        
        return {
            'correlation_matrix': correlation_matrix,
            'avg_adjacent_corr': avg_adjacent_corr,
            'avg_non_adjacent_corr': avg_non_adjacent_corr,
            'significant_pairs': significant_pairs
        }
    
    def _evaluate_mask_prediction(self, test_data_path, num_samples=100):
        """评估模型的掩码预测性能"""
        # 读取测试数据
        with open(test_data_path, 'r', encoding='utf-8') as f:
            sequences = [line.strip().split() for line in f]
        
        # 随机选择样本
        if len(sequences) > num_samples:
            sequences = random.sample(sequences, num_samples)
        
        # 准备评估
        self.model.eval()
        correct = 0
        total = 0
        top5_correct = 0
        
        # 按位置记录准确率
        position_correct = [0] * 32
        position_total = [0] * 32
        
        # 使用tqdm显示进度
        for sequence in tqdm(sequences, desc="Evaluating Mask Prediction"):
            # 复制序列以便修改
            orig_sequence = sequence.copy()
            
            # 随机选择一个位置进行掩码（排除[CLS]和[SEP]）
            mask_pos = random.randint(1, len(sequence) - 2)
            position = mask_pos - 1  # 调整为0-31的位置
            
            # 记录原始标记
            original_token = sequence[mask_pos]
            
            # 替换为[MASK]
            sequence[mask_pos] = "[MASK]"
            
            # 转换为ID
            input_ids = [self.word2id.get(token, self.word2id["[UNK]"]) for token in sequence]
            
            # 转换为张量
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                predictions = outputs.logits[0, mask_pos].cpu().numpy()
            
            # 获取预测结果
            top_indices = np.argsort(predictions)[::-1]
            top1_prediction = self.id2word[top_indices[0]]
            top5_predictions = [self.id2word[idx] for idx in top_indices[:5]]
            
            # 检查是否正确
            if top1_prediction == original_token:
                correct += 1
                position_correct[position] += 1
            
            if original_token in top5_predictions:
                top5_correct += 1
            
            total += 1
            position_total[position] += 1
        
        # 计算总体准确率
        accuracy = correct / total if total > 0 else 0
        top5_accuracy = top5_correct / total if total > 0 else 0
        
        # 计算各位置准确率
        position_accuracy = []
        for i in range(32):
            if position_total[i] > 0:
                position_accuracy.append(position_correct[i] / position_total[i])
            else:
                position_accuracy.append(0)
        
        # 绘制位置准确率图
        plt.figure(figsize=(12, 6))
        
        # 为不同部分使用不同颜色
        colors = ['#3498db'] * 16 + ['#e74c3c'] * 16
        
        plt.bar(range(32), position_accuracy, color=colors)
        
        # 添加部分区域标记
        plt.axvline(x=15.5, color='black', linestyle='--', alpha=0.5)
        
        # 添加区域标签
        plt.text(8, max(position_accuracy) * 1.05, '网络前缀', ha='center')
        plt.text(24, max(position_accuracy) * 1.05, '接口标识符', ha='center')
        
        plt.title(f'IPv6地址各位置的掩码预测准确率\n总体准确率: {accuracy:.1%}, Top-5准确率: {top5_accuracy:.1%}')
        plt.xlabel('位置')
        plt.ylabel('准确率')
        plt.xticks(range(0, 32, 2))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "mask_prediction_accuracy.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存准确率数据
        accuracy_df = pd.DataFrame({
            '位置': range(32),
            '准确率': position_accuracy,
            '区域': ['网络前缀'] * 16 + ['接口标识符'] * 16
        })
        
        accuracy_df.to_csv(os.path.join(self.output_dir, "position_accuracy.csv"), index=False)
        
        logging.info(f"Mask prediction evaluation completed. Overall accuracy: {accuracy:.1%}, Top-5 accuracy: {top5_accuracy:.1%}")
        
        return {
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'position_accuracy': position_accuracy
        }
    
    def visualize_attention(self, address="2001:db8::", layer=11):
        """可视化特定地址的注意力机制"""
        # 将地址转换为标准格式
        try:
            ip = ipaddress.IPv6Address(address)
            exploded = ip.exploded.replace(":", "")
            
            # 确保地址长度为32个十六进制数字
            if len(exploded) != 32:
                exploded = exploded.zfill(32)
            
            # 构建输入序列
            tokens = ["[CLS]"] + [c for c in exploded] + ["[SEP]"]
            input_ids = [self.word2id.get(token, self.word2id["[UNK]"]) for token in tokens]
            
            # 转换为张量
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            
            # 获取注意力
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_tensor, output_attentions=True)
                attentions = outputs.attentions
            
            # 提取指定层的注意力
            layer_attention = attentions[layer][0].cpu().numpy()  # [num_heads, seq_len, seq_len]
            
            # 绘制注意力热图
            num_heads = layer_attention.shape[0]
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))  # 假设有12个注意力头
            axes = axes.flatten()
            
            for head in range(num_heads):
                ax = axes[head]
                attention_matrix = layer_attention[head]
                
                # 只关注地址部分（排除[CLS]和[SEP]）
                address_attention = attention_matrix[1:-1, 1:-1]
                
                sns.heatmap(
                    address_attention,
                    ax=ax,
                    cmap='viridis',
                    vmin=0,
                    square=True,
                    cbar_kws={'shrink': .8}
                )
                
                ax.set_title(f'Head {head+1}')
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.suptitle(f'注意力热图 - 地址: {address}, 层: {layer+1}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(os.path.join(self.output_dir, "attention_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 保存注意力数据
            attention_data = {}
            for head in range(num_heads):
                attention_data[f'head_{head+1}'] = layer_attention[head].tolist()
            
            with open(os.path.join(self.output_dir, "attention_data.json"), 'w', encoding='utf-8') as f:
                json.dump(attention_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Attention visualization for address {address} completed")
            
            # 分析注意力模式
            self._analyze_attention_patterns(layer_attention, tokens)
            
            return True
        except Exception as e:
            logging.error(f"Error visualizing attention: {e}")
            return False
    
    def _analyze_attention_patterns(self, attention, tokens):
        """分析注意力模式"""
        num_heads = attention.shape[0]
        seq_len = attention.shape[1]
        
        # 分析每个注意力头
        head_patterns = []
        
        for head in range(num_heads):
            head_attention = attention[head]
            
            # 绘制注意力热图
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                head_attention,
                cmap='viridis',
                vmin=0,
                square=True,
                xticklabels=tokens,
                yticklabels=tokens,
                cbar_kws={'shrink': .8, 'label': '注意力权重'}
            )
            
            plt.title(f'注意力头 {head+1}/{self.config.num_attention_heads} 的注意力分布')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"attention_head_{head+1}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 保存注意力数据
        attention_df = pd.DataFrame(average_attention)
        attention_df.columns = tokens
        attention_df.index = tokens
        attention_df.to_csv(os.path.join(self.output_dir, "attention_weights.csv"))
        
        logging.info(f"Attention analysis completed for address: {address}")
        
        return {
            'tokens': tokens,
            'attention': attention
        }
    
    def visualize_by_nybble_value(self):
        """按nybble值对嵌入空间进行可视化"""
        # 过滤掉特殊标记
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        special_ids = [self.word2id[token] for token in special_tokens]
        
        # 提取非特殊标记的嵌入和对应的标记
        filtered_embeddings = []
        filtered_tokens = []
        
        for i in range(self.vocab_size):
            if i not in special_ids:
                filtered_embeddings.append(self.embeddings[i])
                filtered_tokens.append(self.id2word[i])
        
        filtered_embeddings = np.array(filtered_embeddings)
        
        # 使用PCA降维到2维
        pca = PCA(n_components=2, random_state=42)
        reduced_embeddings = pca.fit_transform(filtered_embeddings)
        
        # 为每个nybble值分配颜色
        hex_values = [token for token in filtered_tokens if len(token) == 1 and token.isalnum()]
        
        # 创建颜色映射
        unique_values = sorted(set(hex_values))
        color_map = {}
        palette = sns.color_palette("hsv", len(unique_values))
        
        for i, value in enumerate(unique_values):
            color_map[value] = palette[i]
        
        # 绘制散点图
        plt.figure(figsize=(12, 10))
        
        # 为每个nybble值绘制一个散点图
        for value in unique_values:
            indices = [i for i, token in enumerate(filtered_tokens) if token == value]
            if indices:
                plt.scatter(
                    reduced_embeddings[indices, 0],
                    reduced_embeddings[indices, 1],
                    color=color_map[value],
                    label=value,
                    alpha=0.7,
                    s=50
                )
        
        plt.title('按nybble值着色的嵌入空间分布')
        plt.xlabel('PCA维度1')
        plt.ylabel('PCA维度2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "nybble_visualization.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Nybble visualization completed")
    
    def visualize_by_position(self):
        """按位置对嵌入空间进行可视化"""
        # 过滤掉特殊标记
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        special_ids = [self.word2id[token] for token in special_tokens]
        
        # 提取非特殊标记的嵌入和对应的标记
        filtered_embeddings = []
        filtered_tokens = []
        
        for i in range(self.vocab_size):
            if i not in special_ids:
                filtered_embeddings.append(self.embeddings[i])
                filtered_tokens.append(self.id2word[i])
        
        filtered_embeddings = np.array(filtered_embeddings)
        
        # 使用PCA降维到2维
        pca = PCA(n_components=2, random_state=42)
        reduced_embeddings = pca.fit_transform(filtered_embeddings)
        
        # 提取位置信息（如果标记格式为"pos_X"）
        positions = []
        for token in filtered_tokens:
            if '_' in token and token.split('_')[0] == 'pos':
                try:
                    pos = int(token.split('_')[1])
                    positions.append(pos)
                except ValueError:
                    positions.append(-1)  # 无效位置
            else:
                positions.append(-1)  # 非位置标记
        
        # 创建颜色映射
        unique_positions = sorted(set([p for p in positions if p >= 0]))
        color_map = {}
        palette = sns.color_palette("viridis", max(unique_positions) + 1 if unique_positions else 1)
        
        for pos in unique_positions:
            color_map[pos] = palette[pos]
        
        # 绘制散点图
        plt.figure(figsize=(12, 10))
        
        # 为每个位置绘制一个散点图
        for pos in unique_positions:
            indices = [i for i, p in enumerate(positions) if p == pos]
            if indices:
                plt.scatter(
                    reduced_embeddings[indices, 0],
                    reduced_embeddings[indices, 1],
                    color=color_map[pos],
                    label=f'位置 {pos}',
                    alpha=0.7,
                    s=50
                )
        
        plt.title('按位置着色的嵌入空间分布')
        plt.xlabel('PCA维度1')
        plt.ylabel('PCA维度2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "position_visualization.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Position visualization completed")
    
    def run_all_analyses(self, test_data_path, example_address="2001:db8::", num_mask_samples=100):
        """运行所有分析"""
        logging.info("Starting all analyses")
        
        # 1. 分析嵌入空间
        embed_stats = self.analyze_embedding_space()
        
        # 2. 聚类分析
        cluster_stats = self.cluster_analysis(embed_stats['filtered_embeddings'])
        
        # 3. 位置熵分析
        entropy_stats = self.position_entropy_analysis(test_data_path)
        
        # 4. 位置相关性分析
        correlation_stats = self.position_correlation_analysis(test_data_path)
        
        # 5. 掩码预测评估
        mask_stats = self._evaluate_mask_prediction(test_data_path, num_mask_samples)
        
        # 6. 注意力分析
        attention_stats = self.visualize_attention(example_address)
        
        # 7. 按nybble值可视化
        self.visualize_by_nybble_value()
        
        # 8. 按位置可视化
        self.visualize_by_position()
        
        logging.info("All analyses completed")
        
        # 返回所有统计结果
        return {
            'embedding_stats': embed_stats,
            'cluster_stats': cluster_stats,
            'entropy_stats': entropy_stats,
            'correlation_stats': correlation_stats,
            'mask_stats': mask_stats,
            'attention_stats': attention_stats
        }


def main():
    """主函数"""
    import argparse
    
        
    parser = argparse.ArgumentParser(description='IPv6 BERT模型可视化与分析工具')
    parser.add_argument('--model_path', type=str, default='d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/models/01/best_bert_model.pt', help='BERT模型路径')
    parser.add_argument('--vocab_path', type=str, default='d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/processed/01/vocabulary.json', help='词汇表路径')
    parser.add_argument('--test_data', type=str, default='d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/processed/01/test_masked_sequences.txt', help='测试数据路径')
    parser.add_argument('--output_dir', type=str, default='D:/bigchuang/ipv6地址论文/10-6VecLM/BERT/visualizations/01', help='输出目录')
    parser.add_argument('--example_address', type=str, default='2001:4ca0:2001:12:225:90ff:fe1a:d67a', help='用于注意力分析的示例地址')
    parser.add_argument('--num_mask_samples', type=int, default=100, help='掩码预测评估的样本数')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = IPv6BERTVisualizer(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir
    )
    
    # 运行所有分析
    stats = visualizer.run_all_analyses(
        test_data_path=args.test_data,
        example_address=args.example_address,
        num_mask_samples=args.num_mask_samples
    )
    
    # 保存汇总统计结果
    summary = {
        '嵌入空间': {
            '平均相似度': f"{stats['embedding_stats']['avg_sim']:.2f}±{stats['embedding_stats']['std_sim']:.2f}",
            '相似度范围': f"[{stats['embedding_stats']['min_sim']:.2f}, {stats['embedding_stats']['max_sim']:.2f}]",
            '平均向量范数': f"{stats['embedding_stats']['avg_norm']:.2f}"
        },
        '聚类分析': {
            '轮廓系数': f"{stats['cluster_stats']['silhouette']:.2f}",
            'Davies-Bouldin指数': f"{stats['cluster_stats']['db_score']:.2f}",
            'Calinski-Harabasz指数': f"{stats['cluster_stats']['ch_score']:.0f}"
        },
        '位置熵': {
            '前缀平均熵': f"{stats['entropy_stats']['prefix_entropy']:.2f} bits",
            '中间部分平均熵': f"{stats['entropy_stats']['middle_entropy']:.2f} bits",
            '后缀平均熵': f"{stats['entropy_stats']['suffix_entropy']:.2f} bits"
        },
        '位置相关性': {
            '相邻位置平均相关系数': f"{stats['correlation_stats']['avg_adjacent_corr']:.2f}",
            '非相邻位置平均相关系数': f"{stats['correlation_stats']['avg_non_adjacent_corr']:.2f}",
            '显著相关位置对数量': len(stats['correlation_stats']['significant_pairs'])
        },
        '掩码预测': {
            '准确率': f"{stats['mask_stats']['accuracy']:.1%}",
            'Top-5准确率': f"{stats['mask_stats']['top5_accuracy']:.1%}"
        }
    }
    
    # 将汇总结果保存为JSON
    with open(os.path.join(args.output_dir, 'summary_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Summary statistics saved to {os.path.join(args.output_dir, 'summary_stats.json')}")


if __name__ == '__main__':
    main()