import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import argparse
from torch.cuda.amp import autocast, GradScaler
from transformers import BertConfig, BertForMaskedLM, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IPv6BERTDataset(Dataset):
    """IPv6地址BERT数据集"""
    def __init__(self, file_path, vocab_path, max_len=34, masked_lm=True):
        """
        初始化数据集
        
        Args:
            file_path: 词序列文件路径
            vocab_path: 词汇表文件路径
            max_len: 最大序列长度
            masked_lm: 是否使用掩码语言模型数据
        """
        self.max_len = max_len
        self.masked_lm = masked_lm
        
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
        
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.pad_id = self.word2id.get("[PAD]", 0)
        self.mask_id = self.word2id.get("[MASK]", 4)
        
        # 读取序列
        logging.info(f"从 {file_path} 加载数据...")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.sequences = [line.strip().split() for line in f]
        
        logging.info(f"加载了 {len(self.sequences)} 个序列")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        words = self.sequences[idx]
        
        # 转换为ID
        input_ids = [self.word2id.get(word, self.word2id["[UNK]"]) for word in words]
        
        # 创建注意力掩码
        attention_mask = [1] * len(input_ids)
        
        # 如果是MLM，创建标签
        if self.masked_lm:
            labels = [-100 if word == "[MASK]" else self.word2id.get(word, self.word2id["[UNK]"]) 
                     for word in words]
        else:
            labels = input_ids.copy()
        
        # 填充到最大长度
        if len(input_ids) < self.max_len:
            padding_length = self.max_len - len(input_ids)
            input_ids = input_ids + [self.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length
        else:
            input_ids = input_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
            labels = labels[:self.max_len]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def train_bert(config):
    """训练BERT模型"""
    # 设置随机种子
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(config["seed"])
        logging.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.warning("未检测到GPU，将使用CPU训练")
    
    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # 加载词汇表
    with open(config["vocab_path"], 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    
    vocab_size = len(word2id)
    logging.info(f"词汇表大小: {vocab_size}")
    
    # 创建数据集和数据加载器
    train_dataset = IPv6BERTDataset(
        file_path=config["train_data_path"],
        vocab_path=config["vocab_path"],
        max_len=config["max_seq_length"],
        masked_lm=True
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    val_dataset = IPv6BERTDataset(
        file_path=config["val_data_path"],
        vocab_path=config["vocab_path"],
        max_len=config["max_seq_length"],
        masked_lm=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    # 创建BERT配置
    bert_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        hidden_dropout_prob=config["hidden_dropout_prob"],
        attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
        max_position_embeddings=config["max_seq_length"],
        type_vocab_size=1,  # 我们只有一种类型的输入
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=word2id["[PAD]"],
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
    )
    
    # 创建模型
    model = BertForMaskedLM(bert_config)
    model.to(device)
    
    # 创建优化器和学习率调度器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config["learning_rate"],
        eps=config["adam_epsilon"]
    )
    
    # 计算总训练步数
    total_steps = len(train_dataloader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 创建混合精度训练的缩放器
    scaler = GradScaler()
    
    # 训练循环
    logging.info("开始训练...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    batch_losses = []  # 记录每个batch的损失
    
    start_time = time.time()  # 记录训练开始时间
    
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0
        batch_count = 0
        epoch_start_time = time.time()  # 记录epoch开始时间
        
        # 创建进度条
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch in progress_bar:
            batch_start_time = time.time()  # 记录batch开始时间
            
            # 将数据移动到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播（使用混合精度）
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            # 反向传播（使用混合精度）
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 更新学习率
            scheduler.step()
            
            # 记录batch损失
            batch_losses.append(loss.item())
            
            # 更新进度条
            epoch_loss += loss.item()
            batch_count += 1
            batch_time = time.time() - batch_start_time
            batch_speed = config["batch_size"] / batch_time if batch_time > 0 else 0
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "gpu_mem": f"{torch.cuda.memory_allocated(0)/1024**3:.2f}GB" if torch.cuda.is_available() else "N/A",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "batch/s": f"{batch_speed:.1f}"
            })
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']} - Train Loss: {avg_train_loss:.4f}, 耗时: {epoch_time:.2f}秒")
        
        # 验证
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                val_batch_count += 1
        
        # 计算平均验证损失
        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(config["output_dir"], "best_bert_model.pt")
            torch.save(model.state_dict(), model_path)
            logging.info(f"保存最佳模型到 {model_path}")
    
    # 训练完成
    elapsed_time = time.time() - start_time
    logging.info(f"训练完成，耗时: {elapsed_time/60:.2f}分钟")
    
    # 保存最终模型
    final_model_path = os.path.join(config["output_dir"], "final_bert_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"保存最终模型到 {final_model_path}")
    
    # 保存基础模型（不包含MLM头）
    base_model = BertModel(bert_config, add_pooling_layer=False)
    base_model.load_state_dict({k.replace('bert.', ''): v for k, v in model.bert.state_dict().items()})
    base_model_path = os.path.join(config["output_dir"], "bert_base_model.pt")
    torch.save(base_model.state_dict(), base_model_path)
    logging.info(f"保存基础模型到 {base_model_path}")
    
    # 绘制损失曲线
    plot_loss_curves(train_losses, val_losses, batch_losses, config)
    
    # 提取并可视化词嵌入
    visualize_embeddings(model, word2id, config)
    
    return model, base_model

def plot_loss_curves(epoch_losses, val_losses, batch_losses, config):
    """绘制训练损失曲线"""
    logging.info("绘制训练损失曲线...")
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制每个epoch的平均损失
    epochs = range(1, len(epoch_losses) + 1)
    ax1.plot(epochs, epoch_losses, 'b-', marker='o', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', marker='x', label='Validation Loss')
    ax1.set_title('Epoch Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制每个batch的损失
    batches = range(1, len(batch_losses) + 1)
    ax2.plot(batches, batch_losses, 'r-', alpha=0.5)
    ax2.set_title('Batch Loss')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    # 添加平滑曲线（移动平均）
    window_size = min(100, len(batch_losses) // 10)
    if window_size > 0:
        smoothed_losses = []
        for i in range(len(batch_losses) - window_size + 1):
            smoothed_losses.append(sum(batch_losses[i:i+window_size]) / window_size)
        ax2.plot(range(window_size, len(batch_losses) + 1), smoothed_losses, 'g-', linewidth=2, 
                label=f'Moving Avg (window={window_size})')
        ax2.legend()
    
    # 设置图表样式
    plt.tight_layout()
    
    # 保存图表
    loss_curve_path = os.path.join(config["output_dir"], "loss_curves.png")
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    logging.info(f"损失曲线已保存到 {loss_curve_path}")
    
    # 关闭图表
    plt.close()

def visualize_embeddings(model, word2id, config):
    """改进的词嵌入可视化函数"""
    logging.info("提取词嵌入并进行可视化...")
    
    # 提取词嵌入矩阵
    embeddings = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
    
    # 创建词嵌入字典
    word_embeddings = {}
    for word, idx in word2id.items():
        if idx < len(embeddings):  # 确保索引有效
            word_embeddings[word] = embeddings[idx]
    
    # 保存词嵌入
    embeddings_path = os.path.join(config["output_dir"], "word_embeddings.npy")
    np.save(embeddings_path, embeddings)
    logging.info(f"词嵌入已保存到 {embeddings_path}")
    
    # 使用PCA先降维到50维，再用t-SNE降维到2维
    words = list(word_embeddings.keys())
    vectors = np.array([word_embeddings[word] for word in words])
    
    # 过滤掉特殊标记
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    mask = [word not in special_tokens for word in words]
    filtered_words = [word for i, word in enumerate(words) if mask[i]]
    filtered_vectors = vectors[mask]
    
    # 对IPv6地址词进行特殊处理 - 提取nybble和位置信息
    nybbles = []
    positions = []
    for word in filtered_words:
        if len(word) >= 1:
            nybbles.append(word[0])
            if len(word) > 1:
                positions.append(word[1:])
            else:
                positions.append('')
        else:
            nybbles.append('')
            positions.append('')
    
    # 如果词汇量太大，随机采样
    max_words = 2000  # 减少数量以获得更好的可视化效果
    if len(filtered_words) > max_words:
        indices = np.random.choice(len(filtered_words), max_words, replace=False)
        filtered_words = [filtered_words[i] for i in indices]
        filtered_vectors = filtered_vectors[indices]
        nybbles = [nybbles[i] for i in indices]
        positions = [positions[i] for i in indices]
    
    logging.info(f"对 {len(filtered_words)} 个词应用降维...")
    
    # 先使用PCA降维到50维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=42)
    pca_vectors = pca.fit_transform(filtered_vectors)
    
    # 然后使用t-SNE降维到2维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, 
                early_exaggeration=12, learning_rate=200, n_iter=1000)
    reduced_vectors = tsne.fit_transform(pca_vectors)
    
    # 创建可视化数据
    df = pd.DataFrame({
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1],
        'word': filtered_words,
        'nybble': nybbles,
        'position': positions
    })
    
    # 保存t-SNE数据
    tsne_data_path = os.path.join(config["output_dir"], "tsne_data.csv")
    df.to_csv(tsne_data_path, index=False)
    logging.info(f"t-SNE数据已保存到 {tsne_data_path}")
    
    # 按nybble值着色
    plt.figure(figsize=(16, 12))
    unique_nybbles = sorted(list(set(nybbles)))
    palette = sns.color_palette("hsv", len(unique_nybbles))
    
    for i, nybble in enumerate(unique_nybbles):
        mask = df['nybble'] == nybble
        plt.scatter(df[mask]['x'], df[mask]['y'], 
                   color=palette[i], label=nybble, alpha=0.7, s=30)
    
    plt.title('t-SNE Visualization of IPv6 Word Embeddings (colored by nybble)')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    
    nybble_plot_path = os.path.join(config["output_dir"], "tsne_nybble.png")
    plt.savefig(nybble_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"按nybble着色的t-SNE图已保存到 {nybble_plot_path}")
    
    # 按position值着色 - 只显示前16个位置以获得更好的可视化效果
    plt.figure(figsize=(16, 12))
    position_counts = df['position'].value_counts()
    top_positions = position_counts[:16].index.tolist()  # 只取前16个最常见的位置
    
    palette = sns.color_palette("tab20", len(top_positions))
    
    for i, pos in enumerate(top_positions):
        mask = df['position'] == pos
        plt.scatter(df[mask]['x'], df[mask]['y'], 
                   color=palette[i], label=pos, alpha=0.7, s=30)
    
    # 其他位置统一用一种颜色
    mask = ~df['position'].isin(top_positions)
    if mask.any():
        plt.scatter(df[mask]['x'], df[mask]['y'], 
                   color='gray', label='Other', alpha=0.5, s=20)
    
    plt.title('t-SNE Visualization of IPv6 Word Embeddings (colored by position)')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    
    position_plot_path = os.path.join(config["output_dir"], "tsne_position.png")
    plt.savefig(position_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"按position着色的t-SNE图已保存到 {position_plot_path}")
    
    # 保存词嵌入和词汇表的映射关系
    embeddings_dict = {}
    for word, idx in word2id.items():
        if idx < len(embeddings):
            embeddings_dict[word] = embeddings[idx].tolist()
    
    with open(os.path.join(config["output_dir"], "embeddings_dict.json"), 'w', encoding='utf-8') as f:
        json.dump(embeddings_dict, f, ensure_ascii=False, indent=2)
    
    logging.info("词嵌入可视化完成")
    
def extract_embeddings(model_path, vocab_path, output_dir):
    """从预训练模型中提取词嵌入"""
    # 加载词汇表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    
    # 加载模型配置
    bert_config = BertConfig(
        vocab_size=len(word2id),
        hidden_size=768,  # 默认值，可能需要根据实际模型调整
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )
    
    # 创建模型
    model = BertForMaskedLM(bert_config)
    
    # 加载预训练权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # 提取词嵌入
    embeddings = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存词嵌入
    np.save(os.path.join(output_dir, "bert_embeddings.npy"), embeddings)
    
    # 创建词嵌入字典
    embeddings_dict = {}
    for word, idx in word2id.items():
        if idx < len(embeddings):
            embeddings_dict[word] = embeddings[idx].tolist()
    
    # 保存词嵌入字典
    with open(os.path.join(output_dir, "bert_embeddings_dict.json"), 'w', encoding='utf-8') as f:
        json.dump(embeddings_dict, f)
    
    logging.info(f"词嵌入已提取并保存到 {output_dir}")
    
    return embeddings_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练IPv6地址BERT模型")
    parser.add_argument("--train_data_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/processed/02/train_masked_sequences.txt", help="训练数据路径")
    parser.add_argument("--val_data_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/processed/02/val_masked_sequences.txt", help="验证数据路径")
    parser.add_argument("--vocab_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/processed/02/vocabulary.json", help="词汇表路径")
    parser.add_argument("--output_dir", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/models/02/", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--max_seq_length", type=int, default=34, help="最大序列长度")
    parser.add_argument("--hidden_size", type=int, default=768, help="隐藏层大小")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="隐藏层数量")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="注意力头数量")
    parser.add_argument("--intermediate_size", type=int, default=3072, help="中间层大小")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="隐藏层dropout概率")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1, help="注意力dropout概率")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--num_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载器工作进程数")
    
    args = parser.parse_args()
    
    # 创建配置字典
    config = vars(args)
    
    # 训练模型
    model, base_model = train_bert(config)