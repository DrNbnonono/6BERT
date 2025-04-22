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
import math
import argparse
from torch.cuda.amp import autocast, GradScaler

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """基于Transformer的IPv6地址生成模型"""
    def __init__(self, vocab_size, d_model, nhead, dim_feedforward, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer层
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Linear(d_model, d_model)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播
        
        Args:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        
        Returns:
            输出序列 [batch_size, tgt_len, d_model]
        """
        # 嵌入和位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Transformer处理
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        
        # 输出层
        output = self.output_layer(output)
        
        return output

    def get_attention_weights(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        获取注意力权重
        
        Args:
            src: 源序列
            tgt: 目标序列
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        
        Returns:
            attention_weights: 注意力权重列表，每个元素是一个层的注意力权重
        """
        # 嵌入和位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # 存储注意力权重
        attention_weights = []
        
        # 创建一个钩子函数来捕获注意力权重
        def get_attention_hook(module, input, output):
            # 注意力权重通常是输出的第1个元素
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights.append(output[1])
        
        # 为每个编码器层的自注意力模块注册钩子
        hooks = []
        for i, layer in enumerate(self.transformer.encoder.layers):
            hook = layer.self_attn.register_forward_hook(get_attention_hook)
            hooks.append(hook)
        
        # 为每个解码器层的自注意力和交叉注意力模块注册钩子
        for i, layer in enumerate(self.transformer.decoder.layers):
            hook = layer.self_attn.register_forward_hook(get_attention_hook)
            hooks.append(hook)
            hook = layer.multihead_attn.register_forward_hook(get_attention_hook)
            hooks.append(hook)
        
        # 前向传播以获取注意力权重
        memory = self.transformer.encoder(src, mask=src_mask)
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return attention_weights
        
class IPv6TransformerDataset(Dataset):
    """IPv6地址Transformer数据集"""
    def __init__(self, file_path, vocab_path, input_len=16, max_len=32):
        """
        初始化数据集
        
        Args:
            file_path: 词序列文件路径
            vocab_path: 词汇表文件路径
            input_len: 输入序列长度
            max_len: 最大序列长度
        """
        self.input_len = input_len
        self.max_len = max_len
        
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
        
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.pad_id = self.word2id.get("[PAD]", 0)
        
        # 读取序列
        logging.info(f"从 {file_path} 加载数据...")
        with open(file_path, 'r', encoding='utf-8') as f:
            sequences = [line.strip().split() for line in f]
        
        # 过滤掉特殊标记并保留有效序列
        self.sequences = []
        for seq in sequences:
            # 移除[CLS]和[SEP]等特殊标记
            filtered_seq = [word for word in seq if word not in ["[CLS]", "[SEP]", "[MASK]"]]
            if len(filtered_seq) >= self.max_len:
                self.sequences.append(filtered_seq[:self.max_len])
        
        logging.info(f"加载了 {len(self.sequences)} 个有效序列")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # 输入序列（前input_len个词）
        src = torch.tensor([self.word2id.get(word, self.word2id["[UNK]"]) for word in seq[:self.input_len]], dtype=torch.long)
        
        # 目标序列（后max_len-input_len个词）
        tgt = torch.tensor([self.word2id.get(word, self.word2id["[UNK]"]) for word in seq[self.input_len:]], dtype=torch.long)
        
        return src, tgt

def train_transformer(config):
    """训练Transformer模型"""
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
    train_dataset = IPv6TransformerDataset(
        file_path=config["train_data_path"],
        vocab_path=config["vocab_path"],
        input_len=config["input_len"],
        max_len=config["max_len"]
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    val_dataset = IPv6TransformerDataset(
        file_path=config["val_data_path"],
        vocab_path=config["vocab_path"],
        input_len=config["input_len"],
        max_len=config["max_len"]
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    # 创建模型
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        nhead=config["nhead"],
        dim_feedforward=config["dim_feedforward"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dropout=config["dropout"]
    )
    
    # 加载BERT预训练的词嵌入
    if config["bert_embeddings_path"] and os.path.exists(config["bert_embeddings_path"]):
        logging.info(f"加载BERT预训练词嵌入: {config['bert_embeddings_path']}")
        bert_embeddings = np.load(config["bert_embeddings_path"], allow_pickle=True).item()
        
        # 创建嵌入矩阵
        embedding_matrix = torch.zeros(vocab_size, config["d_model"])
        
        # 填充嵌入矩阵
        for word, idx in word2id.items():
            if word in bert_embeddings:
                # 如果BERT嵌入维度与模型不同，需要调整
                bert_vector = torch.tensor(bert_embeddings[word])
                if bert_vector.size(0) != config["d_model"]:
                    # 使用线性投影调整维度
                    projection = nn.Linear(bert_vector.size(0), config["d_model"])
                    with torch.no_grad():
                        embedding_matrix[idx] = projection(bert_vector)
                else:
                    embedding_matrix[idx] = bert_vector
        
        # 加载预训练嵌入
        model.embedding.weight.data.copy_(embedding_matrix)
        logging.info("已加载BERT预训练词嵌入")
    
    model.to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["lr_step_size"], gamma=config["lr_gamma"])
    
    # 创建损失函数
    criterion = nn.CosineEmbeddingLoss()
    
    # 创建混合精度训练的缩放器
    scaler = GradScaler()
    
    # 训练循环
    logging.info("开始训练...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0
        
        # 创建进度条
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for src, tgt in progress_bar:
            # 将数据移动到设备
            src, tgt = src.to(device), tgt.to(device)
            
            # 创建目标掩码（防止看到未来信息）
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # 创建源序列掩码
            src_padding_mask = (src == word2id["[PAD]"]).to(device)
            tgt_padding_mask = (tgt == word2id["[PAD]"]).to(device)
            
            # 创建目标输入和输出
            tgt_input = tgt[:, :-1]  # 移除最后一个词
            tgt_output = tgt[:, 1:]  # 移除第一个词
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播（使用混合精度）
            with autocast():
                # 获取模型输出
                output = model(src, tgt_input, src_mask=src_padding_mask, tgt_mask=tgt_mask)
                
                # 获取词嵌入
                output_embeddings = output[:, -1]  # 取最后一个时间步
                target_embeddings = model.embedding(tgt_output[:, -1])  # 取目标的最后一个词
                
                # 计算余弦相似度损失
                batch_size = output_embeddings.size(0)
                target = torch.ones(batch_size).to(device)  # 目标为1，表示相似
                loss = criterion(output_embeddings, target_embeddings, target)
            
            # 反向传播（使用混合精度）
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 更新进度条
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{epoch_loss/(progress_bar.n+1):.4f}"})
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for src, tgt in tqdm(val_dataloader, desc="Validating"):
                src, tgt = src.to(device), tgt.to(device)
                
                # 创建目标掩码
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
                
                # 创建源序列掩码
                src_padding_mask = (src == word2id["[PAD]"]).to(device)
                tgt_padding_mask = (tgt == word2id["[PAD]"]).to(device)
                
                # 创建目标输入和输出
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # 前向传播
                output = model(src, tgt_input, src_mask=src_padding_mask, tgt_mask=tgt_mask)
                
                # 获取词嵌入
                output_embeddings = output[:, -1]
                target_embeddings = model.embedding(tgt_output[:, -1])
                
                # 计算损失
                batch_size = output_embeddings.size(0)
                target = torch.ones(batch_size).to(device)
                loss = criterion(output_embeddings, target_embeddings, target)
                
                val_loss += loss.item()
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(config["output_dir"], "best_transformer_model.pt")
            torch.save(model.state_dict(), model_path)
            logging.info(f"保存最佳模型到 {model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(config["output_dir"], "final_transformer_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"保存最终模型到 {final_model_path}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config["num_epochs"]+1), train_losses, label='Train Loss')
    plt.plot(range(1, config["num_epochs"]+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(config["output_dir"], "transformer_loss_curve.png")
    plt.savefig(loss_plot_path)
    logging.info(f"损失曲线已保存到 {loss_plot_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练IPv6地址Transformer模型")
    parser.add_argument("--train_data_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/processed/train_sequences.txt", help="训练数据路径")
    parser.add_argument("--val_data_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/processed/val_sequences.txt", help="验证数据路径")
    parser.add_argument("--vocab_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/processed/vocabulary.json", help="词汇表路径")
    parser.add_argument("--bert_embeddings_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/models/word_embeddings.npy", help="BERT词嵌入路径")
    parser.add_argument("--output_dir", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/models/transformer", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    
    args = parser.parse_args()
    
    # 训练配置
    config = {
        "train_data_path": args.train_data_path,
        "val_data_path": args.val_data_path,
        "vocab_path": args.vocab_path,
        "bert_embeddings_path": args.bert_embeddings_path,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "input_len": 16,  # 输入序列长度
        "max_len": 32,    # 最大序列长度
        "d_model": 768,   # 与BERT隐藏层大小一致
        "nhead": 8,       # 注意力头数
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "lr_step_size": 5,
        "lr_gamma": 0.1,
    }
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        logging.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        # 设置CUDA性能优化
        torch.backends.cudnn.benchmark = True
    else:
        logging.warning("未检测到GPU，将使用CPU训练，这可能会很慢")
    
    # 训练模型
    model = train_transformer(config)
    
    logging.info("训练完成！")