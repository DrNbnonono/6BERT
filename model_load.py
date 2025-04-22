import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import logging
import argparse
import random
import ipaddress
from tqdm import tqdm
import math
from transformers import BertModel, BertConfig

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

class BERTModelLoader:
    """BERT模型加载器，用于加载预训练的BERT模型和词嵌入"""
    def __init__(self, model_path=None, embeddings_path=None, vocab_path=None):
        self.model = None
        self.embeddings = None
        self.word2id = None
        self.id2word = None
        self.state_dict = None
        
        # 加载词嵌入
        if embeddings_path and os.path.exists(embeddings_path):
            self.load_embeddings(embeddings_path)
            logging.info(f"已加载词嵌入: {embeddings_path}")
        
        # 加载词汇表
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocabulary(vocab_path)
            logging.info(f"已加载词汇表: {vocab_path}")
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logging.info(f"已加载模型: {model_path}")
    
    def load_vocabulary(self, vocab_path):
        """加载词汇表"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
        self.id2word = {v: k for k, v in self.word2id.items()}
    
    def load_embeddings(self, embeddings_path):
        """加载词嵌入"""
        try:
            # 尝试加载词嵌入字典
            self.embeddings = np.load(embeddings_path, allow_pickle=True).item()
            logging.info(f"已加载词嵌入字典: {embeddings_path}")
        except:
            # 如果失败，尝试加载词嵌入矩阵
            try:
                self.embeddings = np.load(embeddings_path)
                logging.info(f"已加载词嵌入矩阵: {embeddings_path}")
            except Exception as e:
                logging.error(f"加载词嵌入失败: {e}")
                self.embeddings = None
    
    def load_model(self, model_path):
        """加载预训练的模型"""
        try:
            # 加载模型状态字典
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            
            # 检查加载的是模型对象还是状态字典
            if isinstance(state_dict, dict) and not hasattr(state_dict, 'eval'):
                # 如果是状态字典，先不设置模型，等到main函数中创建模型后再加载
                self.state_dict = state_dict
                self.model = None
                logging.info(f"已加载模型状态字典: {model_path}")
            else:
                # 如果是模型对象，直接设置
                self.model = state_dict
                self.state_dict = None
                logging.info(f"已加载模型对象: {model_path}")
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            self.model = None
            self.state_dict = None
    
    def get_word_id(self, word):
        """获取词的ID"""
        return self.word2id.get(word, self.word2id.get("[UNK]", 1))
    
    def get_id_word(self, idx):
        """获取ID对应的词"""
        return self.id2word.get(idx, "[UNK]")
    
    def get_embedding(self, word):
        """获取词的嵌入向量"""
        if self.embeddings is None:
            return None
        
        if isinstance(self.embeddings, dict):
            return self.embeddings.get(word, None)
        else:
            idx = self.get_word_id(word)
            if idx < len(self.embeddings):
                return self.embeddings[idx]
            return None

def greedy_decode(model, loader, src, src_padding_mask, max_len, start_symbol, temperature=1.0, device='cpu'):
    """
    贪婪解码
    
    Args:
        model: Transformer模型
        loader: BERTModelLoader实例
        src: 源序列
        src_padding_mask: 源序列掩码
        max_len: 最大生成长度
        start_symbol: 起始符号ID
        temperature: 温度参数
        device: 设备
    
    Returns:
        生成的序列
    """
    model.eval()
    
    # 初始化目标序列
    memory = model.transformer.encoder(src, src_key_padding_mask=src_padding_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    
    for i in range(max_len-1):
        # 生成目标掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
        
        # 解码
        out = model.transformer.decoder(ys, memory, tgt_mask=tgt_mask)
        out = model.output_layer(out)
        
        # 获取最后一个时间步的输出
        out = out[:, -1]
        
        # 计算与词嵌入的相似度
        similarity = torch.matmul(out, model.embedding.weight.transpose(0, 1))
        
        # 应用温度
        similarity = similarity / temperature
        
        # 获取概率分布
        probs = F.softmax(similarity, dim=-1)
        
        # 采样下一个词
        next_word = torch.multinomial(probs, 1)
        
        # 添加到目标序列
        ys = torch.cat([ys, next_word], dim=1)
    
    return ys

def beam_search_decode(model, loader, src, src_padding_mask, max_len, start_symbol, beam_size=5, temperature=1.0, device='cpu'):
    """
    束搜索解码
    
    Args:
        model: Transformer模型
        loader: BERTModelLoader实例
        src: 源序列
        src_padding_mask: 源序列掩码
        max_len: 最大生成长度
        start_symbol: 起始符号ID
        beam_size: 束大小
        temperature: 温度参数
        device: 设备
    
    Returns:
        生成的序列列表，按分数排序
    """
    model.eval()
    
    # 编码源序列
    memory = model.transformer.encoder(src, src_key_padding_mask=src_padding_mask)
    
    # 初始化束
    sequences = [(torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device), 0.0)]
    
    for i in range(max_len-1):
        all_candidates = []
        
        for seq, score in sequences:
            # 如果序列已经结束，直接添加到候选集
            if seq[0, -1].item() == loader.get_word_id("[PAD]"):
                all_candidates.append((seq, score))
                continue
            
            # 生成目标掩码
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(device)
            
            # 解码
            out = model.transformer.decoder(seq, memory, tgt_mask=tgt_mask)
            out = model.output_layer(out)
            
            # 获取最后一个时间步的输出
            out = out[:, -1]
            
            # 计算与词嵌入的相似度
            similarity = torch.matmul(out, model.embedding.weight.transpose(0, 1))
            
            # 应用温度
            similarity = similarity / temperature
            
            # 获取概率分布
            probs = F.softmax(similarity, dim=-1)
            
            # 获取top-k个候选
            topk_probs, topk_indices = torch.topk(probs, beam_size)
            
            # 添加到候选集
            for j in range(beam_size):
                prob = topk_probs[0, j].item()
                idx = topk_indices[0, j].item()
                new_seq = torch.cat([seq, torch.ones(1, 1).fill_(idx).type(torch.long).to(device)], dim=1)
                new_score = score - math.log(prob)  # 使用负对数似然作为分数
                all_candidates.append((new_seq, new_score))
        
        # 选择top-k个候选
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=False)[:beam_size]
    
    return sequences

def generate_addresses(model, loader, seed_data, num_samples=1000, max_len=32, temperature=0.1, beam_size=None, device='cpu'):
    """
    生成IPv6地址 - 基于前64位预测后64位
    
    Args:
        model: Transformer模型
        loader: BERTModelLoader实例
        seed_data: 种子数据
        num_samples: 生成样本数量
        max_len: 最大生成长度
        temperature: 温度参数
        beam_size: 束搜索大小，None表示使用贪婪搜索
        device: 设备
    
    Returns:
        生成的IPv6地址列表
    """
    generated_addresses = []
    
    # 固定编码器输入长度为16（对应IPv6地址的前64位）
    encoder_input_length = 16
    
    for i in tqdm(range(min(num_samples, len(seed_data))), desc="生成地址"):
        # 准备输入数据 - 只使用前16个词（前64位）
        if len(seed_data[i]) < encoder_input_length:
            # 跳过长度不足的数据
            continue
            
        src_words = seed_data[i][:encoder_input_length]
        src_ids = [loader.get_word_id(w) for w in src_words]
        src = torch.LongTensor([src_ids]).to(device)
        
        # 创建源序列掩码
        src_padding_mask = (src == loader.get_word_id("[PAD]"))
        
        # 获取起始符号 - 使用第17个词作为起始符号
        if len(seed_data[i]) > encoder_input_length:
            start_symbol = loader.get_word_id(seed_data[i][encoder_input_length])
        else:
            # 如果没有第17个词，使用PAD作为起始符号
            start_symbol = loader.get_word_id("[PAD]")
        
        # 生成序列 - 生成后16个词（后64位）
        if beam_size:
            sequences = beam_search_decode(model, loader, src, src_padding_mask, 16, 
                                          start_symbol, beam_size, temperature, device)
            # 取分数最高的序列
            predict = sequences[0][0].cpu().numpy()[0]
        else:
            predict = greedy_decode(model, loader, src, src_padding_mask, 16, 
                                   start_symbol, temperature, device).cpu().numpy()[0]
        
        # 合并源序列和生成序列 - 前16个词 + 后16个词
        full_sequence = np.concatenate([src_ids, predict[1:]])  # 去掉起始符号
        
        # 将ID转换为词
        predict_words = [loader.get_id_word(idx) for idx in full_sequence]
        
        # 构建IPv6地址字符串
        address_parts = []
        current_part = ""
        
        for word in predict_words:
            if word == "[PAD]" or word == "[UNK]" or word == "[CLS]" or word == "[SEP]" or word == "[MASK]":
                continue
            # 只取第一个字符（nybble值）
            if len(word) > 0:
                current_part += word[0]
            if len(current_part) == 4:
                address_parts.append(current_part)
                current_part = ""
        
        # 处理最后一部分（如果有）
        if current_part:
            address_parts.append(current_part)
        
        # 构建完整地址
        ipv6_address = ":".join(address_parts)
        
        # 验证地址格式
        try:
            ipaddress.IPv6Address(ipv6_address)
            generated_addresses.append(ipv6_address)
        except:
            # 跳过无效地址
            continue
    
    # 去重
    generated_addresses = list(set(generated_addresses))
    
    return generated_addresses

def write_data(addresses, output_path):
    """将生成的地址写入文件"""
    with open(output_path, "w") as f:
        for address in addresses:
            f.write(address + "\n")
    logging.info(f"已将 {len(addresses)} 个地址写入: {output_path}")

def load_seed_data(data_path, max_samples=100000):
    """加载种子数据"""
    seed_data = []
    with open(data_path, "r") as f:
        for line in f:
            words = line.strip().split()
            # 过滤掉特殊标记
            words = [w for w in words if w not in ["[CLS]", "[SEP]", "[MASK]"]]
            if words:
                seed_data.append(words)
            if len(seed_data) >= max_samples:
                break
    return seed_data

def main():
    parser = argparse.ArgumentParser(description="生成IPv6地址")
    parser.add_argument("--model_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/models/transformer/best_transformer_model.pt", help="模型路径")
    parser.add_argument("--vocab_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/processed/vocabulary.json", help="词汇表路径")
    parser.add_argument("--embeddings_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/models/word_embeddings.npy", help="词嵌入路径")
    parser.add_argument("--data_path", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/processed/word_sequences.txt", help="种子数据路径")
    parser.add_argument("--output_dir", type=str, default="d:/bigchuang/ipv6地址论文/10-6VecLM/BERT/data/generated", help="输出目录")
    parser.add_argument("--num_samples", type=int, default=10000, help="生成样本数量")
    parser.add_argument("--temperatures", type=str, default="0.1,0.2,0.5,1.0", help="温度参数列表，用逗号分隔")
    parser.add_argument("--beam_size", type=int, default=None, help="束搜索大小，None表示使用贪婪搜索")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和数据
    loader = BERTModelLoader(
        model_path=args.model_path,
        embeddings_path=args.embeddings_path,
        vocab_path=args.vocab_path
    )
    
    # 加载种子数据
    seed_data = load_seed_data(args.data_path, args.num_samples)
    logging.info(f"已加载 {len(seed_data)} 个种子数据")
    
    # 解析温度参数
    temperatures = [float(t) for t in args.temperatures.split(",")]
    
    # 加载模型配置 - 确保与训练时使用的配置一致
    config = {
        "d_model": 768,  # 与BERT隐藏层大小一致
        "nhead": 8,      # 注意力头数量
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
    }
    
    # 创建模型
    vocab_size = len(loader.word2id)
    model = TransformerModel(
        vocab_size, 
        config["d_model"], 
        config["nhead"], 
        config["dim_feedforward"],
        config["num_encoder_layers"], 
        config["num_decoder_layers"], 
        config["dropout"]
    )
    
    # 加载预训练权重
    if loader.model is not None:
        # 如果已经加载了完整模型，直接使用
        model = loader.model
    elif hasattr(loader, 'state_dict') and loader.state_dict is not None:
        # 如果加载了状态字典，应用到模型
        model.load_state_dict(loader.state_dict)
    else:
        # 尝试直接从文件加载状态字典
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)))
        except Exception as e:
            logging.error(f"加载模型权重失败: {e}")
            return
    
    model.to(args.device)
    model.eval()
    
    # 对每个温度参数生成地址
    for temp in temperatures:
        logging.info(f"使用温度 {temp} 生成地址")
        
        # 生成地址
        addresses = generate_addresses(
            model, 
            loader, 
            seed_data, 
            args.num_samples, 
            temperature=temp,
            beam_size=args.beam_size,
            device=args.device
        )
        
        # 写入文件
        output_path = os.path.join(args.output_dir, f"generated_addresses_t{temp:.3f}.txt")
        write_data(addresses, output_path)

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        logging.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("未检测到GPU，将使用CPU生成")
    
    main()