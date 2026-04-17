

import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from swinv2_net import DFNet
from data import test_dataset
from options import opt
from collections import OrderedDict
import time
from os.path import join

# 忽略meshgrid警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

########################################
#  1) 定义 F-measure 计算器 (修复维度不匹配+负数输出)
########################################
class Metric:
    def __init__(self):
        self.prec_sum = 0
        self.recall_sum = 0
        self.count = 0

    def update(self, pred, gt):
        # 确保输入是numpy数组
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
        
        # 核心修复1：处理宽高颠倒问题（统一为H×W）
        if pred.shape != gt.shape:
            # 尝试转置GT匹配预测图
            if gt.T.shape == pred.shape:
                gt = gt.T
            else:
                # 插值到相同尺寸
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 核心修复2：预测值先归一化到0-1（解决负数问题）
        pred = np.clip(pred, 0, 1)
        
        # 二值化（严格按论文Fw公式）
        pred_bin = (pred > 0.5).astype(np.uint8)
        gt_bin   = (gt > 128).astype(np.uint8)

        # 关键：先检查GT是否为空（避免除以0）
        if np.sum(gt_bin) == 0:
            return  # 无标注的样本不参与计算

        tp = (pred_bin * gt_bin).sum()
        fp = (pred_bin * (1 - gt_bin)).sum()
        fn = ((1 - pred_bin) * gt_bin).sum()

        # 论文标准Fw计算（1.3权重）
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        fmeasure  = (1.3 * precision * recall) / (0.3 * precision + recall + 1e-8)

        self.prec_sum += fmeasure
        self.count    += 1

    def get(self):
        if self.count == 0:
            return 0.0
        return self.prec_sum / self.count

########################################
#  2) 加载属性标签 TXT (修复文件名匹配)
########################################
def load_attribute_files(attr_file_path):
    """
    修复：
    1. 移除文件名后缀（如.jpg/.png）以匹配推理时的name
    2. 增强容错性，支持不同分隔符
    """
    attributes = ["BSO","CB","CIB","IC","LI","MSO","OF",
                  "SSO","SA","TC","BW","bRGB","bT"]

    attr_dict = {attr: set() for attr in attributes}

    try:
        print(f"正在从 {attr_file_path} 加载属性标签...")
        with open(attr_file_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            # 跳过标题行
            lines = [l.strip() for l in lines if l.strip()]
            if len(lines) == 0:
                print("警告：属性文件为空！")
                return attributes, attr_dict
            
            # 跳过第一行标题（如果包含表头）
            if 'name' in lines[0] or 'BSO' in lines[0]:
                lines = lines[1:]

            for line_idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 14:  # 1个文件名 + 13个属性
                    print(f"警告：第{line_idx+1}行格式错误，跳过: {line}")
                    continue
                
                # 核心修复：移除文件名后缀（如.jpg/.png）
                name = parts[0].split('.')[0]  # 只保留文件名前缀
                # 遍历13个属性
                for i in range(13):
                    attr_name = attributes[i]
                    attr_value = parts[i+1]
                    try:
                        if float(attr_value) > 0:
                            attr_dict[attr_name].add(name)
                    except ValueError:
                        print(f"警告：第{line_idx+1}行{attr_name}值无效: {attr_value}")
                        continue
        print(f"属性标签加载成功，各属性样本数：")
        for attr in attributes:
            print(f"  {attr}: {len(attr_dict[attr])}")
    except FileNotFoundError:
        print(f"错误: 属性标签文件 {attr_file_path} 不存在！")
        exit(1)
    except Exception as e:
        print(f"错误：加载属性标签失败: {str(e)}")
        exit(1)
        
    return attributes, attr_dict

########################################
#  3) 加载模型 (彻底解决Memory Bank维度不匹配+输出负数)
########################################
# 核心修改1：初始化模型时指定memory_bank的feature_dim=64（和训练权重匹配）
# 先修改DFNet的初始化参数，强制memory_bank使用64维
class ModifiedDFNet(DFNet):
    def __init__(self):
        super().__init__(use_memory_bank=True)
        # 重新初始化memory_bank为64维（覆盖原128维）
        if self.memory_bank is not None:
            from swinv2_net1 import FusionMemoryBank
            self.memory_bank = FusionMemoryBank(memory_size=256, feature_dim=64)
            # 同时更新解码器中的memory_bank引用
            self.decoderi.memory_bank = self.memory_bank
            self.decodert.memory_bank = self.memory_bank
            self.decoder.memory_bank = self.memory_bank

model = ModifiedDFNet()

# 加载权重
ckpt = torch.load(opt.test_model, weights_only=True)

new_state_dict = OrderedDict()
model_state_dict = model.state_dict()  # 获取模型默认参数

# 过滤所有memory_bank相关参数（应急方案）
skip_keywords = [
    "memory_bank",
    "decoderi.memory_bank",
    "decodert.memory_bank",
    "decoder.memory_bank",
]

for k, v in ckpt.items():
    # 处理DataParallel的module.前缀
    name = k[7:] if k.startswith("module.") else k

    # 过滤不匹配的参数
    if any(kw in name for kw in skip_keywords):
        print(f"[SKIP] 过滤掉不匹配参数: {name}")
        # 用模型默认参数填充
        if name in model_state_dict:
            new_state_dict[name] = model_state_dict[name]
        continue
    
    # 保留其他参数
    new_state_dict[name] = v

# 加载权重（strict=False允许部分参数不匹配）
print("Loading DFNet (修复Memory Bank维度，使用64维)...")
model.load_state_dict(new_state_dict, strict=False)

# 初始化未加载的参数
for name, param in model.named_parameters():
    if name not in new_state_dict:
        print(f"[INIT] 初始化未加载的参数: {name}")
        # 根据参数类型选择初始化方式
        if "weight" in name:
            if "conv" in name or "proj" in name:
                torch.nn.init.kaiming_normal_(param.data)
            elif "norm" in name:
                torch.nn.init.ones_(param.data)
        elif "bias" in name:
            torch.nn.init.zeros_(param.data)

model.cuda()
model.eval()

# 验证模型输出（修复负数问题）
with torch.no_grad():
    test_img = torch.randn(1,3,384,384).cuda()
    test_t = torch.randn(1,3,384,384).cuda()
    test_out = model(test_img, test_t, (384,384))
    # 先经过sigmoid再计算均值（解决负数）
    test_out_norm = test_out[2].sigmoid()
    print(f"模型验证输出均值 (sigmoid后): {test_out_norm.mean().item():.4f}")
    if test_out_norm.mean().item() < 1e-6:
        print("警告：模型输出接近全0，可能影响结果！")
    else:
        print("模型输出正常，非全0！")

########################################
#  4) 数据集测试循环 (增强调试)
########################################
test_data_root = opt.test_data_root
maps_path = opt.maps_path

test_sets = ['VT5000/Test']

for dataset in test_sets:
    save_path = join(maps_path, dataset)
    os.makedirs(save_path, exist_ok=True)

    dataset_path = join(test_data_root, dataset)
    test_loader = test_dataset(dataset_path, opt.testsize)

    # 加载VT5000属性标签
    if "VT5000" in dataset:
        attr_file_path = join(dataset_path, "attribute.txt")
        attributes, attr_files = load_attribute_files(attr_file_path)
        attr_metric = {attr: Metric() for attr in attributes}
    else:
        attr_files = None

    total_time = 0
    frame_count = 0
    zero_output_count = 0  # 统计全0输出数量

    for i in range(test_loader.size):
        image, t, gt, (H, W), name = test_loader.load_data()
        # 核心修复：移除文件名后缀以匹配属性标签
        name_prefix = name.split('.')[0]
        
        image = image.cuda()
        t     = t.cuda()
        shape = (W, H)

        # 推理
        with torch.no_grad():
            start_time = time.time()
            out_rgb, out_t, out_f, out_edge, out = model(image, t, shape)
            end_time = time.time()

        total_time += (end_time - start_time)
        frame_count += 1

        # 核心修复3：先经过sigmoid再处理（解决负数输出）
        res = out_f.sigmoid().data.cpu().numpy().squeeze()
        
        # 处理输出（验证是否全0）
        if np.max(res) < 1e-6:
            zero_output_count += 1
            print(f"警告：{name}推理输出全0！")
        
        # 归一化（防止除以0）
        res_min = res.min()
        res_max = res.max()
        if res_max - res_min < 1e-8:
            res_norm = np.zeros_like(res)
        else:
            res_norm = (res - res_min) / (res_max - res_min)
        # 确保归一化后在0-1之间
        res_norm = np.clip(res_norm, 0, 1)

        # 保存预测图
        cv2.imwrite(join(save_path, name), (res_norm * 255).astype(np.uint8))

        # 统计属性指标
        if attr_files is not None:
            # 处理GT
            if isinstance(gt, torch.Tensor):
                gt_np = gt.squeeze().cpu().numpy()
            else:
                gt_np = np.array(gt)
            
            # 遍历所有属性
            for attr in attributes:
                if name_prefix in attr_files[attr]:
                    attr_metric[attr].update(res_norm, gt_np)

    # 输出FPS
    avg_time = total_time / frame_count if frame_count > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"\n[{dataset}] FPS: {fps:.2f}, 全0输出数: {zero_output_count}/{frame_count}")

    # 输出属性指标
    if attr_files is not None:
        print("\n=== Attribute Results (F-measure Fw) ===")
        for attr in attributes:
            fw = attr_metric[attr].get()
            print(f"{attr}: {fw:.3f}")
        print("========================================\n")

print('Test Done!')






'''
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from swinv2_net128 import RealCLIPEnhancedFusionNet  # ★ 与训练保持一致
from data import test_dataset
from options import opt
from collections import OrderedDict
import time
from os.path import join

# 忽略meshgrid警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

########################################
#  1) 定义 F-measure 计算器 (修复维度不匹配+负数输出)
########################################
class Metric:
    def __init__(self):
        self.prec_sum = 0
        self.recall_sum = 0
        self.count = 0

    def update(self, pred, gt):
        # 确保输入是numpy数组
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()
        
        # 核心修复1：处理宽高颠倒问题（统一为H×W）
        if pred.shape != gt.shape:
            # 尝试转置GT匹配预测图
            if gt.T.shape == pred.shape:
                gt = gt.T
            else:
                # 插值到相同尺寸
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 核心修复2：预测值先归一化到0-1（解决负数问题）
        pred = np.clip(pred, 0, 1)
        
        # 二值化（严格按论文Fw公式）
        pred_bin = (pred > 0.5).astype(np.uint8)
        gt_bin   = (gt > 128).astype(np.uint8)

        # 关键：先检查GT是否为空（避免除以0）
        if np.sum(gt_bin) == 0:
            return  # 无标注的样本不参与计算

        tp = (pred_bin * gt_bin).sum()
        fp = (pred_bin * (1 - gt_bin)).sum()
        fn = ((1 - pred_bin) * gt_bin).sum()

        # 论文标准Fw计算（1.3权重）
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        fmeasure  = (1.3 * precision * recall) / (0.3 * precision + recall + 1e-8)

        self.prec_sum += fmeasure
        self.count    += 1

    def get(self):
        if self.count == 0:
            return 0.0
        return self.prec_sum / self.count

########################################
#  2) 加载属性标签 TXT (修复文件名匹配)
########################################
def load_attribute_files(attr_file_path):
    """
    修复：
    1. 移除文件名后缀（如.jpg/.png）以匹配推理时的name
    2. 增强容错性，支持不同分隔符
    """
    attributes = ["BSO","CB","CIB","IC","LI","MSO","OF",
                  "SSO","SA","TC","BW","bRGB","bT"]

    attr_dict = {attr: set() for attr in attributes}

    try:
        print(f"正在从 {attr_file_path} 加载属性标签...")
        with open(attr_file_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            # 跳过标题行
            lines = [l.strip() for l in lines if l.strip()]
            if len(lines) == 0:
                print("警告：属性文件为空！")
                return attributes, attr_dict
            
            # 跳过第一行标题（如果包含表头）
            if 'name' in lines[0] or 'BSO' in lines[0]:
                lines = lines[1:]

            for line_idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 14:  # 1个文件名 + 13个属性
                    print(f"警告：第{line_idx+1}行格式错误，跳过: {line}")
                    continue
                
                # 核心修复：移除文件名后缀（如.jpg/.png）
                name = parts[0].split('.')[0]  # 只保留文件名前缀
                # 遍历13个属性
                for i in range(13):
                    attr_name = attributes[i]
                    attr_value = parts[i+1]
                    try:
                        if float(attr_value) > 0:
                            attr_dict[attr_name].add(name)
                    except ValueError:
                        print(f"警告：第{line_idx+1}行{attr_name}值无效: {attr_value}")
                        continue
        print(f"属性标签加载成功，各属性样本数：")
        for attr in attributes:
            print(f"  {attr}: {len(attr_dict[attr])}")
    except FileNotFoundError:
        print(f"错误: 属性标签文件 {attr_file_path} 不存在！")
        exit(1)
    except Exception as e:
        print(f"错误：加载属性标签失败: {str(e)}")
        exit(1)
        
    return attributes, attr_dict

########################################
#  3) 加载模型 - 与训练完全一致
########################################
# ★ 核心修改：使用与训练完全相同的模型配置
model = RealCLIPEnhancedFusionNet(
    use_real_clip=True,
    clip_model_name="ViT-B/32",
    use_cross_guidance=True,
    use_memory_bank=True,
    use_true_segdet=True,
    freeze_pretrained=True,     # 测试时也保持冻结
    optimized_mode=True,
    gradient_checkpointing=False  # ★ 测试时关闭梯度检查点以加速
).cuda()

# 加载权重
ckpt = torch.load(opt.test_model, map_location='cuda', weights_only=True)

new_state_dict = OrderedDict()
model_state_dict = model.state_dict()  # 获取模型默认参数

# 处理DataParallel的module.前缀
for k, v in ckpt.items():
    # 移除module.前缀（如果是多GPU训练的权重）
    if k.startswith("module."):
        name = k[7:]
    else:
        name = k
    
    # ★ 重要：跳过不匹配的参数（如memory_bank等）
    if name in model_state_dict:
        if model_state_dict[name].shape == v.shape:
            new_state_dict[name] = v
        else:
            print(f"[SKIP] 形状不匹配: {name} | 模型形状: {model_state_dict[name].shape}, 权重形状: {v.shape}")
            # 使用模型默认初始化
            new_state_dict[name] = model_state_dict[name]
    else:
        print(f"[SKIP] 未找到参数: {name}")
        continue

# 加载权重
print(f"加载模型权重: {opt.test_model}")
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

print(f"未加载的权重: {len(missing_keys)}个")
if missing_keys:
    print("前10个缺失键:", missing_keys[:10])
    
print(f"意外的权重: {len(unexpected_keys)}个")
if unexpected_keys:
    print("前10个意外键:", unexpected_keys[:10])

model.eval()

# 验证模型输出
with torch.no_grad():
    # ★ 修改：使用256×256，与训练一致
    test_img = torch.randn(1, 3, 256, 256).cuda()
    test_t = torch.randn(1, 3, 256, 256).cuda()
    
    # ★ 修改：模型现在输出字典，不需要传入shape参数
    test_output = model(test_img, test_t)
    
    # 获取最终的输出（out_f）
    out_f = test_output['out_f']
    out_f_norm = out_f.sigmoid()
    
    print(f"模型验证输出均值 (sigmoid后): {out_f_norm.mean().item():.4f}")
    if out_f_norm.mean().item() < 1e-6:
        print("警告：模型输出接近全0，可能影响结果！")
    else:
        print("模型输出正常，非全0！")

########################################
#  4) 数据集测试循环 (增强调试)
########################################
test_data_root = opt.test_data_root
maps_path = opt.maps_path

test_sets = ['VT5000/Test']

for dataset in test_sets:
    save_path = join(maps_path, dataset)
    os.makedirs(save_path, exist_ok=True)

    dataset_path = join(test_data_root, dataset)
    test_loader = test_dataset(dataset_path, opt.testsize)

    # 加载VT5000属性标签
    if "VT5000" in dataset:
        attr_file_path = join(dataset_path, "attribute.txt")
        attributes, attr_files = load_attribute_files(attr_file_path)
        attr_metric = {attr: Metric() for attr in attributes}
    else:
        attr_files = None

    total_time = 0
    frame_count = 0
    zero_output_count = 0  # 统计全0输出数量

    for i in range(test_loader.size):
        image, t, gt, (H, W), name = test_loader.load_data()
        # 核心修复：移除文件名后缀以匹配属性标签
        name_prefix = name.split('.')[0]
        
        image = image.cuda()
        t = t.cuda()

        # 推理
        with torch.no_grad():
            start_time = time.time()
            
            # ★ 修改：模型现在输出字典，不需要传入shape参数
            outputs = model(image, t)
            
            # 获取融合输出 out_f（与训练时评估的相同）
            out_f = outputs['out_f']
            
            # 如果需要其他输出（如可视化）
            # out_rgb = outputs['out_rgb']
            # out_t = outputs['out_t']
            # out_edge = outputs['out_edge']
            # out = outputs['out']  # 对应out_main
            
            end_time = time.time()

        total_time += (end_time - start_time)
        frame_count += 1

        # 处理输出
        res = out_f.sigmoid().data.cpu().numpy().squeeze()
        
        # 调整尺寸到原始GT大小
        if res.shape != gt.shape:
            res = cv2.resize(res, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # 处理输出（验证是否全0）
        if np.max(res) < 1e-6:
            zero_output_count += 1
            print(f"警告：{name}推理输出全0！")
        
        # 归一化（防止除以0）
        res_min = res.min()
        res_max = res.max()
        if res_max - res_min < 1e-8:
            res_norm = np.zeros_like(res)
        else:
            res_norm = (res - res_min) / (res_max - res_min)
        # 确保归一化后在0-1之间
        res_norm = np.clip(res_norm, 0, 1)

        # 保存预测图
        cv2.imwrite(join(save_path, name), (res_norm * 255).astype(np.uint8))

        # 统计属性指标
        if attr_files is not None:
            # 处理GT
            if isinstance(gt, torch.Tensor):
                gt_np = gt.squeeze().cpu().numpy()
            else:
                gt_np = np.array(gt)
            
            # 确保GT是二维的
            if len(gt_np.shape) > 2:
                gt_np = gt_np.squeeze()
            
            # 遍历所有属性
            for attr in attributes:
                if name_prefix in attr_files[attr]:
                    attr_metric[attr].update(res_norm, gt_np)

    # 输出FPS
    avg_time = total_time / frame_count if frame_count > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"\n[{dataset}] FPS: {fps:.2f}, 全0输出数: {zero_output_count}/{frame_count}")

    # 输出属性指标
    if attr_files is not None:
        print("\n=== Attribute Results (F-measure Fw) ===")
        total_fw = 0
        attr_count = 0
        for attr in attributes:
            fw = attr_metric[attr].get()
            total_fw += fw
            attr_count += 1 if fw > 0 else 0
            print(f"{attr}: {fw:.3f}")
        
        if attr_count > 0:
            print(f"平均Fw: {total_fw / attr_count:.3f}")
        print("========================================\n")

print('Test Done!')
'''




'''
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from swinv2_net import RealCLIPEnhancedFusionNet  # ★ 与训练保持一致
from data import test_dataset
from options import opt
from collections import OrderedDict
import time
from os.path import join

# 忽略meshgrid警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

########################################
#  1) 定义 F-measure 计算器
########################################
class Metric:
    def __init__(self):
        self.prec_sum = 0
        self.recall_sum = 0
        self.count = 0

    def update(self, pred, gt):
        # 确保输入是numpy数组
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        # 处理GT：可能是numpy数组、PIL Image或torch Tensor
        if isinstance(gt, torch.Tensor):
            gt_np = gt.cpu().numpy()
        elif hasattr(gt, 'shape'):  # numpy数组
            gt_np = gt
        elif hasattr(gt, 'size'):  # PIL Image
            # PIL Image转换为numpy数组
            gt_np = np.array(gt)
        else:
            gt_np = np.array(gt)
        
        # 确保是二维数组
        if len(gt_np.shape) > 2:
            gt_np = gt_np.squeeze()
        
        # 核心修复1：处理宽高颠倒问题（统一为H×W）
        if pred.shape != gt_np.shape:
            # 尝试转置GT匹配预测图
            if gt_np.T.shape == pred.shape:
                gt_np = gt_np.T
            else:
                # 插值到相同尺寸
                gt_np = cv2.resize(gt_np, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 核心修复2：预测值先归一化到0-1（解决负数问题）
        pred = np.clip(pred, 0, 1)
        
        # 二值化（严格按论文Fw公式）
        pred_bin = (pred > 0.5).astype(np.uint8)
        gt_bin   = (gt_np > 128).astype(np.uint8)

        # 关键：先检查GT是否为空（避免除以0）
        if np.sum(gt_bin) == 0:
            return  # 无标注的样本不参与计算

        tp = (pred_bin * gt_bin).sum()
        fp = (pred_bin * (1 - gt_bin)).sum()
        fn = ((1 - pred_bin) * gt_bin).sum()

        # 论文标准Fw计算（1.3权重）
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        fmeasure  = (1.3 * precision * recall) / (0.3 * precision + recall + 1e-8)

        self.prec_sum += fmeasure
        self.count    += 1

    def get(self):
        if self.count == 0:
            return 0.0
        return self.prec_sum / self.count

########################################
#  2) 加载属性标签 TXT
########################################
def load_attribute_files(attr_file_path):
    attributes = ["BSO","CB","CIB","IC","LI","MSO","OF",
                  "SSO","SA","TC","BW","bRGB","bT"]

    attr_dict = {attr: set() for attr in attributes}

    try:
        print(f"正在从 {attr_file_path} 加载属性标签...")
        with open(attr_file_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines if l.strip()]
            if len(lines) == 0:
                print("警告：属性文件为空！")
                return attributes, attr_dict
            
            if 'name' in lines[0] or 'BSO' in lines[0]:
                lines = lines[1:]

            for line_idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 14:
                    print(f"警告：第{line_idx+1}行格式错误，跳过: {line}")
                    continue
                
                name = parts[0].split('.')[0]
                for i in range(13):
                    attr_name = attributes[i]
                    attr_value = parts[i+1]
                    try:
                        if float(attr_value) > 0:
                            attr_dict[attr_name].add(name)
                    except ValueError:
                        print(f"警告：第{line_idx+1}行{attr_name}值无效: {attr_value}")
                        continue
        print(f"属性标签加载成功，各属性样本数：")
        for attr in attributes:
            print(f"  {attr}: {len(attr_dict[attr])}")
    except Exception as e:
        print(f"错误：加载属性标签失败: {str(e)}")
        exit(1)
        
    return attributes, attr_dict

########################################
#  3) 加载模型 - 与训练完全一致
########################################
# ★ 核心修改：使用与训练完全相同的模型配置
model = RealCLIPEnhancedFusionNet(
    use_real_clip=True,
    clip_model_name="ViT-B/32",
    use_cross_guidance=True,
    use_memory_bank=True,
    use_true_segdet=True,
    freeze_pretrained=True,
    optimized_mode=True,
    gradient_checkpointing=False
).cuda()

# 加载权重
ckpt = torch.load(opt.test_model, map_location='cuda', weights_only=True)

new_state_dict = OrderedDict()
model_state_dict = model.state_dict()

for k, v in ckpt.items():
    if k.startswith("module."):
        name = k[7:]
    else:
        name = k
    
    if name in model_state_dict:
        if model_state_dict[name].shape == v.shape:
            new_state_dict[name] = v
        else:
            print(f"[SKIP] 形状不匹配: {name} | 模型形状: {model_state_dict[name].shape}, 权重形状: {v.shape}")
            new_state_dict[name] = model_state_dict[name]
    else:
        print(f"[SKIP] 未找到参数: {name}")
        continue

print(f"加载模型权重: {opt.test_model}")
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

print(f"未加载的权重: {len(missing_keys)}个")
if missing_keys:
    print("前10个缺失键:", missing_keys[:10])
    
print(f"意外的权重: {len(unexpected_keys)}个")
if unexpected_keys:
    print("前10个意外键:", unexpected_keys[:10])

model.eval()

# ★ 验证模型输出 - 使用256×256，与训练一致
with torch.no_grad():
    # ★ 修改这里：使用256×256而不是384×384
    test_img = torch.randn(1, 3, 256, 256).cuda()
    test_t = torch.randn(1, 3, 256, 256).cuda()
    
    test_output = model(test_img, test_t)
    
    out_f = test_output['out_f']
    out_f_norm = out_f.sigmoid()
    
    print(f"模型验证输出均值 (sigmoid后): {out_f_norm.mean().item():.4f}")
    if out_f_norm.mean().item() < 1e-6:
        print("警告：模型输出接近全0，可能影响结果！")
    else:
        print("模型输出正常，非全0！")

########################################
#  4) 数据集测试循环 - 修复PIL Image处理
########################################
test_data_root = opt.test_data_root
maps_path = opt.maps_path

test_sets = ['VT5000/Test']

for dataset in test_sets:
    save_path = join(maps_path, dataset)
    os.makedirs(save_path, exist_ok=True)

    dataset_path = join(test_data_root, dataset)
    # ★ 确保使用正确的testsize（256）
    test_loader = test_dataset(dataset_path, opt.testsize)  # testsize=256

    if "VT5000" in dataset:
        attr_file_path = join(dataset_path, "attribute.txt")
        attributes, attr_files = load_attribute_files(attr_file_path)
        attr_metric = {attr: Metric() for attr in attributes}
    else:
        attr_files = None

    total_time = 0
    frame_count = 0
    zero_output_count = 0

    for i in range(test_loader.size):
        image, t, gt, (H, W), name = test_loader.load_data()
        name_prefix = name.split('.')[0]
        
        image = image.cuda()
        t = t.cuda()

        with torch.no_grad():
            start_time = time.time()
            outputs = model(image, t)
            out_f = outputs['out_f']
            end_time = time.time()

        total_time += (end_time - start_time)
        frame_count += 1

        res = out_f.sigmoid().data.cpu().numpy().squeeze()
        
        # ★ 修复：处理GT可能是PIL Image的情况
        # 将GT转换为numpy数组
        if hasattr(gt, 'size'):  # PIL Image
            gt_np = np.array(gt)
        elif isinstance(gt, torch.Tensor):
            gt_np = gt.squeeze().cpu().numpy()
        else:
            gt_np = np.array(gt)
        
        # 确保GT是二维的
        if len(gt_np.shape) > 2:
            gt_np = gt_np.squeeze()
        
        # 调整预测图尺寸到原始GT大小
        if res.shape != gt_np.shape:
            res = cv2.resize(res, (W, H), interpolation=cv2.INTER_LINEAR)
        
        if np.max(res) < 1e-6:
            zero_output_count += 1
            print(f"警告：{name}推理输出全0！")
        
        res_min = res.min()
        res_max = res.max()
        if res_max - res_min < 1e-8:
            res_norm = np.zeros_like(res)
        else:
            res_norm = (res - res_min) / (res_max - res_min)
        res_norm = np.clip(res_norm, 0, 1)

        cv2.imwrite(join(save_path, name), (res_norm * 255).astype(np.uint8))

        if attr_files is not None:
            # GT已经是numpy数组gt_np
            for attr in attributes:
                if name_prefix in attr_files[attr]:
                    attr_metric[attr].update(res_norm, gt_np)

    avg_time = total_time / frame_count if frame_count > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"\n[{dataset}] FPS: {fps:.2f}, 全0输出数: {zero_output_count}/{frame_count}")

    if attr_files is not None:
        print("\n=== Attribute Results (F-measure Fw) ===")
        total_fw = 0
        attr_count = 0
        for attr in attributes:
            fw = attr_metric[attr].get()
            total_fw += fw
            attr_count += 1 if fw > 0 else 0
            print(f"{attr}: {fw:.3f}")
        
        if attr_count > 0:
            print(f"平均Fw: {total_fw / attr_count:.3f}")
        print("========================================\n")

print('Test Done!')
'''
