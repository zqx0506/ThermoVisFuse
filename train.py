'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append('./models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from swinv2_net import DFNet
from data import get_loader, test_dataset
from utils import clip_gradient
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
from loss.ssim import SSIM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True

# build the model
model = DFNet()
if (opt.load is not None):
    model.load_pre(opt.load)
    print('load model from ', opt.load)
gpus = [0, 1]
model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
print(next(model.parameters()).device)

base, body = [], []
for name, param in model.named_parameters():
    if 'swin_image' in name or 'swin_thermal' in name:
        base.append(param)
    else:
        body.append(param)
optimizer = torch.optim.SGD([{'params': base}, {'params': body}], lr=opt.lr, momentum=opt.momentum,
                            weight_decay=opt.decay_rate, nesterov=True)

# set the path
train_root = opt.train_data_root
test_root = opt.val_data_root

save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)


print('load data...')
num_gpus = torch.cuda.device_count()
print(f"========>num_gpus:{num_gpus}==========")
train_loader = get_loader(train_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("SwinMCNet-Train")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{}'.format(opt.epoch, opt.lr,
                                                                                                 opt.batchsize,
                                                                                                 opt.trainsize,
                                                                                                  opt.clip,
                                                                                                 opt.decay_rate,
                                                                                                 opt.load, save_path))


# loss
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

def dice_loss(inputs, targets):
    smooth = 1
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice



ssim_fn = SSIM(window_size=11, size_average=True)
def ssim_loss(inputs, targets):
    probs = torch.sigmoid(inputs)
    return 1 - ssim_fn(probs,targets)

step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 1


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, ts, gts, bodys, details) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            image, t, gt, body, detail = images.cuda(), ts.cuda(), gts.cuda(), bodys.cuda(), details.cuda()

            out_rgb, out_t,out_f, out_edge, out = model(image, t)
            loss_rgb = F.binary_cross_entropy_with_logits(out_rgb, gt) + ssim_loss(out_rgb, gt) + iou_loss(out_rgb, gt)
            loss_t = F.binary_cross_entropy_with_logits(out_t, gt) + ssim_loss(out_t, gt) + iou_loss(out_t, gt)
            loss_edge = dice_loss(out_edge, detail)
            loss1 = F.binary_cross_entropy_with_logits(out, gt) + iou_loss(out, gt) + ssim_loss(out, gt)
            loss2 = F.binary_cross_entropy_with_logits(out_f, gt) + iou_loss(out_f, gt) + ssim_loss(out_f, gt)
            loss = (loss_rgb + loss_t + loss_edge+ loss1  + loss2) / 5

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 20 == 0 or i == total_step or i == 1:
                print(
                    '%s | epoch:%d/%d | step:%d/%d | lr=%.6f | loss=%.6f'
                    % (datetime.now(), epoch, opt.epoch, i, total_step, optimizer.param_groups[0]['lr'], loss.item()
                       ))

                logging.info(
                    '##TRAIN##:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], lr_bk: {:.6f}, Loss: {:.4f}'.
                    format(epoch, opt.epoch, i, total_step, optimizer.param_groups[0]['lr'], loss.data))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)

                res = out[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('out', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('##TRAIN##:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 50 == 0:
            torch.save(model.state_dict(), save_path + 'SwinMCNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'SwinMCNet_epoch_{}.pth'.format(epoch))
        print('save checkpoints successfully!')
        raise


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, t, gt, (H, W), name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            t = t.cuda()
            out_rgb, out_t,out_f, out_edge, out= model(image, t)
            res = out_f
            res = F.interpolate(res, size=gt.shape, mode='bilinear')
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('\n')
        print('##TEST##:Epoch: {}   MAE: {}'.format(epoch, mae))

        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'SwinMCNet_epoch_best.pth')
        print('##SAVE##:bestEpoch: {}   bestMAE: {}'.format(best_epoch, best_mae))
        print('\n')
        logging.info('##TEST##:Epoch:{}   MAE:{}   bestEpoch:{}   bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch + 1):
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch) / (opt.epoch) * 2 - 1)) * opt.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch) / (opt.epoch) * 2 - 1)) * opt.lr
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
'''



'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn

sys.path.append('./models')
from swinv2_net import DFNet
from data import get_loader, test_dataset
from utils import clip_gradient
from options import opt
from loss.ssim import SSIM

# ================== 设备设置 ==================
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus}")
cudnn.benchmark = True

# ================== 构建模型 ==================
model = DFNet().to(device)

# 加载预训练权重（如果有）
if opt.load is not None:
    model.load_pre(opt.load)
    print('load model from ', opt.load)
else:
    model.load_pre(None)  # 从零训练

# 多 GPU 支持
if use_cuda and num_gpus > 1:
    model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

print(f"Model parameters device: {next(model.parameters()).device}")

# ================== 优化器 ==================
base, body = [], []
for name, param in model.named_parameters():
    if 'swin_image' in name or 'swin_thermal' in name:
        base.append(param)
    else:
        body.append(param)

optimizer = torch.optim.SGD(
    [{'params': base}, {'params': body}],
    lr=opt.lr,
    momentum=opt.momentum,
    weight_decay=opt.decay_rate,
    nesterov=True
)

# ================== 数据路径 ==================
train_root = opt.train_data_root
test_root = opt.val_data_root
save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

print('load data...')
train_loader = get_loader(train_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_root, opt.trainsize)
total_step = len(train_loader)
print(f"Total steps per epoch: {total_step}")

# ================== 日志 ==================
logging.basicConfig(
    filename=os.path.join(save_path, 'log.log'),
    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
    level=logging.INFO,
    filemode='a',
    datefmt='%Y-%m-%d %I:%M:%S %p'
)
logging.info("SwinMCNet-Train")
logging.info(
    f'epoch:{opt.epoch}; lr:{opt.lr}; batchsize:{opt.batchsize}; trainsize:{opt.trainsize}; '
    f'clip:{opt.clip}; decay_rate:{opt.decay_rate}; load:{opt.load}; save_path:{save_path}'
)

# ================== 损失函数 ==================
ssim_fn = SSIM(window_size=11, size_average=True)

def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

def dice_loss(inputs, targets):
    smooth = 1
    inputs = torch.sigmoid(inputs).view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice

def ssim_loss(inputs, targets):
    probs = torch.sigmoid(inputs)
    return 1 - ssim_fn(probs, targets)

# ================== 训练状态 ==================
step = 0
writer = SummaryWriter(os.path.join(save_path, 'summary'))
best_mae = 1
best_epoch = 1

# ================== 训练函数 ==================
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, ts, gts, bodys, details) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            image, t, gt, body, detail = images.to(device), ts.to(device), gts.to(device), bodys.to(device), details.to(device)

            out_rgb, out_t, out_f, out_edge, out = model(image, t)

            loss_rgb = F.binary_cross_entropy_with_logits(out_rgb, gt) + ssim_loss(out_rgb, gt) + iou_loss(out_rgb, gt)
            loss_t = F.binary_cross_entropy_with_logits(out_t, gt) + ssim_loss(out_t, gt) + iou_loss(out_t, gt)
            loss_edge = dice_loss(out_edge, detail)
            loss1 = F.binary_cross_entropy_with_logits(out, gt) + iou_loss(out, gt) + ssim_loss(out, gt)
            loss2 = F.binary_cross_entropy_with_logits(out_f, gt) + iou_loss(out_f, gt) + ssim_loss(out_f, gt)
            loss = (loss_rgb + loss_t + loss_edge + loss1 + loss2) / 5

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.item()

            if i % 20 == 0 or i == total_step or i == 1:
                print(f"{datetime.now()} | epoch:{epoch}/{opt.epoch} | step:{i}/{total_step} | "
                      f"lr={optimizer.param_groups[0]['lr']:.6f} | loss={loss.item():.6f}")
                logging.info(
                    f"##TRAIN##:Epoch [{epoch}/{opt.epoch}], Step [{i}/{total_step}], "
                    f"lr_bk:{optimizer.param_groups[0]['lr']:.6f}, Loss:{loss.item():.4f}"
                )

                writer.add_scalar('Loss', loss.item(), global_step=step)

                # 可视化 RGB / GT / Out
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)

                res = out[0].clone().sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('out', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info(f"##TRAIN##:Epoch [{epoch}/{opt.epoch}], Loss_AVG: {loss_all:.4f}")
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'SwinMCNet_epoch_{epoch}.pth'))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, f'SwinMCNet_epoch_{epoch}.pth'))
        print('save checkpoints successfully!')
        raise

# ================== 测试函数 ==================
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, t, gt, (H, W), name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.to(device)
            t = t.to(device)

            out_rgb, out_t, out_f, out_edge, out = model(image, t)
            res = out_f
            res = F.interpolate(res, size=gt.shape, mode='bilinear')
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print(f'##TEST##:Epoch: {epoch}   MAE: {mae}')

        if epoch == 1:
            best_mae = mae
        elif mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'SwinMCNet_epoch_best.pth'))

        print(f'##SAVE##:bestEpoch: {best_epoch}   bestMAE: {best_mae}')
        logging.info(f'##TEST##:Epoch:{epoch}   MAE:{mae}   bestEpoch:{best_epoch}   bestMAE:{best_mae}')

# ================== 主训练循环 ==================
if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch + 1):
        # Cosine-like LR schedule
        optimizer.param_groups[0]['lr'] = (1 - abs(epoch / opt.epoch * 2 - 1)) * opt.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs(epoch / opt.epoch * 2 - 1)) * opt.lr
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)

        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
'''






import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn

sys.path.append('./models')
from swinv2_net1 import DFNet
from data import get_loader, test_dataset
from utils import clip_gradient
from options import opt
from loss.ssim import SSIM

# ================== 设备设置 ==================
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus}")
cudnn.benchmark = True

# ================== 构建模型 ==================
model = DFNet().to(device)

# 加载预训练权重（如果有）
if opt.load is not None:
    model.load_pre(opt.load)
    print('load model from ', opt.load)
else:
    model.load_pre(None)  # 从零训练

# 多 GPU 支持
if use_cuda and num_gpus > 1:
    model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

print(f"Model parameters device: {next(model.parameters()).device}")

# ================== 优化器 ==================
base, body_params = [], []
for name, param in model.named_parameters():
    if 'swin_image' in name or 'swin_thermal' in name:
        base.append(param)
    else:
        body_params.append(param)

optimizer = torch.optim.SGD(
    [{'params': base}, {'params': body_params}],
    lr=opt.lr,
    momentum=opt.momentum,
    weight_decay=opt.decay_rate,
    nesterov=True
)

# ================== 数据路径 ==================
train_root = opt.train_data_root
test_root = opt.val_data_root
save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

print('load data...')
train_loader = get_loader(train_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_root, opt.trainsize)
total_step = len(train_loader)
print(f"Total steps per epoch: {total_step}")

# ================== 日志 ==================
logging.basicConfig(
    filename=os.path.join(save_path, 'log.log'),
    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
    level=logging.INFO,
    filemode='a',
    datefmt='%Y-%m-%d %I:%M:%S %p'
)
logging.info("SwinMCNet-Train")
logging.info(
    f'epoch:{opt.epoch}; lr:{opt.lr}; batchsize:{opt.batchsize}; trainsize:{opt.trainsize}; '
    f'clip:{opt.clip}; decay_rate:{opt.decay_rate}; load:{opt.load}; save_path:{save_path}'
)

# ================== 损失函数 ==================
ssim_fn = SSIM(window_size=11, size_average=True)

def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

def dice_loss(inputs, targets):
    smooth = 1
    inputs = torch.sigmoid(inputs).view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice

def ssim_loss(inputs, targets):
    probs = torch.sigmoid(inputs)
    return 1 - ssim_fn(probs, targets)

# ================== 训练状态 ==================
step = 0
writer = SummaryWriter(os.path.join(save_path, 'summary'))
best_mae = 1
best_epoch = 1

# ================== 训练函数 ==================
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, ts, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            image, t, gt = images.to(device), ts.to(device), gts.to(device)

            out_rgb, out_t, out_f, out_edge, out = model(image, t)

            loss_rgb = F.binary_cross_entropy_with_logits(out_rgb, gt) + ssim_loss(out_rgb, gt) + iou_loss(out_rgb, gt)
            loss_t   = F.binary_cross_entropy_with_logits(out_t, gt) + ssim_loss(out_t, gt) + iou_loss(out_t, gt)
            loss1    = F.binary_cross_entropy_with_logits(out, gt) + iou_loss(out, gt) + ssim_loss(out, gt)
            loss2    = F.binary_cross_entropy_with_logits(out_f, gt) + iou_loss(out_f, gt) + ssim_loss(out_f, gt)

            loss = (loss_rgb + loss_t + loss1 + loss2) / 4

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.item()

            if i % 20 == 0 or i == total_step or i == 1:
                print(f"{datetime.now()} | epoch:{epoch}/{opt.epoch} | step:{i}/{total_step} | "
                      f"lr={optimizer.param_groups[0]['lr']:.6f} | loss={loss.item():.6f}")
                logging.info(
                    f"##TRAIN##:Epoch [{epoch}/{opt.epoch}], Step [{i}/{total_step}], "
                    f"lr_bk:{optimizer.param_groups[0]['lr']:.6f}, Loss:{loss.item():.4f}"
                )

                writer.add_scalar('Loss', loss.item(), global_step=step)

                # 可视化 RGB / GT / Out
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)

                res = out[0].clone().sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('out', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info(f"##TRAIN##:Epoch [{epoch}/{opt.epoch}], Loss_AVG: {loss_all:.4f}")
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'SwinMCNet_epoch_{epoch}.pth'))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, f'SwinMCNet_epoch_{epoch}.pth'))
        print('save checkpoints successfully!')
        raise

# ================== 测试函数 ==================
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, t, gt, (H, W), name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.to(device)
            t = t.to(device)

            out_rgb, out_t, out_f, out_edge, out = model(image, t)
            res = out_f
            res = F.interpolate(res, size=gt.shape, mode='bilinear')
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print(f'##TEST##:Epoch: {epoch}   MAE: {mae}')

        if epoch == 1:
            best_mae = mae
        elif mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'SwinMCNet_epoch_best.pth'))

        print(f'##SAVE##:bestEpoch: {best_epoch}   bestMAE: {best_mae}')
        logging.info(f'##TEST##:Epoch:{epoch}   MAE:{mae}   bestEpoch:{best_epoch}   bestMAE:{best_mae}')

# ================== 主训练循环 ==================
if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch + 1):
        # Cosine-like LR schedule
        optimizer.param_groups[0]['lr'] = (1 - abs(epoch / opt.epoch * 2 - 1)) * opt.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs(epoch / opt.epoch * 2 - 1)) * opt.lr
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)

        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)



'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn

sys.path.append('./models')
from swinv2_net128 import DFNet
from data import get_loader, test_dataset
from utils import clip_gradient
from options import opt
from loss.ssim import SSIM

# ================== 设备设置 ==================
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus}")
cudnn.benchmark = True

# ================== 构建模型 ==================
model = DFNet().to(device)

# 加载预训练权重（如果有）
if opt.load is not None:
    model.load_pre(opt.load)
    print('load model from ', opt.load)
else:
    print("Training from scratch...")
    # 不需要调用 model.load_pre(None)，因为模型已经初始化

# 多 GPU 支持
if use_cuda and num_gpus > 1:
    model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

print(f"Model parameters device: {next(model.parameters()).device}")

# ================== 优化器 ==================
base, body_params = [], []
for name, param in model.named_parameters():
    if 'swin_image' in name or 'swin_thermal' in name:
        base.append(param)
    else:
        body_params.append(param)

optimizer = torch.optim.SGD(
    [{'params': base}, {'params': body_params}],
    lr=opt.lr,
    momentum=opt.momentum,
    weight_decay=opt.decay_rate,
    nesterov=True
)

# ================== 数据路径 ==================
train_root = opt.train_data_root
test_root = opt.val_data_root
save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

print('load data...')
train_loader = get_loader(train_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_root, opt.trainsize)
total_step = len(train_loader)
print(f"Total steps per epoch: {total_step}")

# ================== 日志 ==================
logging.basicConfig(
    filename=os.path.join(save_path, 'log.log'),
    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
    level=logging.INFO,
    filemode='a',
    datefmt='%Y-%m-%d %I:%M:%S %p'
)
logging.info("SwinMCNet-Train")
logging.info(
    f'epoch:{opt.epoch}; lr:{opt.lr}; batchsize:{opt.batchsize}; trainsize:{opt.trainsize}; '
    f'clip:{opt.clip}; decay_rate:{opt.decay_rate}; load:{opt.load}; save_path:{save_path}'
)

# ================== 损失函数 ==================
ssim_fn = SSIM(window_size=11, size_average=True)

def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

def dice_loss(inputs, targets):
    smooth = 1
    inputs = torch.sigmoid(inputs).view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice

def ssim_loss(inputs, targets):
    probs = torch.sigmoid(inputs)
    return 1 - ssim_fn(probs, targets)

# ================== 训练状态 ==================
step = 0
writer = SummaryWriter(os.path.join(save_path, 'summary'))
best_mae = 1
best_epoch = 1

# ================== 训练函数 ==================
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, ts, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            image, t, gt = images.to(device), ts.to(device), gts.to(device)

            out_rgb, out_t, out_f, out_edge, out = model(image, t)

            loss_rgb = F.binary_cross_entropy_with_logits(out_rgb, gt) + ssim_loss(out_rgb, gt) + iou_loss(out_rgb, gt)
            loss_t   = F.binary_cross_entropy_with_logits(out_t, gt) + ssim_loss(out_t, gt) + iou_loss(out_t, gt)
            loss1    = F.binary_cross_entropy_with_logits(out, gt) + iou_loss(out, gt) + ssim_loss(out, gt)
            loss2    = F.binary_cross_entropy_with_logits(out_f, gt) + iou_loss(out_f, gt) + ssim_loss(out_f, gt)

            loss = (loss_rgb + loss_t + loss1 + loss2) / 4

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.item()

            if i % 20 == 0 or i == total_step or i == 1:
                print(f"{datetime.now()} | epoch:{epoch}/{opt.epoch} | step:{i}/{total_step} | "
                      f"lr={optimizer.param_groups[0]['lr']:.6f} | loss={loss.item():.6f}")
                logging.info(
                    f"##TRAIN##:Epoch [{epoch}/{opt.epoch}], Step [{i}/{total_step}], "
                    f"lr_bk:{optimizer.param_groups[0]['lr']:.6f}, Loss:{loss.item():.4f}"
                )

                writer.add_scalar('Loss', loss.item(), global_step=step)

                # 可视化 RGB / GT / Out
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)

                res = out[0].clone().sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('out', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info(f"##TRAIN##:Epoch [{epoch}/{opt.epoch}], Loss_AVG: {loss_all:.4f}")
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        if epoch % 4 == 0:
            # 处理 DataParallel 的情况
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(save_path, f'SwinMCNet_epoch_{epoch}.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(save_path, f'SwinMCNet_epoch_{epoch}.pth'))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 处理 DataParallel 的情况
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(save_path, f'SwinMCNet_epoch_{epoch}_interrupt.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f'SwinMCNet_epoch_{epoch}_interrupt.pth'))
        print('save checkpoints successfully!')
        raise

# ================== 测试函数 ==================
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, t, gt, (H, W), name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.to(device)
            t = t.to(device)

            out_rgb, out_t, out_f, out_edge, out = model(image, t)
            res = out_f
            res = F.interpolate(res, size=gt.shape, mode='bilinear')
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print(f'##TEST##:Epoch: {epoch}   MAE: {mae}')

        if epoch == 1:
            best_mae = mae
        elif mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            # 处理 DataParallel 的情况
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(save_path, 'SwinMCNet_epoch_best.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(save_path, 'SwinMCNet_epoch_best.pth'))

        print(f'##SAVE##:bestEpoch: {best_epoch}   bestMAE: {best_mae}')
        logging.info(f'##TEST##:Epoch:{epoch}   MAE:{mae}   bestEpoch:{best_epoch}   bestMAE:{best_mae}')

# ================== 主训练循环 ==================
if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch + 1):
        # Cosine-like LR schedule
        optimizer.param_groups[0]['lr'] = (1 - abs(epoch / opt.epoch * 2 - 1)) * opt.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs(epoch / opt.epoch * 2 - 1)) * opt.lr
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)

        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
'''














































































































































'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append('./models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from swinv2_net128 import DFNet
from data import get_loader, test_dataset
from utils import clip_gradient
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
from loss.ssim import SSIM

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True

# 检查GPU状态
num_gpus = torch.cuda.device_count()
print(f"========> Available GPUs: {num_gpus} ==========")
if num_gpus > 0:
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU")

# build the model
model = DFNet()
if (opt.load is not None):
    model.load_pre(opt.load)
    print('load model from ', opt.load)

# 单GPU设置 - 直接移动到GPU
model = model.cuda()
print("使用单GPU进行训练")
print(f"Model device: {next(model.parameters()).device}")

base, body = [], []
for name, param in model.named_parameters():
    if 'swin_image' in name or 'swin_thermal' in name:
        base.append(param)
    else:
        body.append(param)
optimizer = torch.optim.SGD([{'params': base}, {'params': body}], lr=opt.lr, momentum=opt.momentum,
                            weight_decay=opt.decay_rate, nesterov=True)

# set the path
train_root = opt.train_data_root
test_root = opt.val_data_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

print('load data...')
train_loader = get_loader(train_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("SwinMCNet-Train")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{}'.format(opt.epoch, opt.lr,
                                                                                                 opt.batchsize,
                                                                                                 opt.trainsize,
                                                                                                  opt.clip,
                                                                                                 opt.decay_rate,
                                                                                                 opt.load, save_path))


# loss
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

def dice_loss(inputs, targets):
    smooth = 1
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice

ssim_fn = SSIM(window_size=11, size_average=True)
def ssim_loss(inputs, targets):
    probs = torch.sigmoid(inputs)
    return 1 - ssim_fn(probs,targets)

step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 1


# train function - 修复数据解包
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, ts, gts) in enumerate(train_loader, start=1):  # 只解包3个值
            optimizer.zero_grad()

            image, t, gt = images.cuda(), ts.cuda(), gts.cuda()

            out_rgb, out_t, out_f, out_edge, out = model(image, t)
            
            # 简化损失计算，移除需要body和detail的损失项
            loss_rgb = F.binary_cross_entropy_with_logits(out_rgb, gt) + ssim_loss(out_rgb, gt) + iou_loss(out_rgb, gt)
            loss_t = F.binary_cross_entropy_with_logits(out_t, gt) + ssim_loss(out_t, gt) + iou_loss(out_t, gt)
            loss1 = F.binary_cross_entropy_with_logits(out, gt) + iou_loss(out, gt) + ssim_loss(out, gt)
            loss2 = F.binary_cross_entropy_with_logits(out_f, gt) + iou_loss(out_f, gt) + ssim_loss(out_f, gt)
            
            # 移除loss_edge，因为缺少detail数据
            loss = (loss_rgb + loss_t + loss1 + loss2) / 4

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 20 == 0 or i == total_step or i == 1:
                print(
                    '%s | epoch:%d/%d | step:%d/%d | lr=%.6f | loss=%.6f'
                    % (datetime.now(), epoch, opt.epoch, i, total_step, optimizer.param_groups[0]['lr'], loss.item()
                       ))

                logging.info(
                    '##TRAIN##:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], lr_bk: {:.6f}, Loss: {:.4f}'.
                    format(epoch, opt.epoch, i, total_step, optimizer.param_groups[0]['lr'], loss.data))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)

                res = out[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('out', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('##TRAIN##:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 50 == 0:
            torch.save(model.state_dict(), save_path + 'SwinMCNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'SwinMCNet_epoch_{}_interrupt.pth'.format(epoch))
        print('save checkpoints successfully!')
        raise


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, t, gt, (H, W), name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            t = t.cuda()
            out_rgb, out_t, out_f, out_edge, out = model(image, t)
            res = out_f
            res = F.interpolate(res, size=gt.shape, mode='bilinear')
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('\n')
        print('##TEST##:Epoch: {}   MAE: {}'.format(epoch, mae))

        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'SwinMCNet_epoch_best.pth')
        print('##SAVE##:bestEpoch: {}   bestMAE: {}'.format(best_epoch, best_mae))
        print('\n')
        logging.info('##TEST##:Epoch:{}   MAE:{}   bestEpoch:{}   bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch + 1):
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch) / (opt.epoch) * 2 - 1)) * opt.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch) / (opt.epoch) * 2 - 1)) * opt.lr
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
'''



'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append('./models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid

# ★ 必改：从模型文件导入模型类（不能再用 DFNet = 实例）
from swinv2_net128 import RealCLIPEnhancedFusionNet  

from data import get_loader, test_dataset
from utils import clip_gradient
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
from loss.ssim import SSIM

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True

# 检查GPU状态
num_gpus = torch.cuda.device_count()
print(f"========> Available GPUs: {num_gpus} ==========")
if num_gpus > 0:
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda:0')
else:
    print("No GPU available, using CPU")
    device = torch.device('cpu')

# ★★ 必改：正确实例化模型（完全取代 DFNet()）
model = RealCLIPEnhancedFusionNet(
    use_real_clip=True,
    clip_model_name="ViT-B/32",
    use_cross_guidance=True,
    use_memory_bank=True,
    use_true_segdet=True,
    freeze_pretrained=True,     # ← 冻结 CLIP 的参数
    optimized_mode=True,
    gradient_checkpointing=True
).to(device)

# 多GPU支持（如果有多个GPU）
if num_gpus > 1:
    model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

print(f"Model device: {next(model.parameters()).device}")

# ★ 如果你需要解冻部分模块
# model.unfreeze_for_finetuning(["fusion_module"])

# 加载预训练权重（可选）
if opt.load is not None:
    model.load_pre(opt.load)
    print('load model from ', opt.load)
else:
    print("Training from scratch...")

# 优化器
base, body = [], []
for name, param in model.named_parameters():
    if 'swin_image' in name or 'swin_thermal' in name:
        base.append(param)
    else:
        body.append(param)

optimizer = torch.optim.SGD(
    [{'params': base}, {'params': body}],
    lr=opt.lr,
    momentum=opt.momentum,
    weight_decay=opt.decay_rate,
    nesterov=True
)

# ================== 路径 ====================
train_root = opt.train_data_root
test_root = opt.val_data_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

print('load data...')
train_loader = get_loader(train_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_root, opt.trainsize)
total_step = len(train_loader)
print(f"Total steps per epoch: {total_step}")

# =============  日志  ==================
logging.basicConfig(
    filename=os.path.join(save_path, 'log.log'),
    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
    level=logging.INFO,
    filemode='a',
    datefmt='%Y-%m-%d %I:%M:%S %p'
)
logging.info("SwinMCNet-Train")

logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
        opt.decay_rate, opt.load, save_path
    )
)

# ============= Loss ===============

def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

def dice_loss(inputs, targets):
    smooth = 1
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice

ssim_fn = SSIM(window_size=11, size_average=True).to(device)
def ssim_loss(inputs, targets):
    probs = torch.sigmoid(inputs)
    return 1 - ssim_fn(probs, targets)

step = 0
writer = SummaryWriter(os.path.join(save_path, 'summary'))
best_mae = 1
best_epoch = 1

# ============= Train =============
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, ts, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            image, t, gt = images.to(device), ts.to(device), gts.to(device)
            model_output = model(image, t)
            out_rgb = model_output['out_rgb']
            out_t = model_output['out_t']
            out_f = model_output['out_f']
            out_edge = model_output['out_edge']
            out = model_output['out']  # 对应out_main

            loss_rgb = (F.binary_cross_entropy_with_logits(out_rgb, gt)
                        + ssim_loss(out_rgb, gt) + iou_loss(out_rgb, gt))
            loss_t = (F.binary_cross_entropy_with_logits(out_t, gt)
                      + ssim_loss(out_t, gt) + iou_loss(out_t, gt))
            loss1 = (F.binary_cross_entropy_with_logits(out, gt)
                     + ssim_loss(out, gt) + iou_loss(out, gt))
            loss2 = (F.binary_cross_entropy_with_logits(out_f, gt)
                     + ssim_loss(out_f, gt) + iou_loss(out_f, gt))

            loss = (loss_rgb + loss_t + loss1 + loss2) / 4

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.item()

            if i % 20 == 0 or i == total_step or i == 1:
                print(
                    f'{datetime.now()} | epoch:{epoch}/{opt.epoch} | step:{i}/{total_step}'
                    f' | lr={optimizer.param_groups[0]["lr"]:.6f} | loss={loss.item():.6f}'
                )
                logging.info(
                    f"##TRAIN##:Epoch [{epoch}/{opt.epoch}], Step [{i}/{total_step}], "
                    f"lr_bk:{optimizer.param_groups[0]['lr']:.6f}, Loss:{loss.item():.4f}"
                )

                # 可视化 RGB / GT / Out
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)

                res = out[0].clone().sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('out', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info(
            f'##TRAIN##:Epoch [{epoch:03d}/{opt.epoch:03d}], Loss_AVG: {loss_all:.4f}'
        )
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        # 按epoch间隔保存模型
        if epoch % 20 == 0:
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), 
                           os.path.join(save_path, f'SwinMCNet_epoch_{epoch}.pth'))
            else:
                torch.save(model.state_dict(), 
                           os.path.join(save_path, f'SwinMCNet_epoch_{epoch}.pth'))
            print(f"Save model at epoch {epoch} to {save_path}")

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), 
                       os.path.join(save_path, f'SwinMCNet_epoch_{epoch}_interrupt.pth'))
        else:
            torch.save(model.state_dict(), 
                       os.path.join(save_path, f'SwinMCNet_epoch_{epoch}_interrupt.pth'))
        print('save checkpoints successfully!')
        raise


# ====================== Main ======================
if __name__ == '__main__':
    print("Start train...")
    try:
        for epoch in range(1, opt.epoch + 1):
            # 学习率调度（余弦类调度）
            optimizer.param_groups[0]['lr'] = (
                (1 - abs((epoch) / (opt.epoch) * 2 - 1)) * opt.lr * 0.1
            )
            optimizer.param_groups[1]['lr'] = (
                (1 - abs((epoch) / (opt.epoch) * 2 - 1)) * opt.lr
            )
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)

            train(train_loader, model, optimizer, epoch, save_path)
    except Exception as e:
        print(f"Training error: {e}")
        logging.error(f"Training error: {e}")
    finally:
        writer.close()
        print("Training finished, close summary writer.")
'''
