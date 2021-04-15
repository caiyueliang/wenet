# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from wenet.dataset.dataset import AudioDataset, CollateFunc
from wenet.transformer.asr_model import init_asr_model, update_model_vocab_size
from wenet.utils.checkpoint import load_checkpoint, save_checkpoint
from wenet.utils.executor import Executor
from wenet.utils.scheduler import WarmupLR


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.rank',               # 分布式训练中的全局序号（多个并发时，实际只有一个是传0，rank为0的可以认为是主进程）
                        dest='rank',
                        default=0,
                        type=int,
                        help='global rank for distributed training')
    parser.add_argument('--ddp.world_size',         # 分布式训练的总进程数/GPU数
                        dest='world_size',
                        default=-1,
                        type=int,
                        help='''number of total processes/gpus for
                        distributed training''')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')
    parser.add_argument('--old_vocab_size', type=int, default=None, help='old_vocab_size')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # Set random seed
    torch.manual_seed(777)
    logging.info("[args] {}".format(args))

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
        logging.info("[configs] {}".format(configs))

    distributed = args.world_size > 1           # 分布式的标识位

    raw_wav = configs['raw_wav']                # 使用原始的wav文件或者kaldi特征进行训练的标志位。（true表示用原始的wav训练）

    # 处理训练集数据，配置中包含数据增强的参数信息
    train_collate_func = CollateFunc(**configs['collate_conf'], raw_wav=raw_wav)

    # 处理验证集，验证集上不进行数据增强
    cv_collate_conf = copy.deepcopy(configs['collate_conf'])
    cv_collate_conf['spec_aug'] = False                                             # 关闭频谱增强
    cv_collate_conf['spec_sub'] = False
    if raw_wav:
        cv_collate_conf['feature_dither'] = 0.0                                     # 特征抖动
        cv_collate_conf['speed_perturb'] = False                                    # 速度扰动
        cv_collate_conf['wav_distortion_conf']['wav_distortion_rate'] = 0           # 波形失真率
        # cv_collate_conf['wav_distortion_conf']['wav_dither'] = 0.0                # 音频抖动

    cv_collate_func = CollateFunc(**cv_collate_conf, raw_wav=raw_wav)

    dataset_conf = configs.get('dataset_conf', {})
    train_dataset = AudioDataset(args.train_data, **dataset_conf, raw_wav=raw_wav)  # 加载数据类
    cv_dataset = AudioDataset(args.cv_data, **dataset_conf, raw_wav=raw_wav)        # 加载数据类

    if distributed:
        # 多GPU分布式训练，需要做一些初始化操作，多个GPU绑定成一个Group。
        logging.info('training on multiple gpus, this gpu {}'.format(args.gpu))
        dist.init_process_group(args.dist_backend,
                                init_method=args.init_method,
                                world_size=args.world_size,
                                rank=args.rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        cv_sampler = torch.utils.data.distributed.DistributedSampler(cv_dataset, shuffle=False)
    else:
        train_sampler = None
        cv_sampler = None

    train_data_loader = DataLoader(train_dataset,
                                   collate_fn=train_collate_func,
                                   sampler=train_sampler,
                                   shuffle=(train_sampler is None),
                                   pin_memory=args.pin_memory,
                                   batch_size=1,
                                   num_workers=args.num_workers)
    cv_data_loader = DataLoader(cv_dataset,
                                collate_fn=cv_collate_func,
                                sampler=cv_sampler,
                                shuffle=False,
                                batch_size=1,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers)

    if raw_wav:
        input_dim = configs['collate_conf']['feature_extraction_conf']['mel_bins']  # 输入纬度: 80
    else:
        input_dim = train_dataset.input_dim
    vocab_size = train_dataset.output_dim

    # Save configs to model_dir/train.yaml for inference and export
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = raw_wav
    if args.rank == 0:
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # Init asr model from configs, 从配置中初始化语音识别模型
    model = init_asr_model(configs, args.old_vocab_size)
    logging.info("[model] {}".format(model))

    # !!!IMPORTANT!!!
    # [重要] 尝试通过脚本导出模型，如果失败，我们应该细化代码以满足脚本导出要求
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements
    script_model = torch.jit.script(model)
    script_model.save(os.path.join(args.model_dir, 'init.zip'))
    executor = Executor()

    # If specify checkpoint, load some info from checkpoint, 如果指定checkpoint，则从checkpoint加载一些信息
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)                 # 加载checkpoint
        model = update_model_vocab_size(model=model, configs=configs)   # 更新output_dim纬度
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1        # start_epoch,cv_loss,step等数据,是从checkpoint读取到
    cv_loss = infos.get('cv_loss', 0.0)
    step = infos.get('step', -1)

    num_epochs = configs.get('max_epoch', 100)
    model_dir = args.model_dir
    writer = None
    if args.rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))  # 输出一些信息到tensor board

    if distributed:
        assert (torch.cuda.is_available())
        # cuda model is required for nn.parallel.DistributedDataParallel
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        device = torch.device("cuda")
    else:
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    final_epoch = None
    configs['rank'] = args.rank
    configs['is_distributed'] = distributed
    if start_epoch == 0 and args.rank == 0:                     # 首次的训练，会保存一个init.pt模型
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)

    # Start training loop
    executor.step = step
    scheduler.set_step(step)
    for epoch in range(start_epoch, num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, scheduler, train_data_loader, device, writer, configs)     # 开始一个epoch的训练
        total_loss, num_seen_utts = executor.cv(model, cv_data_loader, device, configs)             # 验证集上验证结果
        if args.world_size > 1:                                                         # 多个进程的话，会统计多个进程的结果
            # all_reduce expected a sequence parameter, so we use [num_seen_utts].
            # all_reduce需要一个序列参数，所以我们使用[num_seen_utts]
            num_seen_utts = torch.Tensor([num_seen_utts]).to(device)
            # the default operator in all_reduce function is sum. all_reduce函数中的默认运算符是sum
            dist.all_reduce(num_seen_utts)
            total_loss = torch.Tensor([total_loss]).to(device)
            dist.all_reduce(total_loss)
            cv_loss = total_loss[0] / num_seen_utts[0]
            cv_loss = cv_loss.item()
        else:
            cv_loss = total_loss / num_seen_utts

        logging.info('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
        if args.rank == 0:      # 保存checkpoint: rank为0的程序[主程序]才会保存
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_checkpoint(
                model, save_model_path, {
                    'epoch': epoch,
                    'lr': lr,
                    'cv_loss': cv_loss,
                    'step': executor.step
                })
            writer.add_scalars('epoch', {'cv_loss': cv_loss, 'lr': lr}, epoch)
        final_epoch = epoch

    if final_epoch is not None and args.rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')          # 输出最终的模型final.pt
        os.symlink('{}.pt'.format(final_epoch), final_model_path)