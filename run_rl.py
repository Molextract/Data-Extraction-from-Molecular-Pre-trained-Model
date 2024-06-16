#!/usr/bin/env python3

import os
import random
from model import sac
import numpy as np
import dgl
import torch
from tensorboardX import SummaryWriter
from core_motif import GCNActorCritic
import gym
import warnings
import time

# warnings.filterwarnings("ignore", message="WARNING: not removing hydrogen atom without neighbors")
warnings.filterwarnings("ignore", message="WARNING: not removing hydrogen atom without neighbors.$")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def gpu_setup(use_gpu, gpu_id):
    if torch.cuda.is_available() and use_gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda:" + str(gpu_id))
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def train(args, seed, writer=None):
    workerseed = args.seed
    set_seed(workerseed)

    # device
    gpu_use = False
    gpu_id = None
    if args.gpu_id is not None:
        gpu_id = int(args.gpu_id)
        gpu_use = True

    device = gpu_setup(gpu_use, gpu_id)

    env = gym.make('molecule-v0')
    params_env = {'scaffold_idx': args.scaffold_idx,
                  'model': args.gnn,
                  'reward': args.reward,
                  'pretrain': args.pretrain,
                  'device': device,
                  'model_name': f'reg_stre_10.0/scaffold_{args.scaffold_idx}.pth'}
    env.init(docking_config=args.docking_config, params=params_env, ratios=args.ratios,
             reward_step_total=args.reward_step_total, is_normalize=args.normalize_adj,
             has_feature=bool(args.has_feature), max_action=args.max_action, min_action=args.min_action,
             version=args.version)
    env.seed(workerseed)

    SAC = sac(writer, args, env, actor_critic=GCNActorCritic, ac_kwargs=dict(), seed=seed,
              # step s_per_epoch=50, epochs=10, replay_size=int(1e6), gamma=0.99,
              # polyak=0.995, lr=args.init_lr, alpha=args.init_alpha, batch_size=args.batch_size, start_steps=128,
              steps_per_epoch=128, epochs=args.epochn, replay_size=int(1e6), gamma=0.99,
              polyak=0.995, lr=args.init_lr, alpha=args.init_alpha, batch_size=args.batch_size,
              start_steps=args.start_steps,
              update_after=args.update_after, update_every=args.update_every, update_freq=args.update_freq,
              expert_every=5, num_test_episodes=8, max_ep_len=args.max_action,
              save_freq=2000, train_alpha=True)
    SAC.train()
    env.close()


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def molecule_arg_parser():
    parser = arg_parser()

    # Choose RL model
    parser.add_argument('--rl_model', type=str, default='sac')  # sac, td3, ddpg

    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--train', type=int, default=1, help='training or inference')
    # env
    parser.add_argument('--env', type=str, help='environment name: molecule; graph', default='molecule')
    parser.add_argument('--seed', help='RNG seed', type=int, default=666)
    parser.add_argument('--num_steps', type=int, default=int(5e7))
    parser.add_argument('--method', type=str, default='thre')
    parser.add_argument('--buffer', type=bool, default='False')

    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--sw', type=str, default='sw')
    # parser.add_argument('--name_full', type=str, default='test')
    parser.add_argument('--name_full_load', type=str, default='')

    parser.add_argument('--reward_step_total', type=float, default=0.5)
    parser.add_argument('--target', type=str, default='fa7', help='fa7, parp1, 5ht1b')

    parser.add_argument('--intr_rew', type=str, default=None)  # intr, mc
    parser.add_argument('--intr_rew_ratio', type=float, default=5e-1)

    parser.add_argument('--tau', type=float, default=1)

    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--epochn', type=int, default=50)

    # model update
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    # parser.add_argument('--update_every', type=int, default=256)
    parser.add_argument('--update_every', type=int, default=256)
    parser.add_argument('--update_freq', type=int, default=256)
    parser.add_argument('--update_after', type=int, default=2000)
    parser.add_argument('--start_steps', type=int, default=3000)

    # model save and load
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--load_step', type=int, default=250)

    # graph embedding
    parser.add_argument('--gcn_type', type=str, default='GCN')
    parser.add_argument('--gcn_aggregate', type=str, default='sum')
    parser.add_argument('--graph_emb', type=int, default=1)
    parser.add_argument('--emb_size', type=int, default=64)  # default 64
    parser.add_argument('--has_residual', type=int, default=0)
    parser.add_argument('--has_feature', type=int, default=1)

    parser.add_argument('--normalize_adj', type=int, default=0)
    parser.add_argument('--bn', type=int, default=0)

    parser.add_argument('--layer_num_g', type=int, default=3)

    # action
    parser.add_argument('--is_conditional', type=int, default=0)
    # parser.add_argument('--conditional', type=str, default='low')
    parser.add_argument('--max_action', type=int, default=2)
    parser.add_argument('--min_action', type=int, default=1)

    # SAC
    parser.add_argument('--target_entropy', type=float, default=1.)
    # parser.add_argument('--target_entropy', type=float, default=0.05.)
    parser.add_argument('--init_alpha', type=float, default=1.)
    parser.add_argument('--desc', type=str, default='ecfp')  # ecfp
    parser.add_argument('--init_pi_lr', type=float, default=1e-4)
    parser.add_argument('--init_q_lr', type=float, default=1e-4)
    parser.add_argument('--init_alpha_lr', type=float, default=5e-4)
    parser.add_argument('--alpha_max', type=float, default=20.)
    parser.add_argument('--alpha_min', type=float, default=.05)

    # MC dropout
    parser.add_argument('--active_learning', type=str, default='freed_pe')  # "mc", "per", None
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--n_samples', type=int, default=5)

    # On-policy
    parser.add_argument('--n_cpus', type=int, default=1)
    parser.add_argument('--steps_per_epoch', type=int, default=257)

    # 
    parser.add_argument("--scaffold_idx", type=int, default=0)
    parser.add_argument('--gnn', type=str, default='gcl')
    parser.add_argument('--reward', type=str, default='score')

    return parser


def main():
    args = molecule_arg_parser().parse_args()
    print(args)
    args.name_full = args.env + '_' + args.name

    docking_config = dict()

    assert args.target in ['fa7', 'parp1', '5ht1b'], "Wrong target type"
    if args.target == 'fa7':
        box_center = (10.131, 41.879, 32.097)
        box_size = (20.673, 20.198, 21.362)
        docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/fa7/receptor.pdbqt'
    elif args.target == 'parp1':
        box_center = (26.413, 11.282, 27.238)
        box_size = (18.521, 17.479, 19.995)
        docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/parp1/receptor.pdbqt'
    elif args.target == '5ht1b':
        box_center = (-26.602, 5.277, 17.898)
        box_size = (22.5, 22.5, 22.5)
        docking_config['receptor_file'] = 'ReLeaSE_Vina/docking/5ht1b/receptor.pdbqt'
        docking_config['temp_dir'] = '5ht1b_tmp'

    box_parameter = (box_center, box_size)
    docking_config['vina_program'] = 'bin/qvina02'
    docking_config['box_parameter'] = box_parameter
    docking_config['temp_dir'] = 'tmp'
    if args.train:
        docking_config['exhaustiveness'] = 1
    else:
        docking_config['exhaustiveness'] = 4
    docking_config['num_sub_proc'] = 10
    docking_config['num_cpu_dock'] = 5
    docking_config['num_modes'] = 10
    docking_config['timeout_gen3d'] = 30
    docking_config['timeout_dock'] = 100

    ratios = dict()
    ratios['logp'] = 0
    ratios['qed'] = 0
    ratios['sa'] = 0
    ratios['mw'] = 0
    ratios['filter'] = 0
    ratios['docking'] = 1

    args.docking_config = docking_config
    args.ratios = ratios

    # check and clean
    if not os.path.exists('molecule_gen'):
        os.makedirs('molecule_gen')
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')

    # writer = SummaryWriter(comment='_' + args.name)
    folder_path = './log/' + str(args.sw)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    writer = SummaryWriter(folder_path)

    # device
    gpu_use = False
    gpu_id = None
    if args.gpu_id is not None:
        gpu_id = int(args.gpu_id)
        gpu_use = True
    device = gpu_setup(gpu_use, gpu_id)
    args.device = device

    if args.gpu_id is None:
        torch.set_num_threads(256)
        print(torch.get_num_threads())

    t1 = time.time()
    train(args, seed=args.seed, writer=writer)
    t2 = time.time()
    print("Total time:" + str(t1 - t2))


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    set_random_seed()
    main()
