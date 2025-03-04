import argparse
import random
import time

import numpy as np
import torch

from exp.exp_main import Exp_Main

torch.cuda.device_count()
start = time.perf_counter()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DGraFormer: Dynamic Graph Learning Guided Multi-Scale Transformer for Multivariate Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=202501, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='DGraFormer', help='model name')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='data_provider/data/ETT', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # DCGL
    parser.add_argument('--numpoint_win', type=int, default=24,
                        help='the number of points per window (e.g., hourly data for one day)')
    parser.add_argument('--w_bias', type=int, default=0, help='time points bias for the first window')
    parser.add_argument('--d_graph', type=int, default=30, help='dimensionality of the graph')
    parser.add_argument('--d_gcn', type=int, default=1, help='dimensionality of the graph message passing')
    parser.add_argument('--w_ratio', type=float, default=0.5, help='the weight focusing ratio')
    parser.add_argument('--mp_layers', type=int, default=2, help='the number of graph message passing layers ')

    parser.add_argument('--predictor_dropout', type=float, default=0.0, help='predictor_dropout')
    parser.add_argument('--patch_len', type=int, default=8, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')

    # MTT
    parser.add_argument('--n_vars', type=int, default=7, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_sl{}_ll{}_pl{}_nw{}_wb{}_dgra{}_dgcn{}_wr{}_ml{}_dm{}_nh{}_el{}_df{}_eb{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data_path,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.numpoint_win,
                args.w_bias,
                args.d_graph,
                args.d_gcn,
                args.w_ratio,
                args.mp_layers,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_ff,
                args.embed,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            with torch.cuda.device('cuda:0'):
                torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_sl{}_ll{}_pl{}_nw{}_wb{}_dgra{}_dgcn{}_wr{}_ml{}_dm{}_nh{}_el{}_df{}_eb{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data_path,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.numpoint_win,
            args.w_bias,
            args.d_graph,
            args.d_gcn,
            args.w_ratio,
            args.mp_layers,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_ff,
            args.embed,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        with torch.cuda.device('cuda:0'):
            torch.cuda.empty_cache()

end = time.perf_counter()
print('Running time: %s Seconds' % (end - start))
