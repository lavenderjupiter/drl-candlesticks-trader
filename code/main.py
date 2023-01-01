import torch
import numpy as np
import pandas as pd
import random
from torchvision import transforms as T
import argparse
import sys
import signal

import candle_trader
import ohlc_trader
import libs.data as image_data
import libs.ohlc_data as ohlc_data
from libs.utilities import HYPERPARAMS, INIT_PATHES

FOLDER_COUNTS = HYPERPARAMS['folder_counts']
DATE_FROM = '2020-01-01'
IMAGE_DATA_DIR = INIT_PATHES['image']
DATA_DIR = './data/'

def quit(signum, frame):
    sys.exit()

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # default GPU
    torch.cuda.manual_seed_all(seed) # in the case of more than one GPU
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', type=int, default=0, help='Set the Cuda Count. [0, 1, ..., -1(cpu)]')
    parser.add_argument('-t', '--trader', type=str, default='test', help='Train, Validate or Test the Trader. [train, valid, test]')
    parser.add_argument('-g', '--algorithm', type=str, default='dueling-dqn', help='Set the Algorithm. [dqn, dueling-dqn, ppo, sac, bh, rp, hp]')
    parser.add_argument('-a', '--attention', type=bool, default=True, help='Use Attention in the CNN Backbone. [True, False]')
    parser.add_argument('-r', '--raw', type=str, default='None', help='Train the Models by Using Raw Data. [None, single, multi]')
    parser.add_argument('-m', '--mode', type=str, default='single', help='Set the Training Mode to Train More than One or Only One Single Agent. [many, single]')
    parser.add_argument('-f', '--folders', type=str, default='0', help='Set the Training or Testing folder count of Agent. [0 ~ 8]')
    parser.add_argument('-v', '--record', default=False, help='Record the Test Environment. [True, False]')
    parser.add_argument('-l', '--load', default=False, help='Load Checkpoint. [True, False]')
    parser.add_argument('-p', '--cost', type=float, default=0.05, help='transaction cost(percent), e.g. 0.05')
    args = parser.parse_args()

    torch.cuda.empty_cache()
    set_random_seed(2022)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # mean=0.5, std=0.5, normalize the image value to [-1, 1]
    ])

    if args.cuda == -1:
        DEVICE = torch.device('cpu')
    else:
        DEVICE = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available()  else 'cpu')

    if args.raw == 'None':
        trader = candle_trader
        data = image_data
        dataset = data.get_folder_dataset(FOLDER_COUNTS, IMAGE_DATA_DIR, transform, HYPERPARAMS, date_from=DATE_FROM)
        testset = data.get_folder_dataset(FOLDER_COUNTS, IMAGE_DATA_DIR, transform, HYPERPARAMS, date_from=DATE_FROM, learn=False) # comment this line while testing on 15-day testset
        #------ specific test set by a given date and days ------
        # testset = data.get_test_dataset(IMAGE_DATA_DIR, transform, date_from='2022-08-15')
    else:
        trader = ohlc_trader
        trader.MODE = args.raw[0]
        data = ohlc_data
        dataset = data.get_folder_dataset(FOLDER_COUNTS, DATA_DIR, args.raw, HYPERPARAMS, date_from=DATE_FROM)
        testset = data.get_folder_dataset(FOLDER_COUNTS, DATA_DIR, args.raw, HYPERPARAMS, date_from=DATE_FROM, learn=False) # comment this line while testing on 15-day testset
        #------ specific test set by a given date and days ------
        # testset = data.get_test_dataset(DATA_DIR, args.raw, date_from='2022-08-15')

    folders = [int(f) for f in args.folders.split(',')]
    trader.ACCOUNT_FEATURES = ['position', 'fundrate'] # gain, position
    trader.INITIAL_ACCOUNT_BALANCE = 100000
    trader.COMMISSION_PERCENT = args.cost # transaction cost (percent)

    print('using raw data:', args.raw != 'None')
    if args.trader == 'train':
        train_process = trader.train(DEVICE, args.algorithm, args.attention)
        if args.mode == 'many':
            signal.signal(signal.SIGINT, quit)
            signal.signal(signal.SIGTERM, quit)
            for folder in range(folders[0], FOLDER_COUNTS):
                print('>>>> training agent: {}'.format(folder))
                train_set, _ = dataset[folder]
                first_data = train_set[0]
                date_time = first_data[-1]
                print('from date: ', date_time)
                train_process.run(folder, train_set, run_checkpoint=args.load)
        else:
            for folder in folders:
                print('>>>> training agent: {}'.format(folder))
                train_set, _ = dataset[folder]
                first_data = train_set[0]
                date_time = first_data[-1]
                print('from date: ', date_time)
                train_process.run(folder, train_set, run_checkpoint=args.load)
    elif args.trader == 'valid':
        folder = folders[0]
        print('>>>> validating agent: {}'.format(folder))
        _, valid_set = dataset[folder]
        first_data = valid_set[0]
        date_time = first_data[-1]
        print('from date: ', date_time)
        valid_process = trader.valid(args.algorithm)
        valid_process.run(folder, valid_set)
    else:
        first_data = testset[0]
        date_time = first_data[-1]
        print('test from date: ', date_time)
        test_process = trader.test(args.algorithm, date_time)
        

        if args.algorithm in ['dqn', 'dueling-dqn', 'ppo', 'sac']:
            test_process.run(testset, args.record)
        elif args.raw == 'None':
            dataframe = pd.read_csv('./data/15m_ohlcv.csv', index_col=0)
            testset_df = data.get_folder_dataframe(FOLDER_COUNTS, dataframe, HYPERPARAMS, date_from=DATE_FROM, learn=False) # comment this line while testing on 15-day testset
            #------ specific test set by a given date and days ------
            # testset_df = data.get_test_dataframe(dataframe, date_from='2022-08-15')
            test_process.baseline(testset, testset_df)
