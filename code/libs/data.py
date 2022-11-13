import os
from PIL import Image
import datetime as dt

from torch.utils.data import Dataset
from torch.utils.data import Subset

class CoinDataset(Dataset):
    def __init__(self, image_dir, transform = None):
        self.image_dir = image_dir # dataset folder
        self.data_path = self._read_images()
        self.transform = transform

    def set_transform(self, transform):
        self.transform = transform

    def get_date_index(self, date):
        date = date + '_' + '00-00-00'
        for index, path_dict in enumerate(self.data_path):
            date_time = path_dict['path'].split('/')[-1][:-4].split('_')[:-1]
            datetime = '_'.join(s for s in date_time)
            if datetime == date:
                return index

  # get the dataset length (number of image paths)       
    def __len__(self):
        return len(self.data_path)
    
  # get the single image and label    
    def __getitem__(self, index):
        path = self.data_path[index]['path']
        date_time = path.split('/')[-1][:-4].split('_')[:-1]
        date_time_label = date_time[0] + '_' + date_time[1]
        scaled_price = self.data_path[index]['scaled']
        prices = self.data_path[index]['prices']
        fund_rate = self.data_path[index]['fundrate']
        image = Image.open(path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)    

        return image, scaled_price, prices, fund_rate, date_time_label
  
  # read image path and label  
    def _read_images(self):
        data = []
        for (root, dirs, files) in os.walk(self.image_dir):
            files.sort()
            for file in files:  
                img_path = root + '/' + file
                prices = img_path.split('/')[-1][:-4].split('_')[-1] # get prices from file name
                scaled_price, current_close, next_close, fund_rate = prices[1:-1].split(',')
                sample = {'path': img_path, 'scaled': float(scaled_price), 'prices': [float(current_close), float(next_close)], 'fundrate': float(fund_rate)}
                data.append(sample)

        return data

def add_days(date, days):
    start_date  = dt.datetime.strptime(date,'%Y-%m-%d')
    delta = dt.timedelta(days=days)
    end_date  = start_date + delta
    return str(end_date).split()[0]


def get_folder_dataset(count, image_path, transform, hyperparams, date_from, learn=True):
    datasets = {}
    dataset = CoinDataset(image_path, transform=transform)
    train_days = hyperparams['train_days'] #180
    valid_days = hyperparams['valid_days'] #80
    test_days = hyperparams['test_days'] #30
    roll_offset = hyperparams['roll_offset'] #80

    for i in range(count):
        offset_days = roll_offset * i # roll up with the days of roll_offset

        date_start = add_days(date_from, offset_days)
        train_start = dataset.get_date_index(date_start) 
        date_end_trian = add_days(date_start, train_days)
        train_end = dataset.get_date_index(date_end_trian) 
        indices = list(range(train_start, train_end))
        train_set = Subset(dataset, indices)

        date_end_valid = add_days(date_end_trian, valid_days)
        valid_end = dataset.get_date_index(date_end_valid) 
        indices = list(range(train_end, valid_end))
        valid_set = Subset(dataset, indices)
        datasets[i] = [train_set, valid_set]

    if learn:
        return datasets
    else:
        date_finish = add_days(date_end_valid, test_days)
        test_end = dataset.get_date_index(date_finish) 
        indices = list(range(valid_end, test_end))
        test_set = Subset(dataset, indices)
        return test_set      

def get_folder_dataframe(count, dataframe, hyperparams, date_from, learn=True):
    """
    date_from format example '2020-01-01'
    """
    dataframes = {}
    train_days = hyperparams['train_days'] 
    valid_days = hyperparams['valid_days']
    test_days = hyperparams['test_days']
    roll_offset = hyperparams['roll_offset'] 

    for i in range(count):
        offset_days = roll_offset * i # roll up with the days of (validation + roll_offset)

        date_start = add_days(date_from, offset_days) 
        date_end_train = add_days(date_start, train_days) 
        train_set = dataframe[date_start:date_end_train]
        train_set = train_set.iloc[train_set.index < date_end_train]

        date_end_valid = add_days(date_end_train, valid_days) 
        valid_set = dataframe[date_end_train:date_end_valid]
        valid_set = valid_set.iloc[valid_set.index < date_end_valid] 

        dataframes[i] = [train_set, valid_set]
        
    if learn:
        return dataframes
    else:
        date_finish = add_days(date_end_valid, test_days)
        test_set = dataframe[date_end_valid:date_finish]
        test_set = test_set.iloc[test_set.index < date_finish] 
        return test_set


def get_test_dataset(image_path, transform, date_from, days=15):
    dataset = CoinDataset(image_path, transform=transform)
    test_days = days

    test_start = dataset.get_date_index(date_from)
    date_to = add_days(date_from, test_days)
    test_end = dataset.get_date_index(date_to)
    indices = list(range(test_start, test_end))
    test_set = Subset(dataset, indices)
    return test_set     

def get_test_dataframe(dataframe, date_from, days=15):
    """
    date_from format example '2020-01-01'
    """
    test_days = days
    date_finish = add_days(date_from, test_days)
    test_set = dataframe[date_from:date_finish]
    test_set = test_set.iloc[test_set.index < date_finish] 
    return test_set