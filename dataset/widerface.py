from torchvision.datasets import WIDERFace
from torch.utils.data import Dataset, DataLoader

def get_widerface_trainval(root,
                       download = False,
                       batch_size = 1,
                       shuffle = False,
                       num_workers = 0,
                       transform = None,
                       target_transform = None):
    result_list = []

    for split in ['train', 'val']:
        dataset = get_dataset(root, split, transform, target_transform, download)
        loader = get_dataloader(dataset, batch_size, shuffle, num_workers)
        result_list.append(loader)

    return {'train': result_list[0],
            'val': result_list[1]}

def get_widerface_test(root,
                       download = False,
                       batch_size = 1,
                       shuffle = False,
                       num_workers = 0,
                       transform = None,
                       target_transform = None):
    dataset = get_dataset(root, 'test', transform, target_transform, download)
    loader = get_dataloader(dataset, batch_size, shuffle, num_workers)

    return {'test': loader}

def get_dataloader(dataset,
                   batch_size = 1,
                   shuffle = False,
                   num_workers = 0):
    return DataLoader(dataset = dataset,
                      batch_size = batch_size,
                      shuffle = shuffle,
                      num_workers = num_workers)

def get_dataset(root,
                split = 'train',
                transform = None,
                target_transform = None,
                download = False):
    return WIDERFace(root = root,
                     split = split,
                     transform = transform,
                     target_transform = target_transform,
                     download = download)

