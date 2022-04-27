import torch
from torchvision.datasets import WIDERFace
from torch.utils.data import Dataset, DataLoader

def get_widerface_trainval(config,
                       transform = None,
                       target_transform = None):
    result_list = []

    for split in ['train', 'val']:
        dataset = get_dataset(
            root = config['ROOT'], 
            split = split, 
            transform = transform, 
            target_transform = target_transform, 
            download = True
        )
        loader = get_dataloader(
            dataset = dataset, 
            batch_size = config['BATCH_SIZE'], 
            shuffle = config['SHUFFLE'], 
            num_workers = config['NUM_WORKERS']
        )
        result_list.append(loader)

    return {'train': result_list[0],
            'val': result_list[1]}

def get_widerface_test(config,
                       transform = None,
                       target_transform = None):
    dataset = get_dataset(
        root = config['ROOT'], 
        split = 'test', 
        transform = transform, 
        target_transform = target_transform, 
        download = True
    )
    loader = get_dataloader(
        dataset = dataset, 
        batch_size = config['BATCH_SIZE'], 
        shuffle = config['SHUFFLE'], 
        num_workers = config['NUM_WORKERS']
    )

    return {'test': loader}

def get_dataloader(dataset,
                   batch_size = 1,
                   shuffle = False,
                   num_workers = 0):
    return DataLoader(dataset = dataset,
                      batch_size = batch_size,
                      shuffle = shuffle,
                      num_workers = num_workers,
                      collate_fn = detection_collate)

def get_dataset(root,
                split = 'train',
                transform = None,
                target_transform = None,
                download = False):
    return WIDERFace(
        root = root,
        split = split,
        transform = transform,
        target_transform = target_transform,
        download = download
    )

# Collate function, since different samples
# in a batch have different number of ground 
# truth face detections present.
def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        # Since this is a big dataset, we're
        # only working with samples that have
        # 2 or 3 detections.
        if sample[1]['bbox'].shape[0] not in [2,3]:
            continue

        # There are some errors in the dataset,
        # so we're neglecting samples that have
        # degenerated bounding boxes.
        if torch.any((sample[1]['bbox'][:,0:2] - sample[1]['bbox'][:,2:4]) <= 10):
            continue
        if torch.any((sample[1]['bbox'][:,:]) == 0):
            continue

        imgs.append(sample[0])
        targets.append(torch.LongTensor(sample[1]['bbox']))
    return imgs, targets
