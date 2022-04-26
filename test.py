import torch
import torchvision
from torchvision import transforms
import torch.multiprocessing

from dataset.widerface import get_widerface_trainval
from logger.default import Logger
from trainer.default import Trainer

import yaml
import argparse

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Training a face detection Faster-RCNN network on WIDERFace dataset.")

    parser.add_argument('--config', type = str, help = 'Path to the training configuration file.', required = True)
    parser.add_argument('--model-path', type = str, help = 'Path to the model file.', required = True)
    
    opt = parser.parse_args()

    logger = Logger()

    logger.log_info('Reading config file at {}.'.format(opt.config))
    config = None
    with open(opt.config, 'r') as f:
        config = yaml.load(f, Loader = yaml.Loader)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    logger.log_info('Loading WIDERFace dataset.')
    cfg_dataloader = config['DATALOADER']
    widerface_trainval = get_widerface_trainval(
        cfg_dataloader,
        transform = data_transforms
    )

    logger.log_info('Loading Faster-RCNN-Resnet50-FPN.')
    cfg_model = config['MODEL']
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained = cfg_model['PRETRAINED'], 
        num_classes = cfg_model['NUM_CLASSES']
    )

    model.load_state_dict(torch.load(opt.model_path, map_location = 'cpu'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log_info('Using {} as device.'.format(device))

    logger.log_info('Loading Trainer.')
    cfg_trainer = config['TRAINER']
    trainer = Trainer(
        config = cfg_trainer,
        device = device,
        model = model,
        trainval_dataloaders = widerface_trainval,
        optimizer = None,
        lr_scheduler = None,
        logger = logger
    )

    logger.log_info('Strating training.')
    mAP = trainer.valid_epoch()
    logger.log_info('mAP @ 0.5:0.95:0.05: {}'.format(mAP))

