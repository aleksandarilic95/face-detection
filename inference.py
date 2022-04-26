import torch
import torchvision
from torchvision import transforms
import torch.multiprocessing

from dataset.widerface import get_widerface_trainval
from logger.default import Logger
from trainer.default import Trainer

import yaml
import argparse
import cv2

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Inference a face detection Faster-RCNN network on WIDERFace dataset.")

    parser.add_argument('--config', type = str, help = 'Path to the training configuration file.', required = True)
    parser.add_argument('--model-path', type = str, help = 'Path to the model file.', required = True)
    parser.add_argument('--image-path', type = str, help = 'Path to the image file.', required = True)
    
    opt = parser.parse_args()

    config = None
    with open(opt.config, 'r') as f:
        config = yaml.load(f, Loader = yaml.Loader)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cfg_model = config['MODEL']
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained = cfg_model['PRETRAINED'], 
        num_classes = cfg_model['NUM_CLASSES']
    )

    model.load_state_dict(torch.load(opt.model_path, map_location = 'cpu'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    im = cv2.imread(opt.image_path, cv2.COLOR_BGR2RGB)
    im_tensor = data_transforms(im).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(im_tensor)

    cfg_trainer = config['TRAINER']
    keep_indexes = torchvision.ops.nms(out[0]['boxes'], out[0]['scores'], cfg_trainer['NMS_THRESHOLD'])
    
    for index in keep_indexes:
        if out[0]['scores'][index] > cfg_trainer['SCORE_THRESHOLD']:
            x1 = int(out[0]['boxes'][index][0])
            y1 = int(out[0]['boxes'][index][1])
            x2 = int(out[0]['boxes'][index][2])
            y2 = int(out[0]['boxes'][index][3])

            im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 1)

    im_name = opt.image_path.split('/')[-1]
    cv2.imwrite('inference/inference_{}'.format(im_name), im)
