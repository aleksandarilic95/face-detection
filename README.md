# face-detection

Face detection model trained on WIDERFace dataset.
Dataset will automatically download to data/widerface.

To train the network, run:

    python3 train.py --config config/base.yaml
    
To test the network, run:

    python3 test.py --config config/base.yaml --model-path /path/to/model.pt
    
To inference a single image, run:

    python3 inference.py --config config/base.yaml --model-path /path/to/model.pt --image-path samples/test1.jpg
    
Inference results will be saved to inference/

Pretrained model (0.3mAP): [model.pt](https://drive.google.com/file/d/14rfK23pBHUbTWL0OkZ98dYErWbaLfZy8/view?usp=sharing)

Examples:
[test1.jpg](inference/inference_test1.jpg)
[test2.jpg](inference/inference_test2.jpg)