import torch
import yolov4_env


if __name__ == "__main__":
    # Training configurations are set as module-wide
    #   global variables in yolov4_env(yolov4_env.py:22-57)
    # Change the default configurations as below:
    device = 'cuda:5'
    yolov4_env.device = device
    yolov4_env.batch_size = 64
    cspdarknet53_weight_path = '/data/sdv1/xiezhouyang/models/project_assets/detectabc/cspdarknet53.pth'
    yolov4_weight_path = '/data/sdv1/xiezhouyang/models/project_assets/detectabc/yolov4_voc0712.pth'

    model = yolov4_env.make_model()
    # Original YOLOv4 has 3 YOLOv3 heads at 3 scales,
    #   but for training feasibility of VOC, we only use the max scale head.
    _, _, yolov3loss_high = yolov4_env.make_yolo_loss()

    phase = 'train'

    if phase == 'train':
        # by default, the backbone would be frozen during training
        backbone_weights = torch.load(
            cspdarknet53_weight_path, map_location=device)
        model.backbone.load_state_dict(backbone_weights)
        yolov4_env.train(model, None, None, yolov3loss_high)
    elif phase == 'test':
        # Better set lower batch size for testing.
        # Because YOLO produces massive outputs while testing,
        #   which is memory consuming.
        model_weights = torch.load(
            yolov4_weight_path, map_location=device)
        model.load_state_dict(model_weights)
        mAP = yolov4_env.test('test', model, None, None, yolov3loss_high)
        for k, v in mAP.items():
            print(f'mAP {k}: {v}')
    else:
        raise ValueError
