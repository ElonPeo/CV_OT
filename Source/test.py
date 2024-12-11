import torch
from super_gradients.training import models
import os
from super_gradients.training import Trainer
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_val

def main():
    CLASSES = ['0']
    LOCAL_DATASET_DIR = r"D:\CV-OT\dataset"
    TEST_IMAGES_DIR = os.path.join(LOCAL_DATASET_DIR, 'test/images')
    TEST_LABELS_DIR = os.path.join(LOCAL_DATASET_DIR, 'test/labels')
    BATCH_SIZE = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = models.get(
        model_name='yolo_nas_s',
        num_classes=len(CLASSES),
        checkpoint_path=r"checkpoint\My_model\RUN_20241118_203331_103538\ckpt_best.pth"
    ).to(DEVICE)

    dataset_params = {
        'data_dir': LOCAL_DATASET_DIR,
        'test_images_dir': TEST_IMAGES_DIR,
        'test_labels_dir': TEST_LABELS_DIR,
        'classes': CLASSES
    }

    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['test_images_dir'],
            'labels_dir': dataset_params['test_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size': BATCH_SIZE,
            'num_workers': 2
        }
    )

    trainer = Trainer(experiment_name="My_model", ckpt_root_dir="checkpoint")

    # Thực hiện kiểm tra và lưu kết quả
    trainer.test(
        model=model,
        test_loader=test_data,
        test_metrics_list=DetectionMetrics_050(
            score_thres=0.1, 
            top_k_predictions=300, 
            num_cls=len(dataset_params['classes']), 
            normalize_targets=True, 
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01, 
                nms_top_k=1000, 
                max_predictions=300,                                                                              
                nms_threshold=0.7
            )
        )
    )


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
