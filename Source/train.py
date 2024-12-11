import os
from super_gradients.training import Trainer, models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val

def main():
    # Path to the local dataset
    LOCAL_DATASET_DIR = r"D:\CV-OT\dataset"
    TRAIN_IMAGES_DIR = os.path.join(LOCAL_DATASET_DIR, 'train/images')
    TRAIN_LABELS_DIR = os.path.join(LOCAL_DATASET_DIR, 'train/labels')
    VALID_IMAGES_DIR = os.path.join(LOCAL_DATASET_DIR, 'valid/images')
    VALID_LABELS_DIR = os.path.join(LOCAL_DATASET_DIR, 'valid/labels')
    TEST_IMAGES_DIR = os.path.join(LOCAL_DATASET_DIR, 'test/images')
    TEST_LABELS_DIR = os.path.join(LOCAL_DATASET_DIR, 'test/labels')

    # Classes must be specified manually since you are not using Roboflow
    CLASSES = ['0']

    # Constants
    MODEL_ARCH = 'yolo_nas_s'
    BATCH_SIZE = 8
    MAX_EPOCHS = 100
    CHECKPOINT_DIR = r'D:\CV-OT\checkpoint'
    EXPERIMENT_NAME = 'My_model'
 
    # Ensure checkpoint directory exists
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # Initialize Trainer
    trainer = Trainer(experiment_name=EXPERIMENT_NAME, ckpt_root_dir=CHECKPOINT_DIR)

    # Dataset parameters for using local data
    dataset_params = {
        'data_dir': LOCAL_DATASET_DIR,
        'train_images_dir': TRAIN_IMAGES_DIR,
        'train_labels_dir': TRAIN_LABELS_DIR,
        'val_images_dir': VALID_IMAGES_DIR,
        'val_labels_dir': VALID_LABELS_DIR,
        'test_images_dir': TEST_IMAGES_DIR,
        'test_labels_dir': TEST_LABELS_DIR,
        'classes': CLASSES
    }

    # Loaders for the training, validation, and testing datasets
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['train_images_dir'],
            'labels_dir': dataset_params['train_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size': BATCH_SIZE,
            'num_workers': 2
        }
    )
    

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['val_images_dir'],
            'labels_dir': dataset_params['val_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size': BATCH_SIZE,
            'num_workers': 2
        }
    )

    # Define the model
    model = models.get(
        MODEL_ARCH,
        num_classes=len(CLASSES),
        pretrained_weights=None  # Set to None to not use any pretrained weights
    )

    num_classes = len(CLASSES)  # Calculate the number of classes

    # Define the training parameters including the valid metrics with the correct number of classes
    train_params = {
        'silent_mode': False,
        "average_best_models": True,
        "warmup_mode": "LinearEpochLRWarmup",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": MAX_EPOCHS,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=num_classes,
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=num_classes,
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50'
    }

    # Start training
    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_data,
        valid_loader=val_data
    )



if __name__ == '__main__':
    main()


