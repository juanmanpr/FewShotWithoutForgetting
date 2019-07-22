#!/bin/sh

NUM=$1

# Extract features from the validation image split of the Imagenet.
CUDA_VISIBLE_DEVICES=0 python lowshot_save_features.py --config=imagenet_ResNet10CosineClassifierWeightAttN${NUM} --split=val
# Extract features from the training image split of the Imagenet.
CUDA_VISIBLE_DEVICES=0 python lowshot_save_features.py --config=imagenet_ResNet10CosineClassifierWeightAttN${NUM} --split=train

# Extract features from the validation image split of the Imagenet.
CUDA_VISIBLE_DEVICES=0 python lowshot_save_features.py --config=imagenet_ResNet10CosineClassifierWeightAvgN${NUM} --split=val
# Extract features from the training image split of the Imagenet.
CUDA_VISIBLE_DEVICES=0 python lowshot_save_features.py --config=imagenet_ResNet10CosineClassifierWeightAvgN${NUM} --split=train

# Extract features from the validation image split of the Imagenet.
CUDA_VISIBLE_DEVICES=0 python lowshot_save_features.py --config=imagenet_ResNet10CosineClassifierWeightAttN${NUM}_ortho --split=val
# Extract features from the training image split of the Imagenet.
CUDA_VISIBLE_DEVICES=0 python lowshot_save_features.py --config=imagenet_ResNet10CosineClassifierWeightAttN${NUM}_ortho --split=train

# Extract features from the validation image split of the Imagenet.
CUDA_VISIBLE_DEVICES=0 python lowshot_save_features.py --config=imagenet_ResNet10CosineClassifierWeightAvgN${NUM}_ortho --split=val
# Extract features from the training image split of the Imagenet.
CUDA_VISIBLE_DEVICES=0 python lowshot_save_features.py --config=imagenet_ResNet10CosineClassifierWeightAvgN${NUM}_ortho --split=train

# Training the model for the 2-shot.
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage2.py --config=imagenet_ResNet10CosineClassifierWeightAttN${NUM}
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage2.py --config=imagenet_ResNet10CosineClassifierWeightAttN${NUM}_ortho

# Training the model for the 2-shot.
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage2.py --config=imagenet_ResNet10CosineClassifierWeightAvgN${NUM}
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage2.py --config=imagenet_ResNet10CosineClassifierWeightAvgN${NUM}_ortho

