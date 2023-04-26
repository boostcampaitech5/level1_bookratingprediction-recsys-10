#!/bin/bash
# main.py 와 같은 공간에 둔다

# 하이퍼파라미터 값들의 리스트를 정의합니다.
declare -a epochs_list=("10" "15")
declare -a batch_size_list=("256")
declare -a lr_list=("0.001" "0.003" "0.01")
declare -a model_list=("CNN_FM2")
declare -a cnn_embed_dim_list=("64" "128")
declare -a cnn_latent_dim_list=("12" "24" "48")

# 모든 하이퍼파라미터 조합에 대해 실험을 수행합니다.
for epochs in "${epochs_list[@]}"; do
  for batch_size in "${batch_size_list[@]}"; do
    for lr in "${lr_list[@]}"; do
      for model in "${model_list[@]}"; do
        for cnn_embed_dim in "${cnn_embed_dim_list[@]}"; do
          for cnn_latent_dim in "${cnn_latent_dim_list[@]}"; do
            # 모델 학습을 위한 명령어를 실행합니다.
            echo "Running with model=$model, epochs=$epochs, batch_size=$batch_size, lr=$lr, cnn_embed_dim=$cnn_embed_dim, cnn_latent_dim = $cnn_latent_dim,vector_create=False"
            python main.py --model "$model" --epochs "$epochs" --batch_size "$batch_size" --lr "$lr" --cnn_embed_dim "$cnn_embed_dim" --cnn_latent_dim "$cnn_latent_dim"
          done
        done
      done
    done
  done
done