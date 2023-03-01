# TAMS
Unsupervised domain adaptation (UDA) aims to transfer knowledge from a labeled source domain to a related unlabeled target domain. Most existing works focus on minimizing the domain discrepancy to learn global domain-invariant representation by using CNN-based architecture while ignoring both transferable and discriminative local representation, e.g, pixel-level and patch-level representation. In this paper, we propose the Transferable Adversarial Masked Self-distillation based on Vision Transformer (ViT) architecture to enhance the transferability of UDA, named TAMS. Specifically, TAMS jointly optimizes three objectives to learn both task-specific class-level global representation and domain-specific local representation. Firstly, we introduce adversarial masked self-distillation objective to distill representation from a full image to the representation predicted from a masked image, which aims to learn task-specific global class-level representation. Secondly, we introduce masked image modeling objectives to learn local pixel-level representation. Thirdly, we introduce an adversarial weighted cross-domain adaptation objective to capture discriminative potentials of patch tokens, which aims to learn both transferable and discriminative domain-specific patch-level representation. Extensive studies on four benchmarks and the experimental results show that our proposed method can achieve remarkable improvements compared to previous state-of-the-art UDA methods.


### Environment (Python 3.8.12)
```
# Install Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh

# Install required packages
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
pip install tqdm==4.50.2
pip install tensorboard==2.8.0
# apex 0.1
conda install -c conda-forge nvidia-apex
pip install scipy==1.5.2
pip install ml-collections==0.1.0
pip install scikit-learn==0.23.2
```


### Datasets:

- Download [data](https://drive.google.com/file/d/1rnU49vEEdtc3EYVo7QydWzxcSuYqZbUB/view?usp=sharing) and replace the current `data/`

- Download images from [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw), [VisDA-2017](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) and put them under `data/`. For example, images of `Office-31` should be located at `data/office/domain_adaptation_images/`

### Pretrained ViT
Download the following models and put them in `checkpoint/`
- ViT-B_16 [(ImageNet-21K)](https://storage.cloud.google.com/vit_models/imagenet21k/ViT-B_16.npz?_ga=2.49067683.-40935391.1637977007)
- ViT-B_16 [(ImageNet)](https://console.cloud.google.com/storage/browser/_details/vit_models/sam/ViT-B_16.npz;tab=live_object)

### Training:
Add the source-only code. An example on `Office-31` dataset is as follows, where `dslr` is the source domain, `webcam` is the target domain:
```
CUDA_VISIBLE_DEVICES=0 python train.py --train_batch_size 64 --dataset office --name dw_source_only --train_list data/office/dslr_list.txt --test_list data/office/webcam_list.txt --num_classes 31 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256
```
TAMS training:
```
CUDA_VISIBLE_DEVICES=0 python3 main_mim.py --train_batch_size 32 --dataset office --name aw --source_list data/office/amazon_list.txt --target_list data/office/webcam_list.txt --test_list data/office/webcam_list.txt --num_classes 31 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1 --deta 0.1 --eta 0.1 --mask_ratio 0.4 --conf_threshold 0.7

CUDA_VISIBLE_DEVICES=0 python3 main_mim.py --train_batch_size 32 --dataset office --name dw --source_list data/office/dslr_list.txt --target_list data/office/webcam_list.txt --test_list data/office/webcam_list.txt --num_classes 31 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1 --deta 0.1 --eta 0.1 --mask_ratio 0.4 --conf_threshold 0.7

CUDA_VISIBLE_DEVICES=0 python3 main_mim.py --train_batch_size 32 --dataset office --name ad --source_list data/office/amazon_list.txt --target_list data/office/dslr_list.txt --test_list data/office/dslr_list.txt --num_classes 31 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1 --deta 0.1 --eta 0.1 --mask_ratio 0.4 --conf_threshold 0.7

CUDA_VISIBLE_DEVICES=0 python3 main_mim.py --train_batch_size 32 --dataset office --name da --source_list data/office/dslr_list.txt --target_list data/office/amazon_list.txt --test_list data/office/amazon_list.txt  --num_classes 31 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1 --deta 0.1 --eta 0.1 --mask_ratio 0.4 --conf_threshold 0.7

CUDA_VISIBLE_DEVICES=0 python3 main_mim.py --train_batch_size 32 --dataset office --name wa --source_list data/office/webcam_list.txt --target_list data/office/amazon_list.txt --test_list data/office/amazon_list.txt --num_classes 31 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1 --deta 0.1 --eta 0.1 --mask_ratio 0.4 --conf_threshold 0.7
```
### Attention Map Visualization:
```
python visualize.py --dataset office --name wa --num_classes 31 --img_size 256


Our code is largely borrowed from 
[TVT](https://github.com/uta-smile/TVT)
[CDAN](https://github.com/thuml/CDAN) 
[ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)



