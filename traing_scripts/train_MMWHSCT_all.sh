CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train/train.py \
    model=ddpm \
    dataset=MMWHS \
    dataset.data_type=CT \
    dataset.root_dir=./data/MMWHS/CT/all \
    dataset.mode=train \
    model.diffusion_img_size=64 \
    model.diffusion_depth_size=64 \
    model.diffusion_num_channels=6 \
    model.batch_size=12 \
    model.results_folder=./Model/DiffAtlas_MMWHS-CT_all \
    model.load_milestone=False \
    model.save_and_sample_every=100 \
    model.train_num_steps=10001 \
    model.timesteps=300 \
    model.num_workers=20 \

