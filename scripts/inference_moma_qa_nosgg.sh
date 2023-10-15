#!/usr/bin/env bash



GPU_NUM=2
TOTAL_GPU=$((WORLD_SIZE * GPU_NUM))

checkpoint_dir='mPLUG2_MSRVTT_QA.pth'
output_dir='/home/data/models/videoqa_momaqa_'${TOTAL_GPU}

mkdir -p ${output_dir}
python -u -m torch.distributed.launch --nproc_per_node=$GPU_NUM \
    --master_addr=127.0.0.1 \
	--master_port=10024 \
	--nnodes=1 \
	--node_rank=0 \
    --use_env \
    video_qa_mplug2.py \
    --config configs_video/VideoQA_moma_large_nosgg.yaml \
    --text_encoder bert-large-uncased \
    --text_decoder bert-large-uncased \
    --output_dir ${output_dir} \
    --checkpoint ${checkpoint_dir} \
    --do_two_optim \
    --evaluate \
    --do_amp 2>&1 | tee ${output_dir}/train.log