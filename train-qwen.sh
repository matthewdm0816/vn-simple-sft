export MACA_PATH=/opt/maca
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export PATH=${CUDA_PATH}/bin:${MACA_CLANG_PATH}:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=1
export MACA_SMALL_PAGESIZE_ENABLE=1
export PYTORCH_ENABLE_SAME_SAME_RAND_A100=1
export SET_DEVICE_NUMA_PREFERRED=1
export MCCL_P2P_LEVEL=SYS
export MCCL_FAST_WRITE_BACK=1
export MCCL_EARLY_WRITE_BACK=15
export MCCL_NET_GDR_LEVEL=SYS
export MCCL_CROSS_NIC=1
export MHA_BWD_NO_ATOMIC_F64=1
export TOKENIZERS_PARALLELISM=false
export MXLOG_LEVEL=err
#export MXLOG_LEVEL=debug
#export MACA_LAUNCH_BLOCKING=1
export MCCL_ENABLE_FC=0
export LANG=en_US.UTF-8
export OMP_NUM_THREADS=1

export PORT=$(shuf -i 29000-30000 -n 1)
export SLURM_GPUS=$(($(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c)+1))

export PYTORCH_DISABLE_ELEMENTWISE_2_1_BROADCAST=1
export PYTORCH_DISABLE_ELEMENTWISE_2_2_BROADCAST=1
export PYTORCH_DISABLE_CONTINUOUS_REDUCE=1
export PYTORCH_DISABLE_TRIU_TRIL_OPT=1
export PYTORCH_DISABLE_ELEMENTWISE_4_2_OPT=1

#unset PYTORCH_DISABLE_ELEMENTWISE_2_1_BROADCAST
#unset PYTORCH_DISABLE_CONTINUOUS_REDUCE
#unset PYTORCH_DISABLE_TRIU_TRIL_OPT
#unset PYTORCH_DISABLE_ELEMENTWISE_4_2_OPT

export PYTORCH_REDUCE_ENABLE_PRINT=1
unset PYTORCH_REDUCE_ENABLE_PRINT

echo "Number of processes (GPUs): $SLURM_GPUS"

# torchrun --nproc_per_node=8 \
#   test-train-ddp-new.py --per_device_train_batch_size 4 --model_name ../Qwen2.5-1.5B-Instruct # --fp32

accelerate launch --config_file "finetune-fuyu.yaml" --num_processes=$SLURM_GPUS --main_process_port=$PORT \
    test-train-ddp-new.py \
    --per_device_train_batch_size 4 \
    --num_workers 4 \
    --model_name "../Qwen2.5-1.5B-Instruct"
