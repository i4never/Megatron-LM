set -xe
ulimit -n 102400

timestamp=$(date +%Y%m%d_%H%M%S)
env_prefix="/mnt/afs/miniconda3/envs/fc-base"

# Login to wandb
echo "================================"
echo "wandb login start"
${env_prefix}/bin/wandb login 5e8f8c158ea1292e007874620472e59eac754d73
echo "wandb login end"
echo "================================"

# distributed
gpus_per_node=8
master_addr=${MASTER_ADDR}
master_port=${MASTER_PORT}
node_rank=${RANK}
world_size=${WORLD_SIZE}

# data
data_path="/mnt/m2/cong.fu/code/.Megatron-LM/local/data/test_text_document"

# parallel
tp=1
cp=1
ep=4
pp=2
eptp=1  # Default to tp if not set

seq_length=4096
# modeling
# vocab & head
tokenizer_model="/mnt/m2/cong.fu/models/DeepSeek-R1-bf16"
num_layers=16
hidden_size=896
ffn_hidden_size=2304
num_attention_heads=128
# MLA
router_topk=4
num_experts=256
moe_intermediate_size=1024
num_shared_experts=1
q_lora_rank=1536
kv_lora_rank=512
v_head_dim=128
qk_nope_head_dim=128
qk_rope_head_dim=64
rope_theta=10000
# MoE
moe_layer_freq="([0]*1+[1]*15)"

# Train
train_iters=5000
lr=1e-4
lr_warmup_fraction=0.05
min_lr=1e-5
mbs=1
gbs=16

# Load Save
run_name="dev-deepseekv3-TP${tp}-CP${cp}-EP${ep}-PP${pp}-L${num_layers}-H${hidden_size}-NE${num_experts}"
save="/mnt/m2/cong.fu/models/megatron/${timestamp}-${run_name}"
save_interval=1000

################################################################
# torchrun distributed args start
distributed_args=(
    --nproc_per_node ${gpus_per_node}
    --nnodes ${world_size}
    --node_rank ${node_rank}
    --master_addr ${master_addr}
    --master_port ${master_port}
)
# torchrun distributed args end
################################################################
# data args start
data_arg=(
    --seq-length ${seq_length}
    --train-iters ${train_iters}
    --data-path ${data_path}
    --split "90,5,5"
)
# --mock-data
# data args end
################################################################
# Parallel args start
model_parallel_args=(
	--tensor-model-parallel-size ${tp}
	--pipeline-model-parallel-size ${pp}
    --context-parallel-size ${cp}
    --expert-model-parallel-size ${ep}
    --expert-tensor-parallel-size ${eptp}
    --sequence-parallel
)
# Parallel args end
################################################################
# Train args start
training_args=(
    --lr ${lr}
    --lr-warmup-fraction ${lr_warmup_fraction}
    --min-lr ${min_lr}
    --lr-decay-style cosine
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --micro-batch-size ${mbs}
    --global-batch-size ${gbs}
    --recompute-activations
    --recompute-granularity selective
    --use-distributed-optimizer
    --bf16
)
# Train args end
################################################################
# Model args start
modeling_args=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${tokenizer_model}
    --untie-embeddings-and-output-weights
    --transformer-impl transformer_engine
    --num-layers ${num_layers}
    --hidden-size ${hidden_size}
    --num-attention-heads ${num_attention_heads}
    --ffn-hidden-size ${ffn_hidden_size}
    --swiglu
    --position-embedding-type 'rope'
    --rotary-base ${rope_theta}
    --multi-latent-attention
    --q-lora-rank ${q_lora_rank}
    --kv-lora-rank ${kv_lora_rank}
    --qk-head-dim ${qk_nope_head_dim}
    --qk-pos-emb-head-dim ${qk_rope_head_dim}   #TOOD: q_head_dim = qk_head_dim + qk_pos_emb_head_dim 检查 --qk-nope-head-dim in hf
    --v-head-dim ${v_head_dim}
    --kv-channels ${v_head_dim}
    --qk-layernorm
    --normalization RMSNorm
    --disable-bias-linear
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --max-position-embeddings ${seq_length}
    --no-bias-swiglu-fusion
    --no-rope-fusion
)
# --attention-backend flash TODO: MLA理论上可以使用fa
moe_args=(
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall    # TODO: 需要调查
    --moe-router-topk ${router_topk}
    --num-experts ${num_experts}
    --moe-ffn-hidden-size ${moe_intermediate_size}
    --moe-router-load-balancing-type aux_loss   # TODO: 需要调查
    --moe-aux-loss-coeff 0.001
    --moe-layer-freq ${moe_layer_freq}
    --moe-shared-expert-intermediate-size $((${moe_intermediate_size} * ${num_shared_experts}))
    --moe-shared-expert-overlap # why not?
)
# Model args end
################################################################
# Save args start
save_load_args=(
    --save ${save}
    --save-interval ${save_interval}
)
# Save args end
################################################################
# Log args start
logging_args=(
    --log-interval 1
    --log-timers-to-tensorboard
    --tensorboard-dir ${save}/tensorboard
    --tensorboard-queue-size 1
    --wandb-project "deepseek-v3"
    --wandb-exp-name ${timestamp}-${run_name}
)

# Log args end
################################################################
export CUDA_DEVICE_MAX_CONNECTIONS=1    # 确保cuda stream调度和命令分发顺序一致（否则可能影响comm/cal overlap）
export PYTHONPATH='/mnt/m2/cong.fu/code/.Megatron-LM'
${env_prefix}/bin/torchrun ${distributed_args[@]} /mnt/m2/cong.fu/code/.Megatron-LM/patch/pretrain_deepseek_v3.py \
${model_parallel_args[@]} \
${data_arg[@]} \
${training_args[@]} \
${modeling_args[@]} \
${moe_args[@]} \
${logging_args[@]}  \
${save_load_args[@]}
