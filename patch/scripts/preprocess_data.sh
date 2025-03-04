export PYTHONPATH="./"

/mnt/afs/miniconda3/envs/fc-base/bin/python ./patch/tools/preprocess_data.py \
--input /mnt/m2/data/EleutherAI_ThePile_v1/pile/test.jsonl  \
--json-keys "text" \
--output-prefix ./local/data/test  \
--workers 16 \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model /mnt/m2/cong.fu/models/DeepSeek-R1-bf16 \
--append-eod \
--eod-token-id 1
