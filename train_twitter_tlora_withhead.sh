max_epochs=50
lora_type=trunk
batch_size=128

for i in $(seq 27 -1 1); do
    python -u train_twitter_tlora_withhead.py \
    --lora_checkpoint_dir .checkpoints/lora/twitter/with_head/$lora_type/e$max_epochs/$i \
    --lora_type $lora_type \
    --max_epochs $max_epochs \
    --batch_size $batch_size
done