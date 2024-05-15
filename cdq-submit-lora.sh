sleep 1s
echo sleep test sucessful

sleep 30m

sbatch --gres=gpu:3 --comment=efficientllm cdq-jobscript-2.slurm 

sleep 10m

sbatch --gres=gpu:3 --comment=efficientllm cdq-jobscript-3.slurm 

sleep 30m

sbatch --gres=gpu:3 --comment=efficientllm cdq-jobscript-4.slurm 