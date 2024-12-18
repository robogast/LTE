# Code implementation for LTE

Due to size limitation, we do not include MMLU data.
Download MMLU dataset from this repo [MMLU](https://github.com/artidoro/qlora/tree/main).

Example script for reproduce llama2 LTE with Tulu finetuning.

```bash
task='tulu-llama2'

datamodel_dir=your-path
datadata_dir=your-path
datalog_dir=your-path

#save llama files and do experts grouping
python save_llama.py --model_name meta-llama/Llama-2-7b-hf --output_dir $datamodel_dir/llama2-7b/

python kmeans-grouping.py --model_path $datamodel_dir/llama2-7b/last-ckpt.pt --saving_path $datadata_dir/kmeans-group/llama2-7b-kmeans-grouping.pt --num-layer 32 --num-expert 344 --model_type llama

eta=0.1

soft_task=soft-$eta-llama7b
mkdir -p $datalog_dir

# LTE stage 1
torchrun --nnodes 1 --nproc_per_node 8 llama-instruction-tuning.py \
    --dataset tulu \
    --model_name meta-llama/Llama-2-7b-hf \
    --output_dir $datamodel_dir/$soft_task \
    --dist_checkpoint_root_folder $datamodel_dir/tmp \
    --enable_fsdp  --fsdp_config.pure_bf16 \
    --batch_size_training 2 --gradient_accumulation_steps 8 \
    --lr_scheduler step \
    --run_validation False \
    --num_epochs 1 --log_step 50 \
    --lte --moe_type block --moe_experts 344 \
    --moe_routing_mode sigmoid --moe_eta $eta \
    --kmean_grouping --kmean_grouping_path $datadata_dir/kmeans-group/llama2-7b-kmeans-grouping.pt \
    --use_pretrained

# LTE stage 2
hard_task=hard-$eta-llama7b

torchrun --nnodes 1 --nproc_per_node 8 llama-instruction-tuning.py \
    --dataset tulu \
    --model_name meta-llama/Llama-2-7b-hf \
    --output_dir $datamodel_dir/$hard_task \
    --dist_checkpoint_root_folder $datamodel_dir/tmp \
    --enable_fsdp  --fsdp_config.pure_bf16 \
    --batch_size_training 2 --gradient_accumulation_steps 8 \
    --lr_scheduler step \
    --run_validation False \
    --num_epochs 4 --log_step 50 \
    --lte --moe_type block --moe_experts 344 \
    --moe_routing_mode sigmoid --hard \
    --kmean_grouping --kmean_grouping_path $datadata_dir/kmeans-group/llama2-7b-kmeans-grouping.pt \
    --ckpt_path $datamodel_dir/$soft_task/last-ckpt.pt


#MMLU evaluation
python mmlu_llama.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --lte --moe_type block --moe_experts 344 \
    --moe_routing_mode sigmoid --hard \
    --ckpt_path $datamodel_dir/$hard_task/last-ckpt.pt \
    --kmean_grouping --kmean_grouping_path $datadata_dir/kmeans-group/llama2-7b-kmeans-grouping.pt
```


Example script for reproduce llama2 LTE with Wiki finetuning.
```bash
datamodel_dir=your-path
datadata_dir=your-path
datalog_dir=your-path
model_name=meta-llama/Llama-2-7b-hf

eta=0.1

#finetune llama on wiki
torchrun --nnodes 1 --nproc_per_node 8 finetuning.py --enable_fsdp \
    --dataset wiki_dataset \
    --model_name $model_name --fsdp_config.pure_bf16 \
    --output_dir $datamodel_dir/wiki-llama \
    --batch_size_training 4 --gradient_accumulation_steps 2 \
    --dist_checkpoint_root_folder $datamodel_dir/tmp \
    --lr_scheduler cosine \
    --num_epochs 3

python kmeans-grouping.py --model_path your_wiki_ckpt_path --saving_path $datadata_dir/kmeans-group/llama-wiki-kmeans-grouping.pt --num-layer 32 --num-expert 344 --model_type llama

torchrun --nnodes 1 --nproc_per_node 8 finetuning.py \
    --output_dir $datamodel_dir/llama-wiki-lte/lte-soft/llama-wiki-lte-eta$eta \
    --dist_checkpoint_root_folder $datamodel_dir/tmp \
    --enable_fsdp --model_name $model_name \
    --num_epochs 1 --dataset wiki_dataset --log_step 500 \
    --fsdp_config.pure_bf16 --batch_size_training 2 --gradient_accumulation_steps 1 \
    --lte --moe_type block --moe_experts 344 \
    --moe_routing_mode sigmoid --moe_eta $eta \
    --kmean_grouping --kmean_grouping_path $datadata_dir/kmeans-group/llama-wiki-kmeans-grouping.pt \
    --ckpt_path $datamodel_dir/llama-wiki/last-ckpt.pt

torchrun --nnodes 1 --nproc_per_node 8 finetuning.py \
    --output_dir $datamodel_dir/llama-wiki-lte/lte-hard/llama-wiki-lte-eta$eta \
    --dist_checkpoint_root_folder $datamodel_dir/tmp \
    --enable_fsdp --model_name $model_name \
    --num_epochs 1 --dataset wiki_dataset --log_step 500 \
    --fsdp_config.pure_bf16 --batch_size_training 2 --gradient_accumulation_steps 1 \
    --lte  --moe_type block --moe_experts 344 \
    --moe_routing_mode sigmoid --hard \
    --kmean_grouping --kmean_grouping_path $datadata_dir/kmeans-group/llama-wiki-kmeans-grouping.pt \
    --ckpt_path $datamodel_dir/llama-wiki-lte/lte-soft/llama-wiki-lte-eta$eta/last-ckpt.pt
```