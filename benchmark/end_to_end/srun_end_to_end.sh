#!/bin/bash

CUDA_PATH=/usr/local/cuda-11.8

dir=$(dirname $0)
workdir=$(realpath $dir)

launch_type=$1
model_cfgs=$2
workload=$(realpath $3)
cluster=$4

echo "============================================="
echo "launch_type:              $launch_type"
echo "model_cfgs:               $model_cfgs"
echo "workload:                 $workload"
echo "cluster:                  $cluster"
echo "============================================="

usrname=dmhj
job_name=muxserve
# Image_ID=registry.cn-sh-01.sensecore.cn/cpii-ccr/clouddev-snapshot:20240102-14h01m22s
Image_ID=registry.cn-sh-01.sensecore.cn/cpii-ccr/clouddev-snapshot:20240325-13h12m55s


model_cfgs=$(find $model_cfgs | tr "\n" ",")
IFS="," read -ra model_cfgs <<< "$model_cfgs"

# cluster1 afb99c73-b2be-428d-963c-352460ab84cd d43c2524-492e-4df1-be8b-a95e688bd0f7
# cluster2 fa7dc572-ab64-4ad1-b7f2-335823fc8781 d43c2524-492e-4df1-be8b-a95e688bd0f7

if [ "$cluster" -eq 1 ]; then
  partition_id=afb99c73-b2be-428d-963c-352460ab84cd
  workspace_id=d43c2524-492e-4df1-be8b-a95e688bd0f7
elif [ "$cluster" -eq 2 ]; then
  partition_id=fa7dc572-ab64-4ad1-b7f2-335823fc8781
  workspace_id=d43c2524-492e-4df1-be8b-a95e688bd0f7
else
  echo "Invalid cluster value: $cluster"
  exit 1
fi


export YOUR_PASSWD="dmhj@123"

if [ $launch_type = "spatial" ]; then
    spatial_config=$(realpath $model_cfgs)

    # 读取文件的第一行
    first_line=$(head -n 1 "$spatial_config")
    # 去除行首的 "# " 部分
    ngpu_per_llm=${first_line#*# }
    # 分割数字并迭代每个数字
    IFS='_' read -ra ngpu_arr <<< "$ngpu_per_llm"

    alpha=$(echo "$spatial_config" | sed -n 's/.*alpha\([0-9.]*\).*/\1/p')
    scale=$(echo "$spatial_config" | sed -n 's/.*scale\([0-9.]*\).*/\1/p')

    llm_id=0
    for mesh_size in "${ngpu_arr[@]}"; do
        spatial_log="$end_to_end_log_dir/spatial/alpha${alpha}_${scale}_${ts}/llm_$llm_id.log"

        # echo -e "\033[31mcommand:\033[0m\n\t\033[35mbash\033[0m $start_script $spatial_config $workloads $llm_id $ngpus $spatial_log $proc_id $cluster"

        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($mesh_size-1)))

        NPROC_PER_NODE=$mesh_size
        resource=N2lS.Ie.I60.${NPROC_PER_NODE}

        # VLLM_PROC_LOG=$VLLM_PROC_LOG bash $start_script $spatial_config $workloads $llm_id $ngpus $spatial_log $proc_id $cluster
        srun -p $partition_id \
            --workspace-id $workspace_id \
            --async -o log/srun \
            -N 1 -r $resource -j $job_name --framework pytorch -d StandAlone \
            --container-image ${Image_ID} \
            --container-mounts=e70f5aef-dd05-11ed-9103-ba18b4912d57:/mnt/afs \
            bash -c "rm -rf /usr/local/nvidia/lib64/*; \
            su - ${usrname} -c \" \
                cd $workdir && \
                export CUDA_PATH=${CUDA_PATH} && \
                export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${CUDA_PATH}/extras/CUPTI/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH && \
                export PATH=/mnt/afs/dmhj/miniconda3/envs/muxserve/bin:/mnt/afs/dmhj/miniconda3/condabin:$PATH && \
                export PATH=${CUDA_PATH}/bin:$PATH && \
                export PYTHONPATH=${workspace}:${PYTHONPATH} && \
                bash run_end_to_end.sh $launch_type $CUDA_VISIBLE_DEVICES $spatial_config $workload llm-$llm_id && \
                sleep 180 \""

        (( llm_id++ ))
    done
elif [ $launch_type = "muxserve" ]; then
    for config in "${model_cfgs[@]}"; do
        model_cfg=$(realpath $config)
        mesh_size=$(echo "$model_cfg" | sed 's/.*mesh_size\([0-9]\+\).*/\1/')
        NPROC_PER_NODE=$mesh_size
        resource=N2lS.Ie.I60.${NPROC_PER_NODE}
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($mesh_size-1)))

        # echo -e "\033[31mcommand:\033[0m\n  \033[35mbash\033[0m run_end_to_end.sh $launch_type $CUDA_VISIBLE_DEVICES $model_cfg $workload"

        srun -p $partition_id \
            --priority highest \
            --workspace-id $workspace_id \
            --async -o log/srun \
            -N 1 -r $resource -j $job_name --framework pytorch -d StandAlone \
            --container-image ${Image_ID}  \
            --container-mounts=e70f5aef-dd05-11ed-9103-ba18b4912d57:/mnt/afs \
            bash -c "rm -rf /usr/local/nvidia/lib64/*; \
            su - ${usrname} -c \" \
                cd $workdir && \
                export CUDA_PATH=${CUDA_PATH} && \
                export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${CUDA_PATH}/extras/CUPTI/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH && \
                export PATH=/mnt/afs/dmhj/miniconda3/envs/muxserve/bin:/mnt/afs/dmhj/miniconda3/condabin:$PATH && \
                export PATH=${CUDA_PATH}/bin:$PATH && \
                export PYTHONPATH=${workspace}:${PYTHONPATH} && \
                YOUR_PASSWD=$YOUR_PASSWD bash run_end_to_end.sh $launch_type $CUDA_VISIBLE_DEVICES $model_cfg $workload && \
                sleep 180 \""
    done
elif [ $launch_type = "temporal" ]; then
    for config in "${model_cfgs[@]}"; do
        model_cfg=$(realpath $config)
        mesh_size=$(echo "$model_cfg" | sed 's/.*mesh_size\([0-9]\+\).*/\1/')
        NPROC_PER_NODE=$mesh_size
        resource=N2lS.Ie.I60.${NPROC_PER_NODE}
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($mesh_size-1)))

        # echo -e "\033[31mcommand:\033[0m\n  \033[35mbash\033[0m run_end_to_end.sh $launch_type $CUDA_VISIBLE_DEVICES $model_cfg $workload"

        srun -p $partition_id \
            --workspace-id $workspace_id \
            --async -o log/srun \
            -N 1 -r $resource -j $job_name --framework pytorch -d StandAlone \
            --container-image ${Image_ID}  \
            --container-mounts=e70f5aef-dd05-11ed-9103-ba18b4912d57:/mnt/afs \
            bash -c "rm -rf /usr/local/nvidia/lib64/*; \
            su - ${usrname} -c \" \
                cd $workdir && \
                export CUDA_PATH=${CUDA_PATH} && \
                export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${CUDA_PATH}/extras/CUPTI/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH && \
                export PATH=/mnt/afs/dmhj/miniconda3/envs/muxserve/bin:/mnt/afs/dmhj/miniconda3/condabin:$PATH && \
                export PATH=${CUDA_PATH}/bin:$PATH && \
                export PYTHONPATH=${workspace}:${PYTHONPATH} && \
                bash run_end_to_end.sh $launch_type $CUDA_VISIBLE_DEVICES $model_cfg $workload && \
                sleep 180 \""
    done
else
    echo "Launch type $launch_type invalid"
    exit 1
fi
