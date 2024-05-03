import yaml
import glob
import copy

MAP_MESH = {
    '7b': (1, 0.65),
    '13b': (2, 0.5),
    '30b': (2, 0.375),
    '65b': (4, 0.375),
}

if __name__ == "__main__":
    files = glob.glob("benchmark/chatlmsys/yamls/*.yaml")

    for file in files:
        if 'day55_to_day55' not in file:
            continue

        with open(file, 'r') as f:
            yml = yaml.safe_load(f)

        total_gpu_num = 0
        for idx, instance in enumerate(yml['models']):
            model_size = instance['model'].split('-')[-1]
            name = instance['name']
            model = instance['model']
            mesh_size, util = MAP_MESH[model_size]

            if name == 'llm-1' or name == 'llm-3':
                mesh_size = 1

            filename = file.split('/')[-1]
            filename_stem = filename.split('.')[0]

            out_file = f'benchmark/chatlmsys/yamls/spatial/{filename_stem}_GPUnum32_mesh_size{mesh_size}_idx{idx}_spatial.yaml'

            out_data = {
                "num_gpus":
                mesh_size,
                "max_num_seqs":
                256,
                "overload_threshold":
                2,
                "gpu_memory_utilization":
                util,
                "models": [{
                    "name": name,
                    "model": model,
                    "tensor_parallel_size": mesh_size,
                    "pipeline_parallel_size": 1,
                    "placement": [list(range(mesh_size))],
                    "mps_percentage": [100, 90],
                    "max_num_seqs": 256,
                    "model_dtype": "fp16"
                }],
                "workloads": {
                    "workload_file": None
                }
            }

            total_gpu_num += mesh_size

            with open(out_file, "w") as f:
                yaml.dump(out_data, f, sort_keys=False)
            print(out_file)
        print(f"total gpu: {total_gpu_num}")
