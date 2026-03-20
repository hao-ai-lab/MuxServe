import os
import re
import time
import random
import glob
import shlex
import subprocess

from muxserve.logger import get_logger

logger = get_logger()


def get_all_folders_in_current_directory(current_directory):
    folder_names = [
        folder for folder in os.listdir(current_directory)
        if os.path.isdir(os.path.join(current_directory, folder))
    ]
    return folder_names


def get_all_files_in_current_directory(current_directory):
    folder_names = [folder for folder in os.listdir(current_directory)]
    return folder_names


def find_consecutive_number(free_list, n):
    for i in range(8):
        lis = list(range(i, i + n))
        flag = False
        for l in lis:
            if l not in free_list:
                flag = True
                break
        if flag:
            continue
        else:
            return lis
    return None


def check_and_run_script(directory, filename_pattern, idx):
    processed_files = set()

    free_gpu = list(range(0, 8))
    procs = []

    while True:
        files = os.listdir(directory)
        # logger.info("all files: ", files)
        if idx == 1:
            files = files[::-1]

        for file_name in files:

            while len(free_gpu) == 0:
                for (cmd, proc, num_list, beg_t) in procs:
                    if proc.poll() is not None:
                        free_gpu = free_gpu + num_list
                        free_gpu.sort()
                        procs.remove((cmd, proc, num_list, beg_t))
                        logger.info(f'command finished: {cmd}')
                    else:
                        # logger.info(f'cmd running: {cmd}')
                        now = time.perf_counter()
                        if now - beg_t > 30 * 60:
                            logger.info(f'command timeout, killed: {cmd}')
                            proc.terminate()
                            free_gpu = free_gpu + num_list
                            free_gpu.sort()
                            procs.remove((cmd, proc, num_list, beg_t))

            file_path = os.path.join(directory, file_name)
            match = re.match(filename_pattern, file_name)
            if match and file_path not in processed_files:
                mesh_size = match.group('mesh_size')
                current_idx = int(match.group('idx'))
                ngpus = int(match.group('gpunum'))
                num_models = int(match.group('n'))
                maxrate = match.group('maxrate')
                avgrate = match.group('avgrate')
                condense = match.group('condense')

                workload_file = f"/mnt/afs/lurunyu/data/requests_over_time_models_days_from_day60_to_day65_condense{condense}_N{num_models}_maxrate_{maxrate}_avgrate_{avgrate}.json"

                free_gpu.sort()
                if find_consecutive_number(free_gpu, int(mesh_size)):
                    num_list = find_consecutive_number(free_gpu,
                                                       int(mesh_size))
                    command = f"sudo bash benchmark/chatlmsys/profile.sh 1 {mesh_size} {file_path} /mnt/afs/lurunyu/projects/MuxServe/log/mps{idx} {workload_file} {','.join(list(map(str, num_list)))}"
                else:
                    continue

                all_finished = get_all_files_in_current_directory(
                    '/mnt/afs/lurunyu/data/')
                flag = False

                for finish in all_finished:
                    if file_name[:-5] in finish and workload_file[:-5].split(
                            '/')[-1] in finish:
                        logger.info(f"{file_name} has been finished, skip it")
                        flag = True

                if flag:
                    continue

                logger.info(f'command running: {command}')

                # subprocess.run(command, shell=True, check=True)
                proc = subprocess.Popen(
                    command,
                    shell=True,
                )
                for num in num_list:
                    free_gpu.remove(num)
                procs.append((command, proc, num_list, time.perf_counter()))

                s_t = 15
                logger.info(f'sleep {s_t}s ...')
                time.sleep(s_t)

                processed_files.add(file_path)

        s_t = 30
        logger.info(f'sleep {s_t}s ...')
        time.sleep(s_t)


'''
sudo /home/lurunyu/envs/miniconda3/envs/muxserve/bin/python benchmark/chatlmsys/scan.py --directory benchmark/chatlmsys/yamls/placement_gen --mps-idx 1
'''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=
        "Check directory for matching YAML files and run corresponding commands."
    )
    parser.add_argument(
        "--directory",
        help="Path to the directory to check",
        default="/mnt/afs/lurunyu/projects/MuxServe/examples/placement")
    parser.add_argument("--mps-idx",
                        help="Value for idx parameter",
                        type=int,
                        default=0)
    args = parser.parse_args()

    # Modified regex pattern to include optional idx + 3
    # yaml_filename_pattern = r'(?P<variable_prefix>\w+)_GPUnum(?P<gpunum>\d+)_mesh_size(?P<mesh_size>\d+)_idx(?P<idx>\d+)(|_idx(?P<idx_plus_3>\d+)).yaml'
    # yaml_filename_pattern = r'(?P<variable_prefix>\w+)_mesh_size(?P<mesh_size>\d+)_idx(?P<idx>\d+)(|_idx(?P<idx_plus_3>\d+)).yaml'
    # yaml_filename_pattern = r'(?P<gpu>\d+)GPU_N(?P<n>\d+)_maxrate_(?P<maxrate>\d+(\.\d+)?)_(?P<flag>.*?)_mesh_size(?P<mesh_size>\d+)_idx(?P<idx>\d+)(|_idx(?P<idx_plus_3>\d+)).yaml'
    yaml_filename_pattern = r'requests_over_time_models_days_from_day60_to_day65_condense(?P<condense>\d+)_N(?P<n>\d+)_maxrate_(?P<maxrate>\d+(\.\d+)?)_avgrate_(?P<avgrate>\d+(\.\d+)?)_GPUnum(?P<gpunum>\d+(\.\d+)?)_mesh_size(?P<mesh_size>\d+)_idx(?P<idx>\d+).yaml'

    check_and_run_script(args.directory, yaml_filename_pattern, args.mps_idx)
