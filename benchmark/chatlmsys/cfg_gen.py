from muxserve.muxsched.placement import PlacementOptimizer
import yaml
import os

COST_FILE = "/mnt/afs/lurunyu/projects/MuxServe/examples/placement/llama.json"


def get_placement_from_cfg(
        models_yaml: str,
        costfile: str,
        is_greedy=False,
        dump_to_yaml=True,
        dump_dir: str = None,
        verbose: bool = False,
        avg_output_len=337,  # sharegpt data
        avg_prompt_len=161  # sharegpt data
):

    opt = PlacementOptimizer(models_yaml, costfile)

    return opt.optimize(is_greedy,
                        dump_dir=dump_dir,
                        dump_to_yaml=dump_to_yaml,
                        verbose=verbose,
                        avg_output_len=avg_output_len,
                        avg_prompt_len=avg_prompt_len)


if __name__ == "__main__":
    import glob

    to_scan = 'benchmark/chatlmsys/yamls/'
    # files = glob.glob(to_scan + 'requests_over_time_models_days_from_day100_to_day105*.yaml')
    # files = glob.glob(to_scan + '*.yaml')
    files = glob.glob(
        to_scan + 'requests_over_time_models_days_from_day55_to_day55*.yaml')
    # to_scan + 'requests_over_time_models_days_from_day60_to_day65*.yaml')

    dump_dir = 'benchmark/chatlmsys/yamls/muxserve'

    for file in files:

        with open(file, 'r') as f:
            yml = yaml.safe_load(f)
        avg_output_len = yml['avg_output_len']
        avg_prompt_len = yml['avg_prompt_len']

        get_placement_from_cfg(
            file,
            COST_FILE,
            False,
            dump_to_yaml=True,
            dump_dir=dump_dir,
            verbose=True,
            avg_output_len=avg_output_len,  # sharegpt data
            avg_prompt_len=avg_prompt_len  # sharegpt data
        )
