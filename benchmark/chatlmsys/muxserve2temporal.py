import yaml
import glob
import copy

if __name__ == "__main__":
    # files = glob.glob("benchmark/chatlmsys/yamls/placement_gen/*.yaml")
    files = glob.glob("benchmark/chatlmsys/yamls/muxserve/*.yaml")

    for file in files:
        with open(file, 'r') as f:
            yml = yaml.safe_load(f)

        temporal = copy.deepcopy(yml)
        for i in range(len(temporal['models'])):
            temporal['models'][i]['mps_percentage'] = [100, 90]

        filename = file.split('/')[-1]
        filename_stem = filename.split('.')[0]

        out_file = f'benchmark/chatlmsys/yamls/temporal/{filename_stem}_temporal.yaml'

        with open(out_file, 'w') as f:
            yaml.dump(temporal, f, sort_keys=False)
