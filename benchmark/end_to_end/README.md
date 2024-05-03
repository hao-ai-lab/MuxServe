1. Generate configration:

    First, **replace the model path and dataset path** specified in `bench_end_to_end_muxserve.py` with your own path. Specifically, modify the following variables according to the comments.

    - `MODEL_TO_PATH`
    - `SHAREGPT_PATH`
    - `TOKENIZED_DATA_CACHE`

    Run the scripts to generate configuration:
    ```bash
    python bench_end_to_end_muxserve.py
    ```
    This will generate the configuration and workloads file for the corresponding end-to-end evaluation in the paper: `alpha` = 0.7, 0.9, 1.3, 1.7, 2.1.

2. To start the experiment with running the `run_end_to_end.sh` script. Execute the following command in your terminal:

    ```bash
    bash run_end_to_end.sh <launch_type> <cuda_device> <yaml> <workload> [split_llm if 'spatial']
    ```

    - `launch_type` is choosen from [`muxserve`, `spatial`, `temporal`]
    - **Note:** `llm-id` is needed if `launch_type` is temporal; which is in the config file
    - **Note:** Flexsm utilizes Nvidia MPS. Running the muxserve component in the experiment requires **root** privileges. Replace the password in the script with your password(which is marked as `YOUR_PASSWD` in the `run_end_to_end.sh`).

    An example:

    ```bash
    bash run_end_to_end.sh spatial 0 \
    model_cfgs/alpha0.7_scale0.5_max40/spatial_cfg.yaml \
    workloads/alpha0.7_scale0.5_max40/sharegpt_n19_req.json 2
    ```

    Make sure you are in the correct directory where the `run_end_to_end.sh` script is located. This script will initiate the necessary steps to run the end-to-end experiment.

    Once the test is stared, run logs will be generated in `${PROJ_DIR}/benchmark/end_to_end/log` by default.

3. Extract the evaluation result from log file:

    We provide an automated script `plot_p_latency.py` that performs statistical analysis on evaluation results and visualizes them.
