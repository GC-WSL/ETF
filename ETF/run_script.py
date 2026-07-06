import subprocess
import os
import argparse
import glob
import json
from collections import defaultdict
import time

def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.wait()

def find_latest_weight(weight_dir):

    pattern = os.path.join(weight_dir, "iter_*?00-*.pth")
    weights = glob.glob(pattern)
    if not weights:
        return None

    weights.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return weights[0]

def find_latest_log_dir(base_dir):

    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return None

    subdirs.sort(key=lambda x: os.path.getctime(os.path.join(base_dir, x)), reverse=True)
    return os.path.join(base_dir, subdirs[0])

def collect_json_files(experiment_dirs):

    json_files = []
    for dir_path in experiment_dirs:
        json_file = os.path.join(dir_path, os.path.basename(dir_path) + ".json")
        if os.path.exists(json_file):
            json_files.append(json_file)
    return json_files

def calculate_average_metrics(json_files):

    metrics_sum = defaultdict(float)
    metrics_count = defaultdict(int)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        metrics_sum[key] += value
                        metrics_count[key] += 1
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    return {k: metrics_sum[k]/metrics_count[k] for k in metrics_sum}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test', action='store_true', help='Just test mode by using latest pth')
    parser.add_argument('--times', type=int, choices=range(2,10), nargs='?', const=3,
                       help='times for experiments(2-9),default 3')

    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs per node')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Rank of this node')
    parser.add_argument('--master_addr', type=str, default='127.0.0.3', help='Master IP address')
    parser.add_argument('--master_port', type=int, default=29600, help='Master port')

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ['PYTHONPATH'] = f"{base_dir}:{os.environ.get('PYTHONPATH', '')}"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # 

    # filename = 'vhg_0_sim-msca+cls-erodil-s-mask_test-k11_2k_ome'
    # filename = 'psdm_sim-msca+cls-erodil-s-mask_test-k19_3k_ome'
    filename = 'isaid2x_sim-msca+cls-erodil-s-mask_test-k7_10k_ctfa'
    launcher = 'pytorch'  # or 'none' / 'torchrun'
    train_config = f"configs/{filename}.py"
    test_config = f"configs/{filename}.py"
    work_dir = os.path.join(base_dir, f"work_dirs/{filename}")

    experiment_dirs = []
    json_files = []


    if launcher == 'pytorch':
        extral_cmd = (
            f"python -m torch.distributed.launch "
            f"--nnodes={args.nnodes} "
            f"--node_rank={args.node_rank} "
            f"--master_addr={args.master_addr} "
            f"--nproc_per_node={args.gpus} "
            f"--master_port={args.master_port}"
        )
    elif launcher == 'torchrun':
        extral_cmd = (
            f"torchrun "
            f"--nnodes={args.nnodes} "
            f"--node_rank={args.node_rank} "
            f"--rdzv_id=123456 "
            f"--rdzv_backend=c10d "
            f"--rdzv_endpoint={args.master_addr}:{args.master_port} "
            f"--nproc_per_node={args.gpus}"
        )
    else:
        extral_cmd = "python"

    if args.test:

        latest_weight = find_latest_weight(work_dir)
        if latest_weight:
            test_cmd = f"python {base_dir}/tools/test.py {test_config} {latest_weight}"
            print("Starting the test...")
            run_command(test_cmd)
            if log_dir := find_latest_log_dir(work_dir):
                experiment_dirs.append(log_dir)
        else:
            print("No available weight file was found.")
    elif args.times:

        for i in range(1, args.times+1):
            print(f"\n=== The {i}th experiment ===")

            train_cmd = f"{extral_cmd} {base_dir}/tools/train.py {train_config} --launcher {launcher} "
            run_command(train_cmd)

            if latest_weight := find_latest_weight(work_dir):
                test_cmd = f"python {base_dir}/tools/test.py {test_config} {latest_weight}"
                run_command(test_cmd)

                if log_dir := find_latest_log_dir(work_dir):
                    experiment_dirs.append(log_dir)
            else:
                print(f"The {i}th experiment failed to generate the weight file.")
            time.sleep(1)
        
        json_files = collect_json_files(experiment_dirs)
        if json_files:
            avg_metrics = calculate_average_metrics(json_files)
            print("\n=== Average evaluation metrics ===")
            for metric, value in avg_metrics.items():
                print(f"{metric}: {value:.4f}")
        else:
            print("No available JSON result file found")
    else:

        print("Starting training...")
        run_command(f"{extral_cmd} {base_dir}/tools/train.py {train_config} --launcher {launcher} ")
        print("Training complete")