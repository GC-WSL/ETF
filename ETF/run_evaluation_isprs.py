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

datasets_dict={'isaid':'iSAID_512_sampled_2',
               'psdm':'Potsdam_pd',
               'vhg':'Vaihingen_256'}

def get_datasets_name(filename):
    if 'isaid' in filename:
        return datasets_dict['isaid']
    elif 'psdm' in filename:
        return datasets_dict['psdm']
    elif 'vhg' in filename:
        return datasets_dict['vhg']
    else :
        raise ValueError('Filename error!')

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ['PYTHONPATH'] = f"{base_dir}:{os.environ.get('PYTHONPATH', '')}"
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--stage',choices=['all','mid','post'], default='all',
                        help='The evaluation stage,all: generate prob-np and then use postprocessor to eval'\
                             'mid:just generate prob-np by mmseg-test mode'\
                             'post:just use exiting prob-np to get postprocess-result')
    args = parser.parse_args()

    filename = 'psdm_sim-msca+cls-erodil-s-mask_test-k19_3k_ome'

    test_config = f"configs/{filename}.py"
    work_dir = os.path.join(base_dir, f"work_dirs/{filename}")

    latest_weight = None # default None: find the latest weights automatically

    data_name = get_datasets_name(filename)

    print("work dir:",work_dir)
    if latest_weight is None:
        latest_weight = find_latest_weight(work_dir)

    if latest_weight and (args.stage in ['all','mid']):
            test_cmd = (f"python {base_dir}/tools/test.py {test_config} {latest_weight} "
                        "--save_infer ")
            print("Starting the test... Generating the prob np file...")
            run_command(test_cmd)
    else:
        print("No available weight file was found.")
    model = filename.split('_')[-1]
    eval_command = (
        f"python {base_dir}/tools/eval_crf.py "
        f"--root ../datasets/{data_name} "
        f"--predict-dir work_dirs/{filename}/prob_npy "
        "--reduce-zero " 
        "--num-cls 5 "
        # f"--save-heatmap --heatmap-dir work_dirs/{filename}/heatmap "
        # f"--save-color --color-dir work_dirs/{filename}/colormask_crf "
        # f"--save-mask --mask-dir ../datasets/{data_name}/crf_pseudo_{model}_simclip "
        "--crf "
        "--split val "
        # "--sub 1000 " # for vis
        # "--save-gt "
    )
    # reduce-zero  for Potsdam to ignore gt's zero-class

    if args.stage in ['all','post']:
        # Run the eval command
        print("Starting myCLIP evaluation...")
        run_command(eval_command)
        print("Evaluation completed.")