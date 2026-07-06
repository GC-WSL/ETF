import argparse
import os
from PIL import Image
import numpy as np
import tqdm
import glob
import cv2


def run(args):
    dir1= ['img_dir','ann_dir']

    dir2= [f'test{args.idname}'] if args.with_test else  ['train','val']
    for d1 in dir1:
        for d2 in dir2:
            file_list = glob.glob(os.path.join(args.data_root,d1,d2,'*.png'))
            if 'ann' in d1:
                label(file_list,os.path.join(args.out_path,d2+args.suffix),args.n_class,full=args.full)
    return

def label(file_list,out_path,n_class,t=0.1,full=True):
    with open(out_path,'w') as f:
        with tqdm.tqdm(range(len(file_list))) as bar:
            for each in file_list:
                
                img = Image.open(each)
                img_label,counts = np.unique(np.asarray(img),return_counts=True)

                img_label = img_label[(img_label>0) & (img_label<n_class)] if counts[0]/counts.sum()<=t else img_label[img_label<n_class]

                str_arr = ' '.join(np.array2string(img_label, separator=',')[1:-1].split(','))
                name = each.split(os.path.sep)[-1].replace('.png',' ')

                if '_instance_color_RGB' in name:
                    name = name.replace('_instance_color_RGB','')
                if full:
                    f.write(name+str_arr+'\n')
                else:
                    f.write(name+'\n')
                bar.update()
    return 


def mean_bgr(args):
    dir1= ['img_dir']
    dir2= ['train','val']
    channel_means = np.zeros(3)
    count = 0
    for d1 in dir1:
        for d2 in dir2:
            file_list = glob.glob(os.path.join(args.data_root,d1,d2,'*.png'))
            if not file_list:
                print(f"No .png files found in {os.path.join(args.data_root, d1, d2)}")
                continue

            for img_path in tqdm.tqdm(file_list, desc=f"Processing {d2}"):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
                if img is not None:
                    channel_means += img.mean(axis=(0, 1)) 
                    count += 1

    if count > 0:
        mean_bgr = channel_means / count  
        print(f"Mean BGR: {mean_bgr}")
    else:
        print("No images were processed.")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root',type=str,default='./iSAID_512_sampled_2')
    parser.add_argument('--out_path',type=str,default='./iSAID_512_sampled_2')
    parser.add_argument('--full',default=False,action='store_true',help='Full means the label is included whether or not')
    parser.add_argument('--suffix',default='.txt',type=str)
    parser.add_argument('--n_class',type=int,default=16)
    args = parser.parse_args()
    if not args.full:
        args.suffix = '_id' + args.suffix

    args.with_test = True
    args.idname = 9
    run(args)
    # mean_bgr(args)