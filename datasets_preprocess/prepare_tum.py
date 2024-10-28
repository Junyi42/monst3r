import glob
import os
import shutil
import numpy as np

def read_file_list(filename):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp, data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp, data) tuples
    second_list -- second dictionary of (stamp, data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1, data1), (stamp2, data2))
    """
    # Convert keys to sets for efficient removal
    first_keys = set(first_list.keys())
    second_keys = set(second_list.keys())
    
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches

dirs = glob.glob("../data/tum/*/")
dirs = sorted(dirs)
# extract frames
for dir in dirs:
    frames = []
    gt = []
    first_file = dir + 'rgb.txt'
    second_file = dir + 'groundtruth.txt'

    first_list = read_file_list(first_file)
    second_list = read_file_list(second_file)
    matches = associate(first_list, second_list, 0.0, 0.02)

    # for a,b in matches[:10]:
    #     print("%f %s %f %s"%(a," ".join(first_list[a]),b," ".join(second_list[b])))
    for a,b in matches:
        frames.append(dir + first_list[a][0])
        gt.append([b]+second_list[b])
    
    # sample 90 frames at the stride of 3
    frames = frames[::3][:90]
    # cut frames after 90
    new_dir = dir + 'rgb_90/'

    for frame in frames:
        os.makedirs(new_dir, exist_ok=True)
        shutil.copy(frame, new_dir)
        # print(f'cp {frame} {new_dir}')

    gt_90 = gt[::3][:90]
    with open(dir + 'groundtruth_90.txt', 'w') as f:
        for pose in gt_90:
            f.write(f"{' '.join(map(str, pose))}\n")