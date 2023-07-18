import os 
import numpy as np
import shutil
import matplotlib.pyplot as plt

def load_data(data_dir):
    data = []
    for data_file in os.listdir(data_dir):
        if data_file.endswith(".npy") and not data_file.startswith("poses")\
            and not data_file.startswith("variance")\
                and not data_file.startswith("checkpoint"):
            data.append(np.load(data_dir + "/" + data_file))
    return np.concatenate(data)

def compute_bin_idx(y, accuracy_bins):
    bins = []
    for i in range(len(accuracy_bins))[0:-2]:
        bins.append((y >= accuracy_bins[i]) & (y < accuracy_bins[i+1]))
    bins.append((y >= accuracy_bins[-2]) & (y <= accuracy_bins[-1]))
    return bins

def balance(data0, data1, bins0, bins1):
    min_max0 = np.min([len(data0[bin]) for bin in bins0])
    min_max1 = np.min([len(data1[bin]) for bin in bins1])
    min_max = np.min([min_max0, min_max1])
    print("min_max: ", min_max)
    data0_equal = np.concatenate([data0[bin][:min_max] for bin in bins0])
    data1_equal = np.concatenate([data1[bin][:min_max] for bin in bins1])
    return data0_equal, data1_equal

# def balance(data, bins):
#     min_max = np.min([len(data[bin]) for bin in bins])
#     return np.concatenate([data[bin][:min_max] for bin in bins])


accuracy_bins = np.array([0, 0.001, 0.01, 0.1, 1])

# data_dir_tiago = "./tiago/"
# data_tiago = load_data(data_dir_tiago)
# data_tiago = np.load('tiago.npy')
# bins_tiago = compute_bin_idx(data_tiago[:,2], accuracy_bins)

data_dir_vehicle = "./vehicle/"
data_vehicle = load_data(data_dir_vehicle)
# data_vehicle = np.load('vehicle.npy')
# bins_vehicle = compute_bin_idx(data_vehicle[:,2], accuracy_bins)
# for bin in bins_vehicle:
#     print(np.count_nonzero(bin))
plt.hist(data_vehicle[:,2], accuracy_bins)
plt.savefig('hist.svg')
# tiago_balanced, vehicle_balanced = balance(data_tiago, data_vehicle, bins_tiago, bins_vehicle)
# print(tiago_balanced.shape)
# print(vehicle_balanced.shape)

# np.random.shuffle(tiago_balanced)
# vehicle_balanced = balance(data_vehicle, bins_vehicle)
# np.random.shuffle(vehicle_balanced)

# np.save("./" + "tiago_balanced.npy", tiago_balanced)
# np.save("./" + "vehicle_balanced.npy", vehicle_balanced)
