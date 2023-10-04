#include <stdio.h>
#include <string>
#include <iostream> 
#include <fstream>
#include <cmath>
#include <math.h>
#include <random>
#include <chrono>
#include<ctime>

#include <npy.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>
#include <curand.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstddef>

#include <algorithm>    
#include "utils.cu"

#include <boost/program_options.hpp>

typedef thrust::tuple<DeviceFloatPairIterator, DeviceFloatIterator, DeviceFloatIterator, DeviceFloatIterator, DeviceIntIterator> DeviceIteratorTuple;
typedef thrust::zip_iterator<DeviceIteratorTuple> DeviceZipIterator;

namespace po = boost::program_options;

struct Arguments {
    std::string data_dir = "./data/";
    std::string data_file_in = "";
    std::string data_file_out = "";
    int max_samples = 4000000;
    float robot_width = 4.07;
    float robot_height = 1.74;
    bool shuffle = true;
    bool cps_only = false;
    std::string meta_dir = "";
};

Arguments parse_args(int argc, char** argv) {
    Arguments a;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("data_dir", po::value<std::string>(), "where to read the data")
        ("data_file_in", po::value<std::string>(), "where to read the data")
        ("data_file_out", po::value<std::string>(), "where to write the data")
        ("max_samples", po::value<int>(), "maximum number of samples for z-test")
        ("robot_width,w", po::value<float>(), "robot width")
        ("robot_height,h", po::value<float>(), "robot height")
        ("shuffle", po::value<bool>(),"whether or not to shuffle data")
        ("cps_only", po::value<bool>(),"whether or not to only compute collision probabilities")
        ("meta_dir", po::value<std::string>(), "path to meta folder containing accuracy_bins.npy and bin_accuracy.npy")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(1);
    }
    if(vm.count("data_dir")) {
        a.data_dir = vm["data_dir"].as<std::string>();
    }
    if(vm.count("data_file_in")) {
        a.data_file_in = vm["data_file_in"].as<std::string>();
    }
    if(vm.count("data_file_out")) {
        a.data_file_out = vm["data_file_out"].as<std::string>();
    }
    if(vm.count("max_samples")) {
        a.max_samples = vm["max_samples"].as<int>();
    }
    if(vm.count("robot_width")) {
        a.robot_width = vm["robot_width"].as<float>();
    }
    if(vm.count("robot_height")) {
        a.robot_height = vm["robot_height"].as<float>();
    }
    if(vm.count("shuffle")) {
        a.shuffle = vm["shuffle"].as<bool>();
    }
    if(vm.count("cps_only")) {
        a.cps_only = vm["cps_only"].as<bool>();
    }
    if(vm.count("meta_dir")) {
        a.meta_dir = vm["meta_dir"].as<std::string>();
    }
    return a;
}

#define THREADS 512


__global__ void monte_carlo_sample_collision_dataset_uniform(float* robot_base,
                                                                Pose* poses,
                                                                StdDev* std_devs,
                                                                float* pose_idxs,
                                                                float* std_dev_idxs,
                                                                Position* positions,
                                                                float* cps,
                                                                float* accuracy_bins,
                                                                float* bin_accuracy,
                                                                int n_accuracy_bins,
                                                                int* done,
                                                                int iteration,
                                                                int n_samples,
                                                                int n_batch,
                                                                int num_left,
                                                                curandState* state) {
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(gidx >= num_left)
        return;

    curandState* localState = &state[gidx];

    int pose_idx;
    int std_dev_idx;
    Position pos;
    int n_samplestrue = 0;
    Pose pose;
    StdDev std_dev;

    n_samplestrue = (int) cps[gidx];
    pos = positions[gidx];
    pose_idx = pose_idxs[gidx];
    std_dev_idx = std_dev_idxs[gidx];
    pose = poses[pose_idx];
    std_dev = std_devs[std_dev_idx];


    float obstacle[8];
    create_rect(obstacle, pose.width, pose.height);    
    float sampled_obstacle[8];
    float robot[8];

    memcpy(robot, robot_base, sizeof(float) * 8);
    rot_trans_rectangle(robot, pos.x, pos.y, pose.theta);

    for (int i = 0; i < n_batch; i++)
    {
        sample_rectangle(obstacle, sampled_obstacle, std_dev, localState);
        n_samplestrue += convex_collide(robot, sampled_obstacle);
    }
    float slack = calcSlack(n_samples, n_samplestrue);

    float p = (float) n_samplestrue / (float) n_samples;
    int d = 0;
    if(slack <= bin_accuracy[getBin(p, accuracy_bins, n_accuracy_bins)]){
        d = 1;
    }

    done[gidx] = d;
    cps[gidx] = n_samplestrue;
}

int main(int argc, char* argv[])
{   
    Arguments args = parse_args(argc, argv);
    fs::path data_dir = args.data_dir;
    fs::path meta_dir = args.meta_dir;
    fs::path data_file_in = args.data_file_in;
    fs::path data_file_out = args.data_file_out;

    std::vector<Pose> poses;
    std::vector<Variance> variances;
    std::vector<PositionWithVarAndPoseIdx> positionWithVarAndPoseIdx;
    std::vector<float> accuracy_bins;
    std::vector<float> bin_accuracy;

    if(!fs::exists(data_dir)){
        std::cout << "Error: data_dir " << data_dir << " does not exist." << std::endl;
        exit(1);
    }
    if(meta_dir.empty()){
        fs::create_directory(data_dir / "meta");
        size_t accuracy_bins_shape[2] = {(size_t) 4};
        size_t bin_accuracy_shape[2] = {(size_t) 3};
        accuracy_bins = {0.0, 0.01, 0.1, 1.0};
        bin_accuracy = {0.0001, 0.001, 0.01};
        npy::SaveArrayAsNumpy((data_dir / "meta/accuracy_bins.npy").c_str(), false, 1, accuracy_bins_shape, reinterpret_cast<float*>(accuracy_bins.data()));
        npy::SaveArrayAsNumpy((data_dir / "meta/bin_accuracy.npy").c_str(), false, 1, bin_accuracy_shape, reinterpret_cast<float*>(bin_accuracy.data()));
    }
    if(data_file_in.empty()){
        fs::create_directories(data_dir / "tmp");
        data_file_in = data_dir / "tmp/0.npy";
        std::cout << "Using default input file: " << data_file_in << std::endl;
    }
    if(data_file_out.empty()){
        data_file_out = data_dir / "0.npy";
        std::cout << "Using default output file: " << data_file_out << "" << std::endl;
    }
    if(fs::exists(data_file_out)){
        std::cout << "Warning: " << data_file_out << " already exists, will be overwritten" << std::endl;
    }
    if(!fs::exists(data_dir / "poses.npy")){
        std::cout << "Error: " << data_dir / "poses.npy" << " does not exist." << std::endl;
        exit(1);
    }
    if(!fs::exists(data_dir / "variances.npy")){
        std::cout << "Error: " << data_dir / "variances.npy" << " does not exist." << std::endl;
        exit(1);
    }

    std::cout << "Reading data..." << std::endl;
    try {
        poses = load_numpy_array<Pose>((data_dir / "poses.npy").c_str());
        variances = load_numpy_array<Variance>((data_dir / "variances.npy").c_str());
        positionWithVarAndPoseIdx = load_numpy_array<PositionWithVarAndPoseIdx>(data_file_in.c_str());
        accuracy_bins = load_numpy_array<float>((data_dir / "meta/accuracy_bins.npy").c_str());
        bin_accuracy = load_numpy_array<float>((data_dir / "meta/bin_accuracy.npy").c_str());
    } catch (const std::exception& e) {
        std::cout << "Error while reading numpy arrays" << std::endl;
        std::cout << e.what() << std::endl;
        exit(1);
    }

    int num_poses = poses.size();
    int num_variances = variances.size();
    int num_data_points = positionWithVarAndPoseIdx.size();
    std::vector<Position> positions(num_data_points);

    std::cout << "num poses: " << num_poses << std::endl;
    std::cout << "num variances: " << num_variances << std::endl;
    std::cout << "num data points: " << num_data_points << std::endl;
    
    std::vector<StdDev> std_devs(num_variances);
    std::vector<int> index(num_data_points);
    std::iota(index.begin(), index.end(), 0);

    float* var_idxs = (float*) malloc(sizeof(float) * num_data_points);
    float* pose_idxs = (float*) malloc(sizeof(float) * num_data_points);

    for(int i = 0; i < variances.size(); i++){
        std_devs[i].x = sqrt(variances[i].x);
        std_devs[i].y = sqrt(variances[i].y);
        std_devs[i].theta = sqrt(variances[i].theta);
        std_devs[i].width = sqrt(variances[i].width);
        std_devs[i].height = sqrt(variances[i].height);
    }


    float* robot = (float*) malloc(sizeof(float)*4*2);
    float* cp = (float*) malloc(sizeof(float) * num_data_points);
    
    float* d_accuracy_bins;
    float* d_bin_accuracy;
    float* d_robot; 
    Pose* d_poses; 
    StdDev* d_std_devs; 
    Position* d_positions; 
    float* d_pose_idxs; 
    float* d_var_idxs; 
    float* d_cp; 
    int* d_done; 
    int* d_index;

    cudaMalloc(&d_accuracy_bins, sizeof(float)*(accuracy_bins.size()));
    cudaMalloc(&d_bin_accuracy, sizeof(float)*(accuracy_bins.size()));
    cudaMalloc(&d_robot, sizeof(float)*(4*2));
    cudaMalloc(&d_poses, sizeof(Pose)*num_poses);
    cudaMalloc(&d_std_devs, sizeof(StdDev)*num_variances);
    cudaMalloc(&d_positions, sizeof(Position)*num_data_points);
    cudaMalloc(&d_pose_idxs, sizeof(float)*(num_data_points));
    cudaMalloc(&d_var_idxs, sizeof(float)*(num_data_points));
    cudaMalloc(&d_cp, sizeof(float)*(num_data_points));
    cudaMalloc(&d_done, sizeof(int)*(num_data_points));
    cudaMalloc(&d_index, sizeof(int)*(num_data_points));

    DeviceZipIterator d_iter(thrust::make_tuple(thrust::device_pointer_cast(d_positions), 
                                        thrust::device_pointer_cast(d_cp),
                                        thrust::device_pointer_cast(d_var_idxs),
                                        thrust::device_pointer_cast(d_pose_idxs),
                                        thrust::device_pointer_cast(d_index)));


    std::vector<PoseCPVarAndPoseIdx> dataset(num_data_points);

    curandState *devStates;
    cudaMalloc((void **)&devStates, num_data_points *  sizeof(curandState));

    dim3 threadsPerBlock(THREADS);
    dim3 numBlocks((int) ceil( (float) num_data_points/threadsPerBlock.x));  

    // Initialize array
    create_rect(robot, args.robot_width, args.robot_height);

    // Transfer data from host to device memory
    cudaMemcpy(d_robot, robot, sizeof(float)*(4*2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_poses, poses.data(), sizeof(Pose)*num_poses, cudaMemcpyHostToDevice);
    cudaMemcpy(d_std_devs, std_devs.data(), sizeof(Variance)*num_variances, cudaMemcpyHostToDevice);
    cudaMemcpy(d_accuracy_bins, accuracy_bins.data(), sizeof(float)*accuracy_bins.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bin_accuracy, bin_accuracy.data(), sizeof(float)*bin_accuracy.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index, index.data(), sizeof(int)*num_data_points, cudaMemcpyHostToDevice);
    std::srand(std::time(0));

    setup_kernel<<<numBlocks, THREADS>>>(devStates, std::rand());

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::cout << "Total number of configurations: " << num_data_points << std::endl;
    std::cout << "Begin computation..." << std::endl;

    for (int i = 0; i < num_data_points; i++)
    {
        positions[i].x = positionWithVarAndPoseIdx[i].x;
        positions[i].y = positionWithVarAndPoseIdx[i].y;
        var_idxs[i] = positionWithVarAndPoseIdx[i].var_idx;
        pose_idxs[i] = positionWithVarAndPoseIdx[i].pose_idx;
    }
    std::iota(index.begin(), index.end(), 0);
    cudaMemcpy(d_index, index.data(), sizeof(int)*num_data_points, cudaMemcpyHostToDevice);
    // upload positions, pose_idxs, var_idxs
    cudaMemcpy(d_positions, positions.data(), sizeof(Position) * num_data_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pose_idxs, pose_idxs, sizeof(float) * num_data_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_idxs, var_idxs, sizeof(float) * num_data_points, cudaMemcpyHostToDevice);
    gpuErrchk( cudaPeekAtLastError() );
    int num_left = num_data_points;
    int batch_done = 0;
    int iteration = 0;
    int n_samples = 0;
    int n_batch = 10000;
    while(num_left > 0 && n_samples < args.max_samples){
        numBlocks = ((int) ceil((float) num_left/threadsPerBlock.x));  
        // if(n_samples < 20000)
        //     n_batch = 1000;
        // else
        //     n_batch = 100000;
        n_samples += n_batch;
        monte_carlo_sample_collision_dataset_uniform<<<numBlocks, threadsPerBlock>>>(
            d_robot,
            d_poses,
            d_std_devs,
            d_pose_idxs,
            d_var_idxs,
            d_positions,
            d_cp,
            d_accuracy_bins,
            d_bin_accuracy,
            accuracy_bins.size(),
            d_done,
            iteration,
            n_samples,
            n_batch,
            num_left,
            devStates
        );
        gpuErrchk( cudaPeekAtLastError() );
        batch_done = thrust::count(thrust::device, thrust::device_pointer_cast(d_done), thrust::device_pointer_cast(d_done + num_left), 1);
        gpuErrchk( cudaPeekAtLastError() );
        if(batch_done > 0){
            thrust::sort_by_key(thrust::device_pointer_cast(d_done), thrust::device_pointer_cast(d_done + num_left), d_iter);
            num_left -= batch_done;
            numBlocks = ((int) ceil((float) batch_done/threadsPerBlock.x));  
            write_collision_probability<<<numBlocks, threadsPerBlock>>>(d_cp + num_left, batch_done, n_samples);
            cudaMemcpy(positions.data() + num_left, d_positions + num_left, sizeof(Position) * batch_done, cudaMemcpyDeviceToHost);
            cudaMemcpy(cp + num_left, d_cp + num_left, sizeof(float) * batch_done, cudaMemcpyDeviceToHost);
            cudaMemcpy(var_idxs + num_left, d_var_idxs + num_left, sizeof(float) * batch_done, cudaMemcpyDeviceToHost);
            cudaMemcpy(pose_idxs + num_left, d_pose_idxs + num_left, sizeof(float) * batch_done, cudaMemcpyDeviceToHost);
            cudaMemcpy(index.data() + num_left, d_index + num_left, sizeof(int) * batch_done, cudaMemcpyDeviceToHost);
        }
        iteration++;
    }


    std::cout << "Num left: " << num_left << std::endl;
    if(num_left > 0){
        numBlocks = ((int) ceil((float) num_left/threadsPerBlock.x));  
        write_collision_probability<<<numBlocks, threadsPerBlock>>>(d_cp, num_data_points, n_samples);
        gpuErrchk( cudaPeekAtLastError() );
        cudaMemcpy(positions.data(), d_positions, sizeof(Position) * num_left, cudaMemcpyDeviceToHost);
        cudaMemcpy(cp, d_cp, sizeof(float) * num_left, cudaMemcpyDeviceToHost);
        cudaMemcpy(var_idxs, d_var_idxs, sizeof(float) * num_left, cudaMemcpyDeviceToHost);
        cudaMemcpy(pose_idxs, d_pose_idxs, sizeof(float) * num_left, cudaMemcpyDeviceToHost); 
        cudaMemcpy(index.data(), d_index, sizeof(int) * num_left, cudaMemcpyDeviceToHost);
    }

    CUDA_CALL(cudaDeviceSynchronize());

    std::vector<float> cps(num_data_points);
    if(args.cps_only){
        for (int i = 0; i < num_data_points; i++)
        {
            cps[index[i]] = cp[i];
        }
    }
    else{
        for (int j = 0; j < num_data_points; j++)
        {
            dataset[index[j]].x = positions[j].x;
            dataset[index[j]].y = positions[j].y;
            dataset[index[j]].cp = cp[j];
            dataset[index[j]].var_idx = var_idxs[j];
            dataset[index[j]].pose_idx = pose_idxs[j];
        }
    }

    if(args.shuffle)
    {
        if(!args.cps_only)
            {shuffle(cps.begin(), cps.end(), std::default_random_engine(0));}
        else
            {shuffle(dataset.begin(), dataset.end(), std::default_random_engine(0));}
    }


    // write dataset
    if(args.cps_only) {
        size_t cps_shape[2] = {(size_t) num_data_points};
        npy::SaveArrayAsNumpy(data_file_out.c_str(), false, 1, cps_shape, reinterpret_cast<float*>(cps.data()));
    } else {    
        size_t ds_shape[2] = {(size_t) num_data_points, (size_t) 5};
        npy::SaveArrayAsNumpy(data_file_out.c_str(), false, 2, ds_shape, reinterpret_cast<float*>(dataset.data()));
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Finished computation" << std::endl;
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::minutes>(end - begin).count() << " [min]" << std::endl;

    // free memory
    cudaFree(devStates);
    cudaFree(d_accuracy_bins);
    cudaFree(d_bin_accuracy);
    cudaFree(d_robot);
    cudaFree(d_poses);
    cudaFree(d_std_devs);
    cudaFree(d_positions);
    cudaFree(d_pose_idxs);
    cudaFree(d_var_idxs);
    cudaFree(d_cp);
    cudaFree(d_done);
    std::cout << "Done." << std::endl;
}
