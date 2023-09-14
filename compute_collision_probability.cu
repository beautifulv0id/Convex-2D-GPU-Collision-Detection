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
namespace po = boost::program_options;

struct Arguments {
    std::string data_in = "./data_in/";
    std::string data_out = "./data_out/";
    int max_samples = 4000000;
    float robot_width = 4.07;
    float robot_height = 1.74;
    bool shuffle = true;
};

Arguments parse_args(int argc, char** argv) {
    Arguments a;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("data_in", po::value<std::string>(), "where to read the data")
        ("data_out", po::value<std::string>(), "where to write the data")
        ("max_samples", po::value<int>(), "maximum number of samples for z-test")
        ("robot_width,w", po::value<float>(), "robot width")
        ("robot_height,h", po::value<float>(), "robot height")
        ("shuffle", po::value<bool>(),"whether or not to shuffle data")

    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(1);
    }
    if(vm.count("data_in")) {
        a.data_in = vm["data_in"].as<std::string>();
    }
    if(vm.count("data_out")) {
        a.data_out = vm["data_out"].as<std::string>();
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
    std::string data_in = args.data_in;
    std::string data_out = args.data_out;
    int start_batch_count = get_num_batches_in_dir(data_out);
    int num_batches = get_num_batches_in_dir(data_in);

    std::cout << "Reading data..." << std::endl;

    std::vector<Pose> poses = load_numpy_array<Pose>(data_out + std::string("/poses.npy"));
    std::vector<Variance> variances = load_numpy_array<Variance>(data_out + std::string("/variances.npy"));
    std::vector<PositionWithVarAndPoseIdx> positionWithVarAndPoseIdx = load_numpy_array<PositionWithVarAndPoseIdx>(data_in + std::string("/0.npy"));
    std::vector<float> accuracy_bins = load_numpy_array<float>(data_out + std::string("/meta") + std::string("/accuracy_bins.npy"));
    std::vector<float> bin_accuracy = load_numpy_array<float>(data_out + std::string("/meta") + std::string("/bin_accuracy.npy"));


    int num_poses = poses.size();
    int num_variances = variances.size();
    int num_data_points = positionWithVarAndPoseIdx.size();

    std::vector<Position> positions(num_data_points);


    std::cout << "num poses: " << num_poses << std::endl;
    std::cout << "num variances: " << num_variances << std::endl;
    std::cout << "num data points: " << num_data_points << std::endl;
    
    // StdDev* std_devs = (Variance*) malloc(sizeof(StdDev)*num_variances);  
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
    dim3 numBlocks((int) std::max(1.0, ceil(num_data_points/threadsPerBlock.x)));  

    // Initialize array
    create_rect(robot, args.robot_width, args.robot_height);

    // Transfer data from host to device memory
    cudaMemcpy(d_robot, robot, sizeof(float)*(4*2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_poses, poses.data(), sizeof(Pose)*num_poses, cudaMemcpyHostToDevice);
    cudaMemcpy(d_std_devs, std_devs.data(), sizeof(Variance)*num_variances, cudaMemcpyHostToDevice);
    cudaMemcpy(d_accuracy_bins, accuracy_bins.data(), sizeof(float)*accuracy_bins.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bin_accuracy, bin_accuracy.data(), sizeof(float)*bin_accuracy.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index, index.data(), sizeof(int)*num_data_points, cudaMemcpyHostToDevice);
    // std::srand(std::time(0));

    setup_kernel<<<numBlocks, THREADS>>>(devStates, std::rand());

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::cout << "Total number of configurations: " << num_data_points * num_batches << std::endl;
    std::cout << "Begin computation..." << std::endl;
    int counter = 0;
    printf("batches generated: %i/%i\n", counter, num_batches);

    for (int batch_index = 0; batch_index < num_batches; batch_index++)
    {
        std::vector<PositionWithVarAndPoseIdx> pos_with_idx = load_numpy_array<PositionWithVarAndPoseIdx>(data_in + std::string("/") + std::to_string(batch_index) + std::string(".npy"));
        for (int i = 0; i < num_data_points; i++)
        {
            positions[i].x = pos_with_idx[i].x;
            positions[i].y = pos_with_idx[i].y;
            var_idxs[i] = pos_with_idx[i].var_idx;
            pose_idxs[i] = pos_with_idx[i].pose_idx;
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
        int n_batch = 0;
        while(num_left > 0 && n_samples < args.max_samples){
            numBlocks = ((int) ceil((float) num_left/threadsPerBlock.x));  
            if(n_samples < 20000)
                n_batch = 1000;
            else
                n_batch = 100000;
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

        // write data
        for (int j = 0; j < num_data_points; j++)
        {
            dataset[index[j]].x = positions[j].x;
            dataset[index[j]].y = positions[j].y;
            dataset[index[j]].cp = cp[j];
            dataset[index[j]].var_idx = var_idxs[j];
            dataset[index[j]].pose_idx = pose_idxs[j];
        }

        if(args.shuffle)
        {
            shuffle(dataset.begin(), dataset.end(), std::default_random_engine(0));
        }


        // write dataset
        size_t ds_shape[2] = {(size_t) num_data_points, (size_t) 5};

        npy::SaveArrayAsNumpy(data_out + std::string("/") + std::to_string(start_batch_count + batch_index) + ".npy", false, 2, ds_shape, reinterpret_cast<float*>(dataset.data()));
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        printf("\33[2K\r");
        printf("batches generated: %i/%i, Time: %i [min]", ++counter, num_batches, (int) std::chrono::duration_cast<std::chrono::minutes>(end - begin).count());
        fflush(stdout); 
    }
    std::cout << std::endl;
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
