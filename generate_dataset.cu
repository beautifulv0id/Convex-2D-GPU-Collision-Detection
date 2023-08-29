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
#include <boost/program_options.hpp>

 /*
 This file creates a dataset of collision probabilities between a robot and an obstacle modeled as rectangles for different configurations using Monte Carlo sampling.
 One data-point defines the width, height and variance of the obstacle, as well as the position and orientation of the robot w.r.t the obstacle coordinate frame.
 First, num_poses poses and num_variances variances are unifomely sampled from user-defined bounds. A pose contains the width and height of the obstacle and angle 
 theta of the robot. Variances are defined for the position and orientation of the obstacle. Finally, for each data-point a robot position is uniformly sampled, 
 as well as a random pose and variance from the pregenerated poses and variances. 
 */

namespace po = boost::program_options;

struct Arguments {
    std::string data_dir = "./data/";
    int num_batches = 100;
    int batch_size = 100000;
    int start_batch_count = 0;
    int num_poses = 64*64*64*64;
    int num_variances = 64*64*64*64;
    int max_samples = 4000000;
    std::vector<float> min_variance = {0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> max_variance = {0.3, 0.3, 0.3, 0.3, 0.3};
    std::vector<float> min_pose = {0.1, 0.1, 0.0};
    std::vector<float> max_pose = {5, 5, 2*M_PI};
    std::vector<float> accuracy_bins = {0.0, 0.01, 0.1, 1.0};
    std::vector<float> bin_accuracy = {0.0001, 0.001, 0.01};
    float robot_width = 4.07;
    float robot_height = 1.74;
    float spread = 3;
};

Arguments parse_args(int argc, char** argv) {
    Arguments a;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("data_dir", po::value<std::string>(), "where to store the data")
        ("num_batches,n", po::value<int>(), "number of batches")
        ("batch_size,b", po::value<int>(), "number of samples per batch")
        ("start_batch_count,s", po::value<int>(), "start value for batches")
        ("num_poses", po::value<int>(), "number of poses")
        ("num_variances", po::value<int>(), "number of variances")
        ("max_samples", po::value<int>(), "maximum number of samples for z-test")
        ("accuracy_bins", po::value<std::vector<float>>()->multitoken(), "accuracy bins e.g. [0.0001 0.001 0.01 0]")
        ("bin_accuracy", po::value<std::vector<float>>()->multitoken(), "accuracy for each bin e.g. [0.0001, 0.001, 0.01]")
        ("min_variance", po::value<std::vector<float>>()->multitoken(), "min variance for each dimension e.g. [0.0, 0.0, 0.0, 0.0, 0.0]")
        ("max_variance", po::value<std::vector<float>>()->multitoken(), "max variance for each dimension e.g. [0.3, 0.3, 0.3, 0.3, 0.3]")
        ("min_pose", po::value<std::vector<float>>()->multitoken(), "min pose for each dimension e.g. [0.1, 0.1, 0.0]")
        ("max_pose", po::value<std::vector<float>>()->multitoken(), "max pose for each dimension e.g. [5, 5, 2*M_PI]")
        ("robot_width,w", po::value<float>(), "robot width")
        ("robot_height,h", po::value<float>(), "robot height")
        ("spread", po::value<float>(), "spread of poses");
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
    if(vm.count("num_batches")) {
        a.num_batches = vm["num_batches"].as<int>();
    }
    if(vm.count("batch_size")) {
        a.batch_size = vm["batch_size"].as<int>();
    }
    if(vm.count("start_batch_count")) {
        a.start_batch_count = vm["start_batch_count"].as<int>();
    }
    if(vm.count("num_poses")) {
        a.num_poses = vm["num_poses"].as<int>();
    }
    if(vm.count("num_variances")) {
        a.num_variances = vm["num_variances"].as<int>();
    }
    if(vm.count("max_samples")) {
        a.max_samples = vm["max_samples"].as<int>();
    }
    if (vm.count("accuracy_bins")) {
        std::vector<float> values = vm["accuracy_bins"].as<std::vector<float>>();
        a.accuracy_bins = values;
    }
    if (vm.count("bin_accuracy")) {
        std::vector<float> values = vm["bin_accuracy"].as<std::vector<float>>();
        a.bin_accuracy = values;
    }
    if (vm.count("min_variance")) {
        std::vector<float> values = vm["min_variance"].as<std::vector<float>>();
        assert(values.size() == 5);
        a.min_variance = values;
    }
    if (vm.count("max_variance")) {
        std::vector<float> values = vm["max_variance"].as<std::vector<float>>();
        assert(values.size() == 5);
        a.max_variance = values;
    }
    if (vm.count("min_pose")) {
        std::vector<float> values = vm["min_pose"].as<std::vector<float>>();
        assert(values.size() == 3);
        a.min_pose = values;
    }
    if (vm.count("max_pose")) {
        std::vector<float> values = vm["max_pose"].as<std::vector<float>>();
        assert(values.size() == 3);
        a.max_pose = values;
    }
    if(vm.count("robot_width")) {
        a.robot_width = vm["robot_width"].as<float>();
    }
    if(vm.count("robot_height")) {
        a.robot_height = vm["robot_height"].as<float>();
    }
    if(vm.count("spread")) {
        a.spread = vm["spread"].as<float>();
    }
    return a;
}


#define THREADS 512


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

struct Position
{
    float x, y;
};

struct Variance
{
    float x, y, theta, width, height;
};

struct Pose
{
    float width, height, theta;
};


typedef Variance StdDev;
typedef thrust::device_vector<Position>::iterator   DeviceFloatPairIterator;
typedef thrust::device_vector<float>::iterator   DeviceFloatIterator;
typedef thrust::device_vector<int>::iterator   DeviceIntIterator;
typedef thrust::tuple<DeviceFloatPairIterator, DeviceFloatIterator, DeviceFloatIterator, DeviceFloatIterator> DeviceIteratorTuple;
typedef thrust::zip_iterator<DeviceIteratorTuple> DeviceZipIterator;


__global__ void setup_kernel(curandState *state, int seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets different seed, a different sequence
       number, no offset */
    curand_init(seed+id, id, 0, &state[id]);
}

__device__ __host__
void create_rect(float* r, float w, float h)
{
    r[0] = -w/2;
    r[1] = -h/2;
    r[2] = w/2;
    r[3] = -h/2;
    r[4] = w/2;
    r[5] = h/2;
    r[6] = -w/2;
    r[7] = h/2; 
}

__device__ void rot_trans_rectangle(float* r, float dx, float dy, float dt){
    float c = cosf(dt);
    float s = sinf(dt);
    float x,y;
    for(int i = 0; i < 4; i++){
        x = r[2*i];
        y = r[2*i+1];
        r[2*i] = c*x-s*y + dx;
        r[2*i+1] = s*x+c*y + dy;
    }
}

__device__ void sample_rectangle(float* r_in, float* r_out, StdDev& std_dev, curandState* state)
{
    float dx = curand_normal(state) * std_dev.x;
    float dy = curand_normal(state) * std_dev.y; 
    float dt = curand_normal(state) * std_dev.theta;
    float dw = curand_normal(state) * std_dev.width; 
    float dh = curand_normal(state) * std_dev.height;
    
    memcpy(r_out, r_in, sizeof(float) * 8);
    float dwh[8];
    create_rect(dwh, dw, dh);
    for (int i = 0; i < 8; i++){ r_out[i] += dwh[i]; }
    rot_trans_rectangle(r_out, dx, dy, dt);
}

__device__ int convex_collide(float* r1, float* r2) {
    float* rs[2] = {r1, r2};
    float norm[2];
    float p1[4];
    float p2[4];
    float* r;

    int collide = 1;
    for(int j = 0; j < 2; j++){
        r = rs[j];
        for(int i = 0; i < 4; i++){
            norm[0] = r[(i+1)*2%8] - r[i*2];
            norm[1] = r[((i+1)*2+1)%8] - r[i*2+1];
            for(int k = 0; k < 4; k++){
                p1[k] = norm[0]*r1[k*2]+norm[1]*r1[k*2+1];
                p2[k] = norm[0]*r2[k*2]+norm[1]*r2[k*2+1];
            }
            thrust::pair<float *, float *> result1 = thrust::minmax_element(thrust::device, p1, p1 + 4);
            thrust::pair<float *, float *> result2 = thrust::minmax_element(thrust::device, p2, p2 + 4);
            if(*result1.second < *result2.first || *result2.second < *result1.first){
                collide = 0;
            }
        }
    }
    return collide;
}

__device__
float calcSlack(int nsamples, int nsamples_true){
    float z = 1.96;
    float alpha = 0.025;
    if((nsamples_true == nsamples) || (nsamples_true == 0)){
        return log(1.0 / alpha) / nsamples;
    }
    else{
        return z / nsamples * sqrt((float) nsamples_true - nsamples_true * nsamples_true / (float) nsamples);
    }
}

__device__
int getBin(float p, float* accuracy_bins, int n_accuracy_bins){
    int bin = 0;
    for (int i = 0; i < n_accuracy_bins; i++){
        if(p >= accuracy_bins[i] && p <= accuracy_bins[i+1]){
            bin = i;
        }
    }
    return bin;
}


__global__ void write_collision_probability(float* n_true, int n_done, int n_samples){
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(gidx >= n_done)
        return;
    n_true[gidx] =  n_true[gidx] / (float) n_samples;
}


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
                                                                int num_poses,
                                                                int num_variances,
                                                                float r_offset,
                                                                float spread,
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
    if(iteration == 0){
        pose_idx = curand(localState) % num_poses;
        std_dev_idx = curand(localState) % num_variances;
        pose = poses[pose_idx];
        std_dev = std_devs[std_dev_idx];

        float theta = curand_uniform(localState) * 2 * M_PI;
        float shift = curand_normal(localState) * ((std_dev.y+std_dev.x)/2)*spread;
        pos.x = cosf(theta) * ((pose.width/2+r_offset + std_dev.x) + shift); 
        pos.y = sinf(theta) * ((pose.height/2+r_offset + std_dev.y) + shift); 
        positions[gidx] = pos;
        pose_idxs[gidx] = pose_idx;
        std_dev_idxs[gidx] = std_dev_idx;    
    } else {
        n_samplestrue = (int) cps[gidx];
        pos = positions[gidx];
        pose_idx = pose_idxs[gidx];
        std_dev_idx = std_dev_idxs[gidx];
        pose = poses[pose_idx];
        std_dev = std_devs[std_dev_idx];
    }


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
    std::string data_dir = args.data_dir;
    int num_batches = args.num_batches;
    int batch_size = args.batch_size;
    int start_batch_count = args.start_batch_count;
    int num_poses = args.num_poses;
    int num_variances = args.num_variances;
    int pose_dim = 3;
    int variance_dim = 5;
    std::vector<float>& accuracy_bins = args.accuracy_bins;
    std::vector<float>& bin_accuracy = args.bin_accuracy;

    
    std::cout << "data dir: " << data_dir << std::endl;
    std::cout << "num batches: " << num_batches << std::endl;
    std::cout << "num batch: " << batch_size << std::endl;
    std::cout << "start batch count: " << start_batch_count << std::endl;

    // write_config(data_dir, batch_size, num_batches);

    Pose* poses = (Pose*) malloc(sizeof(Pose)*num_poses);
    StdDev* std_devs = (Variance*) malloc(sizeof(StdDev)*num_variances);  
    std::vector<Variance> variances(num_variances);  

    std::default_random_engine generator;
    std::vector<std::uniform_real_distribution<float>> pose_uniforms;
    std::vector<std::uniform_real_distribution<float>> variance_uniforms;

    for(int i = 0; i < pose_dim; i++){
        pose_uniforms.push_back(std::uniform_real_distribution<float>(args.min_pose[i], args.max_pose[i]));
    }
    for(int i = 0; i < variance_dim; i++){
        variance_uniforms.push_back(std::uniform_real_distribution<float>(args.min_variance[i], args.max_variance[i]));
    }

    for (int i = 0; i < num_poses; i++)
    {
        poses[i].width = pose_uniforms[0](generator);
        poses[i].height = pose_uniforms[1](generator);
        poses[i].theta = pose_uniforms[3](generator);
    }    
    for (int i = 0; i < num_variances; i++)
    {
        variances[i].x = variance_uniforms[0](generator);
        variances[i].y = variance_uniforms[1](generator);
        variances[i].theta = variance_uniforms[3](generator);
        variances[i].width = variance_uniforms[4](generator);
        variances[i].height = variance_uniforms[5](generator);     
        std_devs[i].x = sqrt(variances[i].x);
        std_devs[i].y = sqrt(variances[i].y);
        std_devs[i].theta = sqrt(variances[i].theta);
        std_devs[i].width = sqrt(variances[i].width);   
        std_devs[i].height = sqrt(variances[i].height);   
    }

    // write poses and variances
    size_t poses_shape[2] = {(size_t) num_poses, (size_t) 3};
    size_t variances_shape[2] = {(size_t) num_variances, (size_t) 5};
    npy::SaveArrayAsNumpy(data_dir + std::string("/poses.npy"), false, 2, poses_shape, (float*) poses);
    npy::SaveArrayAsNumpy(data_dir + std::string("/variances.npy"), false, 2, variances_shape, (float*) variances.data());

    struct stat st = {0};

    if (stat(data_dir.c_str(), &st) == -1) {
        mkdir(data_dir.c_str(), 0700);
    }
    if(stat((data_dir + "/meta").c_str(), &st) == -1){
        mkdir((data_dir + "/meta").c_str(), 0700);
    }
    size_t accuracy_bins_shape[1] = {(size_t) accuracy_bins.size()};
    size_t bin_accuracy_shape[1] = {(size_t) bin_accuracy.size()};
    npy::SaveArrayAsNumpy(data_dir + "/meta" + std::string("/accuracy_bins.npy"), false, 1, accuracy_bins_shape, accuracy_bins);
    npy::SaveArrayAsNumpy(data_dir + "/meta" + std::string("/bin_accuracy.npy"), false, 1, bin_accuracy_shape, bin_accuracy);

    float* robot = (float*) malloc(sizeof(float)*4*2);
    Position* positions = (Position*) malloc(sizeof(Position) * batch_size);
    float* pose_idxs = (float*) malloc(sizeof(float) * batch_size);
    float* var_idxs = (float*) malloc(sizeof(float) * batch_size);
    float* cp = (float*) malloc(sizeof(float) * batch_size);
    
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

    cudaMalloc(&d_accuracy_bins, sizeof(float)*(accuracy_bins.size()));
    cudaMalloc(&d_bin_accuracy, sizeof(float)*(bin_accuracy.size()));
    cudaMalloc(&d_robot, sizeof(float)*(4*2));
    cudaMalloc(&d_poses, sizeof(Pose)*num_poses);
    cudaMalloc(&d_std_devs, sizeof(StdDev)*num_variances);
    cudaMalloc(&d_positions, sizeof(Position)*batch_size);
    cudaMalloc(&d_pose_idxs, sizeof(float)*(batch_size));
    cudaMalloc(&d_var_idxs, sizeof(float)*(batch_size));
    cudaMalloc(&d_cp, sizeof(float)*(batch_size));
    cudaMalloc(&d_done, sizeof(int)*(batch_size));

    DeviceZipIterator d_iter(thrust::make_tuple(thrust::device_pointer_cast(d_positions), 
                                        thrust::device_pointer_cast(d_cp),
                                        thrust::device_pointer_cast(d_var_idxs),
                                        thrust::device_pointer_cast(d_pose_idxs)));


    std::vector<float> dataset(batch_size*5);

    curandState *devStates;
    cudaMalloc((void **)&devStates, batch_size *  sizeof(curandState));

    dim3 threadsPerBlock(THREADS);
    dim3 numBlocks((int) std::max(1.0, ceil(batch_size/threadsPerBlock.x)));  

    // Initialize array
    create_rect(robot, args.robot_width, args.robot_height);
    float r_offset = (args.robot_width+args.robot_height)/4;

    // Transfer data from host to device memory
    cudaMemcpy(d_robot, robot, sizeof(float)*(4*2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_poses, poses, sizeof(Pose)*num_poses, cudaMemcpyHostToDevice);
    cudaMemcpy(d_std_devs, std_devs, sizeof(Variance)*num_variances, cudaMemcpyHostToDevice);
    cudaMemcpy(d_accuracy_bins, accuracy_bins.data(), sizeof(float)*(accuracy_bins.size()), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bin_accuracy, bin_accuracy.data(), sizeof(float)*(bin_accuracy.size()), cudaMemcpyHostToDevice);
    // std::srand(std::time(0));
    CUDA_CALL(cudaPeekAtLastError());

    setup_kernel<<<numBlocks, THREADS>>>(devStates, std::rand());
    CUDA_CALL(cudaPeekAtLastError());

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::cout << "Total number of configurations: " << batch_size * num_batches << std::endl;
    std::cout << "Begin computation..." << std::endl;
    int counter = 0;
    printf("batches generated: %i/%i", counter, num_batches);

    for (int batch_index = 0; batch_index < num_batches; batch_index++)
    {
        int num_left = batch_size;
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
                num_poses,
                num_variances,
                r_offset,
                args.spread,
                devStates
            );
            gpuErrchk( cudaPeekAtLastError() );
            batch_done = thrust::count(thrust::device, thrust::device_pointer_cast(d_done), thrust::device_pointer_cast(d_done + num_left), 1);
            if(batch_done > 0){
                thrust::sort_by_key(thrust::device_pointer_cast(d_done), thrust::device_pointer_cast(d_done + num_left), d_iter);
                num_left -= batch_done;
                numBlocks = ((int) ceil((float) batch_done/threadsPerBlock.x));  
                write_collision_probability<<<numBlocks, threadsPerBlock>>>(d_cp + num_left, batch_done, n_samples);
                cudaMemcpy(positions + num_left, d_positions + num_left, sizeof(Position) * batch_done, cudaMemcpyDeviceToHost);
                cudaMemcpy(cp + num_left, d_cp + num_left, sizeof(float) * batch_done, cudaMemcpyDeviceToHost);
                cudaMemcpy(var_idxs + num_left, d_var_idxs + num_left, sizeof(float) * batch_done, cudaMemcpyDeviceToHost);
                cudaMemcpy(pose_idxs + num_left, d_pose_idxs + num_left, sizeof(float) * batch_done, cudaMemcpyDeviceToHost);
            }
            iteration++;
            // printf("num left %i, n_samples %i\n", num_left, n_samples);
        }

        if(num_left > 0){
            printf("copying remaining %i over (0, %i)\n", num_left, num_left);
            numBlocks = ((int) ceil((float) num_left/threadsPerBlock.x));  
            write_collision_probability<<<numBlocks, threadsPerBlock>>>(d_cp, num_left, n_samples);
            gpuErrchk( cudaPeekAtLastError() );
            cudaMemcpy(positions, d_positions, sizeof(Position) * num_left, cudaMemcpyDeviceToHost);
            cudaMemcpy(cp, d_cp, sizeof(float) * num_left, cudaMemcpyDeviceToHost);
            cudaMemcpy(var_idxs, d_var_idxs, sizeof(float) * num_left, cudaMemcpyDeviceToHost);
            cudaMemcpy(pose_idxs, d_pose_idxs, sizeof(float) * num_left, cudaMemcpyDeviceToHost); 
        }


        CUDA_CALL(cudaDeviceSynchronize());

        // write data
        float* d = dataset.data();
        float* c = cp;

        for (int j = 0; j < batch_size; j++)
        {
            d[0] = positions[j].x;  // x
            d[1] = positions[j].y;  // y
            d[2] = *c;  // cp
            d[3] = var_idxs[j]; // var_idx
            d[4] = pose_idxs[j]; // pose_idx
            c+=1;
            d+=5;
        }

        // write dataset
        size_t ds_shape[2] = {(size_t) batch_size, (size_t) 5};

        npy::SaveArrayAsNumpy(data_dir + std::string("/") + std::to_string(start_batch_count + batch_index) + ".npy", false, 2, ds_shape, dataset);
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
