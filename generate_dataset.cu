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

 /*
 This file creates a dataset of collision probabilities between a robot and an obstacle modeled as rectangles for different configurations using Monte Carlo sampling.
 One data-point defines the width, height and variance of the obstacle, as well as the position and orientation of the robot w.r.t the obstacle coordinate frame.
 First, NUM_POSES poses and NUM_VARIANCES variances are unifomely sampled from user-defined bounds. A pose contains the width and height of the obstacle and angle 
 theta of the robot. Variances are defined for the position and orientation of the obstacle. Finally, for each data-point a robot position is uniformly sampled, 
 as well as a random pose and variance from the pregenerated poses and variances. 
 */

#define UNIFORM_SAMPLING 0

#define N 4000000

// SIMULATION
#define R_WIDTH 4.07 // width of the robot
#define R_HEIGHT 1.74 // height of the robot

// TIAGO
// #define R_WIDTH 0.75 // width of the robot
// #define R_HEIGHT 0.8 // height of the robot


#define R_OFFSET ((R_WIDTH + R_HEIGHT) / 4)

#define NUM_POSES (64*64*64*64) // number of poses that are sampled, a pose contains the width, height of the obstacle and angle theta of the robot
#define NUM_VARIANCES (64*64*64*64) // number of variances that are sampled, 

#define POS_MIN -12.0 // minimium x-, y-position of the robot
#define POS_MAX 12.0 // maximum x-, y-position of the robot

#define OBSTACLE_WIDTH_MIN 0.1 // minimum width, height of obstacles
#define OBSTACLE_WIDTH_MAX 5.0 // maximum width, height of obstacles

#define VAR_MIN 0.0 // minimum positional and rotational variance 
#define VAR_MAX 0.3 // maximum positional and rotational variance 


#define THREADS 512
 
// float accuracy_bins[] = {0, 0.001, 0.01, 0.1, 1};
// float bin_slack[] = {0.00005,0.0005, 0.001, 0.01, 0};
#define N_ACCURACY_BINS 3
float accuracy_bins[N_ACCURACY_BINS+1] = {0, 0.01, 0.1, 1};
float bin_slack[N_ACCURACY_BINS+1] = {0.0001, 0.001, 0.01, 0};
// float bin_slack[] = {0.00002,0.0001, 0.0005, 0.001, 0};



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

void write_config(std::string data_dir, int num_batch, int num_batches){
    std::ofstream confFile;
    confFile.open (data_dir + std::string("/config.txt"), std::ios::out);
    confFile << "UNIFORM_SAMPLING" << "\t" << UNIFORM_SAMPLING << "\n";
    confFile << "N" << "\t" << N << "\n";
    confFile << "R_WIDTH" << "\t" << R_WIDTH << "\n";
    confFile << "R_HEIGHT" << "\t" << R_HEIGHT << "\n";
    confFile << "NUM_BATCH" << "\t" << num_batch << "\n";
    confFile << "NUM_BATCHES" << "\t" << num_batches << "\n";
    confFile << "NUM_DATA_POINTS" << "\t" << num_batch * num_batches << "\n";
    confFile << "NUM_POSES" << "\t" << NUM_POSES << "\n";
    confFile << "NUM_VARIANCES" << "\t" << NUM_VARIANCES << "\n";
    confFile << "POS_MIN" << "\t" << POS_MIN << "\n";
    confFile << "POS_MAX" << "\t" << POS_MAX << "\n";
    confFile << "OBSTACLE_WIDTH_MIN" << "\t" << OBSTACLE_WIDTH_MIN << "\n";
    confFile << "OBSTACLE_WIDTH_MAX" << "\t" << OBSTACLE_WIDTH_MAX << "\n";
    confFile << "VAR_MIN" << "\t" << VAR_MIN << "\n";
    confFile << "VAR_MAX" << "\t" << VAR_MAX << "\n";
    confFile << "ACCURACY_BINS" << "\t";
    for (size_t i = 0; i < N_ACCURACY_BINS+1; i++)
        confFile << accuracy_bins[i] << " ";
    confFile << "\n";
    confFile << "BIN_SLACK" << "\t";
    for (size_t i = 0; i < N_ACCURACY_BINS; i++)
        confFile << bin_slack[i] << " ";
    confFile << "\n";
    
    confFile.close();

}
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
int getBin(float p, float* accuracy_bins){
    int bin = 0;
    for (int i = 0; i < N_ACCURACY_BINS; i++){
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
                                                                float* bin_slack,
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
    if(iteration == 0){
        pose_idx = curand(localState) % NUM_POSES;
        std_dev_idx = curand(localState) % NUM_VARIANCES;
        pose = poses[pose_idx];
        std_dev = std_devs[std_dev_idx];
        if(UNIFORM_SAMPLING == 1){
            pos.x = POS_MIN + curand_uniform(localState) * (POS_MAX-POS_MIN);
            pos.y = POS_MIN + curand_uniform(localState) * (POS_MAX-POS_MIN);
        } else {
            float theta = curand_uniform(localState) * 2 * M_PI;
            float shift = curand_normal(localState) * ((pose.width+pose.height)/2+R_OFFSET)/2;
            pos.x = cosf(theta) * (pose.width/2+R_OFFSET) * 1.3 + shift; // * ((std_dev.x+std_dev.width/2)/4);
            pos.y = sinf(theta) * (pose.height/2+R_OFFSET) * 1.3 + shift; // * ((std_dev.y+std_dev.height/2)/4);
        }    
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
    if(slack <= bin_slack[getBin(p, accuracy_bins)]){
        d = 1;
    }

    done[gidx] = d;
    cps[gidx] = n_samplestrue;
}

int main(int argc, char* argv[])
{   
    std::string data_dir = "data";
    int start_batch_count = 0;
    int num_batch = 1000000;
    int num_batches = 100;
    if(argc > 1){
        data_dir = std::string(argv[1]);
        if(argc > 2)
            num_batches = std::stoi(argv[2]);
        if (argc > 3) 
            num_batch = std::stoi(argv[3]);
        if (argc > 4)
            start_batch_count = std::stoi(argv[4]);

    }
    
    std::cout << "data dir: " << data_dir << std::endl;
    std::cout << "num batches: " << num_batches << std::endl;
    std::cout << "num batch: " << num_batch << std::endl;
    std::cout << "start batch count: " << start_batch_count << std::endl;

    int nDevices;
    cudaGetDeviceCount(&nDevices);

    printf("Number of devices: %d\n", nDevices);

    for (int i = 0; i < 1; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (MHz): %d\n",
                prop.memoryClockRate/1024);
        printf("  Memory Bus Width (bits): %d\n",
                prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
                2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
        printf("  maxThreadsDim.x: %i\n", prop.maxThreadsDim[0]);
        printf("  maxThreadsDim.y: %i\n", prop.maxThreadsDim[1]);        
        printf("  maxThreadsDim.z: %i\n", prop.maxThreadsDim[2]);
        printf("  maxThreadsPerBlock: %i\n", prop.maxThreadsPerBlock);
        printf("  maxBlocksPerMultiProcessor: %i\n", prop.maxBlocksPerMultiProcessor);
        printf("  maxGridSize.x: %i\n", prop.maxGridSize[0]);
        printf("  maxGridSize.y: %i\n", prop.maxGridSize[1]);        
        printf("  maxGridSize.z: %i\n", prop.maxGridSize[2]);

    }

    write_config(data_dir, num_batch, num_batches);

    Pose* poses = (Pose*) malloc(sizeof(Pose)*NUM_POSES);
    StdDev* std_devs = (Variance*) malloc(sizeof(StdDev)*NUM_VARIANCES);  
    std::vector<Variance> variances(NUM_VARIANCES);  

    std::default_random_engine generator;
    auto obstacle_uniform = std::uniform_real_distribution<float>(OBSTACLE_WIDTH_MIN, OBSTACLE_WIDTH_MAX);
    auto obstacle_scale_uniform = std::uniform_real_distribution<float>(0.25, 1.);
    auto theta_uniform = std::uniform_real_distribution<float>(0.0, 2.0*M_PI);
    auto variance_uniform = std::uniform_real_distribution<float>(VAR_MIN, VAR_MAX);

    for (int i = 0; i < NUM_POSES; i++)
    {
        poses[i].width = obstacle_uniform(generator);
        poses[i].height = obstacle_uniform(generator);
        poses[i].theta = theta_uniform(generator);
    }    
    for (int i = 0; i < NUM_VARIANCES; i++)
    {
        variances[i].x = variance_uniform(generator);
        variances[i].y = variance_uniform(generator);
        variances[i].theta = variance_uniform(generator);       
        variances[i].width = variance_uniform(generator);       
        variances[i].height = variance_uniform(generator);       
        std_devs[i].x = sqrt(variances[i].x);
        std_devs[i].y = sqrt(variances[i].y);
        std_devs[i].theta = sqrt(variances[i].theta);
        std_devs[i].width = sqrt(variances[i].width);   
        std_devs[i].height = sqrt(variances[i].height);   
    }

    // write poses and variances
    size_t poses_shape[2] = {(size_t) NUM_POSES, (size_t) 3};
    size_t variances_shape[2] = {(size_t) NUM_VARIANCES, (size_t) 5};
    npy::SaveArrayAsNumpy(data_dir + std::string("/poses.npy"), false, 2, poses_shape, (float*) poses);
    npy::SaveArrayAsNumpy(data_dir + std::string("/variances.npy"), false, 2, variances_shape, (float*) variances.data());

    struct stat st = {0};

    if (stat(data_dir.c_str(), &st) == -1) {
        mkdir(data_dir.c_str(), 0700);
    }
    if(stat((data_dir + "/meta").c_str(), &st) == -1){
        mkdir((data_dir + "/meta").c_str(), 0700);
    }
    size_t accuracy_bins_shape[1] = {(size_t) (sizeof(accuracy_bins)/sizeof(accuracy_bins[0]))};
    size_t bin_slack_shape[1] = {(size_t) (sizeof(bin_slack)/sizeof(bin_slack[0]))};
    npy::SaveArrayAsNumpy(data_dir + "/meta" + std::string("/accuracy_bins.npy"), false, 1, accuracy_bins_shape, accuracy_bins);
    npy::SaveArrayAsNumpy(data_dir + "/meta" + std::string("/bin_slack.npy"), false, 1, bin_slack_shape, bin_slack);

    float* robot = (float*) malloc(sizeof(float)*4*2);
    Position* positions = (Position*) malloc(sizeof(Position) * num_batch);
    float* pose_idxs = (float*) malloc(sizeof(float) * num_batch);
    float* var_idxs = (float*) malloc(sizeof(float) * num_batch);
    float* cp = (float*) malloc(sizeof(float) * num_batch);
    
    float* d_accuracy_bins;
    float* d_bin_slack;
    float* d_robot; 
    Pose* d_poses; 
    StdDev* d_std_devs; 
    Position* d_positions; 
    float* d_pose_idxs; 
    float* d_var_idxs; 
    float* d_cp; 
    int* d_done; 

    cudaMalloc(&d_accuracy_bins, sizeof(float)*(N_ACCURACY_BINS+1));
    cudaMalloc(&d_bin_slack, sizeof(float)*(N_ACCURACY_BINS+1));
    cudaMalloc(&d_robot, sizeof(float)*(4*2));
    cudaMalloc(&d_poses, sizeof(Pose)*NUM_POSES);
    cudaMalloc(&d_std_devs, sizeof(StdDev)*NUM_VARIANCES);
    cudaMalloc(&d_positions, sizeof(Position)*num_batch);
    cudaMalloc(&d_pose_idxs, sizeof(float)*(num_batch));
    cudaMalloc(&d_var_idxs, sizeof(float)*(num_batch));
    cudaMalloc(&d_cp, sizeof(float)*(num_batch));
    cudaMalloc(&d_done, sizeof(int)*(num_batch));

    DeviceZipIterator d_iter(thrust::make_tuple(thrust::device_pointer_cast(d_positions), 
                                        thrust::device_pointer_cast(d_cp),
                                        thrust::device_pointer_cast(d_var_idxs),
                                        thrust::device_pointer_cast(d_pose_idxs)));


    std::vector<float> dataset(num_batch*5);

    curandState *devStates;
    cudaMalloc((void **)&devStates, num_batch *  sizeof(curandState));

    dim3 threadsPerBlock(THREADS);
    dim3 numBlocks((int) std::max(1.0, ceil(num_batch/threadsPerBlock.x)));  

    // Initialize array
    create_rect(robot, R_WIDTH, R_HEIGHT);

    // Transfer data from host to device memory
    cudaMemcpy(d_robot, robot, sizeof(float)*(4*2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_poses, poses, sizeof(Pose)*NUM_POSES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_std_devs, std_devs, sizeof(Variance)*NUM_VARIANCES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_accuracy_bins, accuracy_bins, sizeof(float)*(N_ACCURACY_BINS+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bin_slack, bin_slack, sizeof(float)*(N_ACCURACY_BINS+1), cudaMemcpyHostToDevice);
    // std::srand(std::time(0));
    CUDA_CALL(cudaPeekAtLastError());

    setup_kernel<<<numBlocks, THREADS>>>(devStates, std::rand());
    CUDA_CALL(cudaPeekAtLastError());

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::cout << "Total number of configurations: " << num_batch * num_batches << std::endl;
    std::cout << "Begin computation..." << std::endl;
    int counter = 0;
    printf("batches generated: %i/%i", counter, num_batches);

    for (int batch_index = 0; batch_index < num_batches; batch_index++)
    {
        int num_left = num_batch;
        int batch_done = 0;
        int iteration = 0;
        int n_samples = 0;
        int n_batch = 0;
        while(num_left > 0 && n_samples < N){
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
                d_bin_slack,
                d_done,
                iteration,
                n_samples,
                n_batch,
                num_left,
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

        for (int j = 0; j < num_batch; j++)
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
        size_t ds_shape[2] = {(size_t) num_batch, (size_t) 5};

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
    cudaFree(d_bin_slack);
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
