#include <stdio.h>
#include <string>
#include <iostream> 
#include <fstream>
#include <cmath>
#include <math.h>
#include <random>
#include <chrono>

#include <npy.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>
#include <curand.h>

 /*
 This file creates a dataset of collision probabilities between a robot and an obstacle modeled as rectangles for different configurations using Monte Carlo sampling.
 One data-point defines the width, height and variance of the obstacle, as well as the position and orientation of the robot w.r.t the obstacle coordinate frame.
 First, NUM_POSES poses and NUM_VARIANCES variances are unifomely sampled from user-defined bounds. A pose contains the width and height of the obstacle and angle 
 theta of the robot. Variances are defined for the position and orientation of the obstacle. Finally, for each data-point a robot position is uniformly sampled, 
 as well as a random pose and variance from the pregenerated poses and variances. 
 */

#define N 10000 // numer of monte carlo smaples per iteration
#define MAX_RESAMPLE 10
#define R_WIDTH 4.07 // width of the robot
#define R_HEIGHT 1.74 // height of the robot

// #define NUM_BATCH 16777216 // number of configurations that are generated per batch
// #define NUM_BATCHES 100 // number of batches 
#define NUM_BATCH 16777216 // number of configurations that are generated per batch
#define NUM_BATCHES 10 // number of batches 
#define NUM_DATA_POINTS double(NUM_BATCH)*NUM_BATCHES

#define NUM_POSES 64*64 // number of poses that are sampled, a pose contains the width, height of the obstacle and angle theta of the robot
#define NUM_VARIANCES 64*64 // number of variances that are sampled, 

#define POS_MIN -6.0 // minimium x-, y-position of the robot
#define POS_MAX 6.0 // maximum x-, y-position of the robot

#define OBSTACLE_WIDTH_MIN 1.0 // minimum width, height of obstacles
#define OBSTACLE_WIDTH_MAX 5.0 // maximum width, height of obstacles

#define VAR_MIN 0.001 // minimum positional and rotational variance 
#define VAR_MAX 0.3 // maximum positional and rotational variance 

#define THREADS 1024
#define BLOCKS (int) ceil(NUM_BATCH/(float) THREADS)

#define DATA_DIR "data"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

struct FloatPair
{
    float x, y;
};

typedef thrust::device_vector<FloatPair>::iterator   DeviceFloatPairIterator;
typedef thrust::device_vector<float>::iterator   DeviceFloatIterator;
typedef thrust::device_vector<int>::iterator   DeviceIntIterator;
typedef thrust::tuple<DeviceFloatPairIterator, DeviceFloatIterator, DeviceFloatIterator, DeviceFloatIterator> DeviceIteratorTuple;
typedef thrust::zip_iterator<DeviceIteratorTuple> DeviceZipIterator;

void write_config(){
    std::ofstream confFile;
    confFile.open (DATA_DIR + std::string("/config.txt"), std::ios::out);
    confFile << "N" << "\t" << N << "\n";
    confFile << "R_WIDTH" << "\t" << R_WIDTH << "\n";
    confFile << "R_HEIGHT" << "\t" << R_HEIGHT << "\n";
    confFile << "NUM_BATCH" << "\t" << NUM_BATCH << "\n";
    confFile << "NUM_BATCHES" << "\t" << NUM_BATCHES << "\n";
    confFile << "NUM_DATA_POINTS" << "\t" << NUM_DATA_POINTS << "\n";
    confFile << "NUM_POSES" << "\t" << NUM_POSES << "\n";
    confFile << "NUM_VARIANCES" << "\t" << NUM_VARIANCES << "\n";
    confFile << "POS_MIN" << "\t" << POS_MIN << "\n";
    confFile << "POS_MAX" << "\t" << POS_MAX << "\n";
    confFile << "OBSTACLE_WIDTH_MIN" << "\t" << OBSTACLE_WIDTH_MIN << "\n";
    confFile << "OBSTACLE_WIDTH_MAX" << "\t" << OBSTACLE_WIDTH_MAX << "\n";
    confFile << "VAR_MIN" << "\t" << VAR_MIN << "\n";
    confFile << "VAR_MAX" << "\t" << VAR_MAX << "\n";
    confFile.close();

}

struct invert_functor
{
  __host__ __device__
  void operator()(int& x)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    x = 1 - x;
  }
};

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets different seed, a different sequence
       number, no offset */
    curand_init(7+id, id, 0, &state[id]);
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

__device__ void sample_rectangle(float* r_in, float* r_out, float* std_dev, curandState* state)
{
    float dx = curand_normal(state) * std_dev[0];
    float dy = curand_normal(state) * std_dev[1]; 
    float dt = curand_normal(state) * std_dev[2];
    
    memcpy(r_out, r_in, sizeof(float) * 8);

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
    if(nsamples_true == nsamples){
        return log(1.0 / alpha) / nsamples;
    }
    else{
        return z / nsamples * sqrt((float) nsamples_true - nsamples_true * nsamples_true / (float) nsamples);
    }
}


__global__ void monte_carlo_sample_collision_dataset_uniform(float* robot_base, float* poses, float* std_devs, float* pose_idxs, float* std_dev_idxs, float* positions, float* cp, float* accuracy_bins, float* bin_slack, int* done, int iteration, int num_left, curandState* state) {
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(gidx >= num_left)
        return;

    curandState* localState = &state[gidx];

    int pose_idx;
    int std_dev_idx;
    float x, y;
    int collisions = 0;

    if(iteration == 0){
        x = curand_normal(localState) * 4/1.96;
        y = curand_normal(localState) * 4/1.96;
        pose_idx = curand(localState) % NUM_POSES;
        std_dev_idx = curand(localState) % NUM_VARIANCES;
    } else {
        collisions = (int) cp[gidx];
        x = positions[gidx*2];
        y = positions[gidx*2+1];
        pose_idx = pose_idxs[gidx];
        std_dev_idx = std_dev_idxs[gidx];
    }

    float w = poses[(pose_idx)*3];
    float h = poses[(pose_idx)*3+1];
    float t = poses[(pose_idx)*3+2];

    float std_dev[3];
    for (int i = 0; i < 3; i++)
        std_dev[i] = std_devs[std_dev_idx*3+i];

    float obstacle[8];
    create_rect(obstacle, w, h);    
    float sampled_obstacle[8];
    float robot[8];

    memcpy(robot, robot_base, sizeof(float) * 8);
    rot_trans_rectangle(robot, x, y, t);

    for (int i = 0; i < N; i++)
    {
        sample_rectangle(obstacle, sampled_obstacle, (float*) &std_dev, localState);
        collisions += convex_collide(robot, sampled_obstacle);
    }

    float slack = calcSlack(N, collisions);

    float p =  (float) collisions / (float) (N * iteration);
    int s = 0;
    for (int i = 0; i < 3; i++){
        if(p >= accuracy_bins[i] && p < accuracy_bins[i+1] && slack < bin_slack[i]){
            s = 1;
        }
    }

    done[gidx] = s;
    cp[gidx] = p;
    positions[gidx*2] = x;
    positions[gidx*2+1] = y;
    pose_idxs[gidx] = pose_idx;
    std_dev_idxs[gidx] = std_dev_idx;
}



int main(){
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    printf("Number of devices: %d\n", nDevices);

    for (int i = 0; i < nDevices; i++) {
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

    write_config();

    std::cout << "Allocate data..." << std::endl;

    float* poses = (float*) malloc(sizeof(float)*(NUM_POSES*3));
    float* std_devs = (float*) malloc(sizeof(float)*(NUM_VARIANCES*3));  
    std::vector<float> variances(NUM_VARIANCES*3);  

    std::default_random_engine generator;
    auto obstacle_uniform = std::uniform_real_distribution<float>(OBSTACLE_WIDTH_MIN, OBSTACLE_WIDTH_MAX);
    auto obstacle_scale_uniform = std::uniform_real_distribution<float>(0.25, 1.);
    auto theta_uniform = std::uniform_real_distribution<float>(0.0, 2.0*M_PI);
    auto variance_uniform = std::uniform_real_distribution<float>(VAR_MIN, VAR_MAX);
    
    for (int i = 0; i < NUM_POSES; i++)
    {
        float width = obstacle_uniform(generator);
        poses[3*i] = width;
        poses[3*i+1] = width*obstacle_scale_uniform(generator);
        poses[3*i+2] = theta_uniform(generator);
    }    
    for (int i = 0; i < NUM_VARIANCES; i++)
    {
        // variances[3*i+0] = variance_uniform(generator);
        // variances[3*i+1] = variance_uniform(generator);
        // variances[3*i+2] = variance_uniform(generator);       
        // std_devs[3*i+0] = sqrt(variances[3*i+0]);
        // std_devs[3*i+1] = sqrt(variances[3*i+1]);
        // std_devs[3*i+2] = sqrt(variances[3*i+2]);  
        float var = variance_uniform(generator);
        variances[3*i+0] = var;
        variances[3*i+1] = var;
        variances[3*i+2] = var;       
        std_devs[3*i+0] = sqrt(var);
        std_devs[3*i+1] = sqrt(var);
        std_devs[3*i+2] = sqrt(var);     

    }

    // write poses and variances
    size_t poses_shape[2] = {(size_t) NUM_POSES, (size_t) 3};
    size_t variances_shape[2] = {(size_t) NUM_VARIANCES, (size_t) 3};
    npy::SaveArrayAsNumpy(DATA_DIR + std::string("/poses.npy"), false, 2, poses_shape, poses);
    npy::SaveArrayAsNumpy(DATA_DIR + std::string("/variances.npy"), false, 2, variances_shape, variances);

    float accuracy_bins[4] = {0, 0.01, 0.1, 1.};
    float bin_slack[4] = {0.0001, 0.005, 0.01};

    float* robot = (float*) malloc(sizeof(float)*4*2);
    float* positions = (float*) malloc(sizeof(float) * NUM_BATCH*2);
    float* pose_idxs = (float*) malloc(sizeof(float) * NUM_BATCH);
    float* var_idxs = (float*) malloc(sizeof(float) * NUM_BATCH);
    float* cp = (float*) malloc(sizeof(float) * NUM_BATCH);
    
    float* d_accuracy_bins;
    float* d_bin_slack;
    float* d_robot; 
    float* d_poses; 
    float* d_std_devs; 
    float* d_positions; 
    float* d_pose_idxs; 
    float* d_var_idxs; 
    float* d_cp; 
    int* d_done; 

    cudaMalloc(&d_accuracy_bins, sizeof(float)*(4));
    cudaMalloc(&d_bin_slack, sizeof(float)*(4));
    cudaMalloc(&d_robot, sizeof(float)*(4*2));
    cudaMalloc(&d_poses, sizeof(float)*(NUM_POSES*3));
    cudaMalloc(&d_std_devs, sizeof(float)*(NUM_VARIANCES*3));
    cudaMalloc(&d_positions, sizeof(float)*(NUM_BATCH*2));
    cudaMalloc(&d_pose_idxs, sizeof(float)*(NUM_BATCH));
    cudaMalloc(&d_var_idxs, sizeof(float)*(NUM_BATCH));
    cudaMalloc(&d_cp, sizeof(float)*(NUM_BATCH));
    cudaMalloc(&d_done, sizeof(int)*(NUM_BATCH));

    DeviceZipIterator d_iter(thrust::make_tuple(thrust::device_pointer_cast((FloatPair*) d_positions), 
                                        thrust::device_pointer_cast(d_cp),
                                        thrust::device_pointer_cast(d_var_idxs),
                                        thrust::device_pointer_cast(d_pose_idxs)));


    std::vector<float> dataset(NUM_BATCH*5);

    curandState *devStates;
    cudaMalloc((void **)&devStates, NUM_BATCH *  sizeof(curandState));

    dim3 threadsPerBlock(1024);
    dim3 numBlocks((int) ceil(NUM_BATCH/threadsPerBlock.x));  

    // Initialize array
    create_rect(robot, R_WIDTH, R_HEIGHT);

    // Transfer data from host to device memory
    cudaMemcpy(d_robot, robot, sizeof(float)*(4*2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_poses, poses, sizeof(float)*(NUM_POSES*3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_std_devs, std_devs, sizeof(float)*(NUM_VARIANCES*3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_accuracy_bins, accuracy_bins, sizeof(float)*(4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bin_slack, bin_slack, sizeof(float)*(4), cudaMemcpyHostToDevice);
    setup_kernel<<<BLOCKS, THREADS>>>(devStates);
    CUDA_CALL(cudaPeekAtLastError());

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::cout << "Total number of configurations: " << NUM_BATCH << std::endl;
    std::cout << "Begin computation..." << std::endl;
    int counter = 0;
    printf("batches generated: %i/%i", counter, NUM_BATCHES);
    fflush(stdout); 

    for (int i = 0; i < NUM_BATCHES; i++)
    {
        int num_left = NUM_BATCH;
        int batch_done = 0;
        int iteration = 0;
        while(num_left != 0 && iteration < MAX_RESAMPLE){
            numBlocks = ((int) ceil(num_left/threadsPerBlock.x));  
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
                num_left,
                devStates
            );
            batch_done = thrust::count(thrust::device, thrust::device_pointer_cast(d_done), thrust::device_pointer_cast(d_done + num_left), 1);
            if(batch_done > 0){
                thrust::sort_by_key(thrust::device_pointer_cast(d_done), thrust::device_pointer_cast(d_done + num_left), d_iter);
                num_left -= batch_done;
                cudaMemcpy(positions + num_left, d_positions + num_left, sizeof(FloatPair) * batch_done, cudaMemcpyDeviceToHost);
                cudaMemcpy(cp + num_left, d_cp + num_left, sizeof(float) * batch_done, cudaMemcpyDeviceToHost);
                cudaMemcpy(var_idxs + num_left, d_var_idxs + num_left, sizeof(float) * batch_done, cudaMemcpyDeviceToHost);
                cudaMemcpy(pose_idxs + num_left, d_pose_idxs + num_left, sizeof(float) * batch_done, cudaMemcpyDeviceToHost);
            }
            printf("iteration %i, batch_done %i, left %i\n", iteration, batch_done, num_left);
            iteration++;
        }

        CUDA_CALL(cudaDeviceSynchronize());
        printf("\33[2K\r");
        printf("batches generated: %i/%i", ++counter, NUM_BATCHES);
        fflush(stdout); 

        // write data
        float* d = dataset.data();
        float* c = cp;
        // float* n = normals.data();

        for (int j = 0; j < NUM_BATCH; j++)
        {
            d[0] = positions[j*2];  // x
            d[1] = positions[j*2+1];  // y
            d[2] = *c;  // cp
            d[3] = var_idxs[j]; // var_idx
            d[4] = pose_idxs[j]; // pose_idx
            c+=1;
            d+=5;
        }

        // write dataset
        size_t ds_shape[2] = {(size_t) NUM_BATCH, (size_t) 5};

        npy::SaveArrayAsNumpy(DATA_DIR + std::string("/") + std::to_string(i) + ".npy", false, 2, ds_shape, dataset);
    }
    std::cout << std::endl;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Finished computation" << std::endl;
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " [ms]" << std::endl;

    // free memory
    cudaFree(devStates);
    std::cout << "Done." << std::endl;
}
