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

#define N 10000 // numer of monte carlo smaples

#define R_WIDTH 4.07 // width of the robot
#define R_HEIGHT 1.74 // height of the robot

#define NUM_BATCH 16777216 // number of configurations that are generated per batch
#define NUM_BATCHES 10 // number of batches 
#define NUM_DATA_POINTS NUM_BATCH*NUM_BATCHES

#define NUM_POSES 64*64 // number of poses that are sampled, a pose contains the width, height of the obstacle and angle theta of the robot
#define NUM_VARIANCES 64*64 // number of variances that are sampled, 

#define POS_MIN -6.0 // minimium x-, y-position of the robot
#define POS_MAX 6.0 // maximum x-, y-position of the robot

#define OBSTACLE_WIDTH_MIN 2.0 // minimum width, height of obstacles
#define OBSTACLE_WIDTH_MAX 5.0 // maximum width, height of obstacles

#define VAR_MIN 0.001 // minimum positional and rotational variance 
#define VAR_MAX 0.3 // maximum positional and rotational variance 

#define THREADS 1024
#define BLOCKS (int) ceil(NUM_BATCH/(float) THREADS)

#define DATA_DIR "data"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)


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



__global__ void monte_carlo_sample_collision_dataset_uniform(float* robot_base, float* poses, float* std_devs, float* pose_idxs, float* std_dev_idxs, float* positions, float* cp, curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= NUM_BATCH)
        return;
    
    float x = curand_uniform(state) * POS_MIN;
    float y = curand_uniform(state) * POS_MAX;

    int pose_idx = curand(state) % NUM_POSES;
    float w = poses[(pose_idx)*3];
    float h = poses[(pose_idx)*3+1];
    float t = poses[(pose_idx)*3+2];

    int std_dev_idx = curand(state) % NUM_VARIANCES;
    float std_dev[3];
    for (int i = 0; i < 3; i++)
        std_dev[i] = std_devs[std_dev_idx*2+i];

    curandState* localState = &state[idx];

    float obstacle[8];
    create_rect(obstacle, w, h);    
    float sampled_obstacle[8];
    float robot[8];

    memcpy(robot, robot_base, sizeof(float) * 8);
    rot_trans_rectangle(robot, x, y, t);

    int collisions = 0;
    for (int i = 0; i < N; i++)
    {
        sample_rectangle(obstacle, sampled_obstacle, (float*) &std_dev, localState);
        collisions += convex_collide(robot, sampled_obstacle);
    }
    cp[idx] = (float) collisions / (float) N;
    positions[idx*2] = x;
    positions[idx*2+1] = y;
    pose_idxs[idx] = pose_idx;
    std_dev_idxs[idx] = std_dev_idx;
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

    thrust::host_vector<float> poses(NUM_POSES*3);
    thrust::host_vector<float> std_devs(NUM_VARIANCES*3);  
    std::vector<float> variances(NUM_VARIANCES*3);  

    std::default_random_engine generator;
    auto obstacle_uniform = std::uniform_real_distribution<float>(OBSTACLE_WIDTH_MIN, OBSTACLE_WIDTH_MAX);
    auto obstacle_scale_uniform = std::uniform_real_distribution<float>(0.25, 0.75);
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
        variances[3*i+0] = variance_uniform(generator);
        variances[3*i+1] = variance_uniform(generator);
        variances[3*i+2] = variance_uniform(generator);       
        std_devs[3*i+0] = sqrt(variances[3*i+0]);
        std_devs[3*i+1] = sqrt(variances[3*i+1]);
        std_devs[3*i+2] = sqrt(variances[3*i+2]);  
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
    npy::SaveArrayAsNumpy(DATA_DIR + std::string("/poses.npy"), false, 2, poses_shape, thrust::raw_pointer_cast(poses.data()));
    npy::SaveArrayAsNumpy(DATA_DIR + std::string("/variances.npy"), false, 2, variances_shape, variances);


    thrust::host_vector<float> robot(4*2);
    thrust::host_vector<float> positions(NUM_BATCH*2);
    thrust::host_vector<float> pose_idxs(NUM_BATCH);
    thrust::host_vector<float> var_idxs(NUM_BATCH);
    // thrust::host_vector<float> normals(NUM_BATCH*2);
    thrust::host_vector<float> cp(NUM_BATCH);
    
    thrust::device_vector<float> d_robot(4*2);
    thrust::device_vector<float> d_poses(NUM_POSES*3);
    thrust::device_vector<float> d_std_devs(NUM_VARIANCES*3);

    thrust::device_vector<float> d_positions(NUM_BATCH*2);
    thrust::device_vector<float> d_pose_idxs(NUM_BATCH);
    thrust::device_vector<float> d_var_idxs(NUM_BATCH);
    // thrust::device_vector<float> d_normals(NUM_BATCH*2);
    thrust::device_vector<float> d_cp(NUM_BATCH);

    std::vector<float> dataset(NUM_BATCH*8);

    curandState *devStates;
    cudaMalloc((void **)&devStates, NUM_BATCH *  sizeof(curandState));

    dim3 threadsPerBlock(1024);
    dim3 numBlocks((int) ceil(NUM_BATCH/threadsPerBlock.x));  

    // Initialize array
    create_rect(robot.data(), R_WIDTH, R_HEIGHT);

    // Transfer data from host to device memory
    d_robot = robot;
    setup_kernel<<<BLOCKS, THREADS>>>(devStates);
    CUDA_CALL(cudaPeekAtLastError());
    d_poses = poses;
    d_std_devs = std_devs;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::cout << "Total number of configurations: " << NUM_BATCH << std::endl;
    std::cout << "Begin computation..." << std::endl;
    int counter = 0;
    printf("batches generated: %i/%i", counter, NUM_BATCHES);
    fflush(stdout); 

    for (int i = 0; i < NUM_BATCHES; i++)
    {
        monte_carlo_sample_collision_dataset_uniform<<<BLOCKS, THREADS>>>(
            thrust::raw_pointer_cast(d_robot.data()),
            thrust::raw_pointer_cast(d_poses.data()),
            thrust::raw_pointer_cast(d_std_devs.data()),
            thrust::raw_pointer_cast(d_pose_idxs.data()),
            thrust::raw_pointer_cast(d_var_idxs.data()),
            thrust::raw_pointer_cast(d_positions.data()),
            thrust::raw_pointer_cast(d_cp.data()),
            devStates
        );
        CUDA_CALL(cudaDeviceSynchronize());
        printf("\33[2K\r");
        printf("batches generated: %i/%i", ++counter, NUM_BATCHES);
        fflush(stdout); 

        positions = d_positions;
        cp = d_cp;
        var_idxs = d_var_idxs;
        pose_idxs = d_pose_idxs;

        // write data
        float* d = dataset.data();
        float* c = cp.data();
        // float* n = normals.data();

        for (int j = 0; j < NUM_BATCH; j++)
        {
            d[0] = positions[j*2];  // x
            d[1] = positions[j*2+1];  // y
            d[2] = 0.0; // dx
            d[3] = 0.0; // dy
            d[4] = *c;  // cp
            d[5] = var_idxs[j]; // var_idx
            d[6] = pose_idxs[j]; // pose_idx
            d[7] = 1.0; // weight
            // n+=2;
            c+=1;
            d+=8;
        }

        // write dataset
        size_t ds_shape[2] = {(size_t) NUM_BATCH, (size_t) 8};

        npy::SaveArrayAsNumpy(DATA_DIR + std::string("/") + std::to_string(i) + ".npy", false, 2, ds_shape, dataset);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Finished computation" << std::endl;
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " [ms]" << std::endl;

    // free memory
    cudaFree(devStates);
    std::cout << "Done." << std::endl;
}
