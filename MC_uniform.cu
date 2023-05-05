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

#define N 4000 // numer of monte carlo smaples per iteration
#define MAX_RESAMPLE 1000
#define R_WIDTH 4.07 // width of the robot
#define R_HEIGHT 1.74 // height of the robot

#define NUM_BATCH 16777216 // number of configurations that are generated per batch
#define NUM_BATCHES 100 // number of batches 
#define NUM_DATA_POINTS double(NUM_BATCH)*NUM_BATCHES

#define NUM_POSES 64*64*64 // number of poses that are sampled, a pose contains the width, height of the obstacle and angle theta of the robot
#define NUM_VARIANCES 64*64*64*4 // number of variances that are sampled, 

#define POS_MIN -6.0 // minimium x-, y-position of the robot
#define POS_MAX 6.0 // maximum x-, y-position of the robot

#define OBSTACLE_WIDTH_MIN 0.1 // minimum width, height of obstacles
#define OBSTACLE_WIDTH_MAX 5.0 // maximum width, height of obstacles

#define VAR_MIN 0.001 // minimum positional and rotational variance 
#define VAR_MAX 0.3 // maximum positional and rotational variance 

#define N_ACCURACY_BINS 3

#define THREADS 1024
#define BLOCKS (int) ceil(NUM_BATCH/(float) THREADS)

#define DATA_DIR "data"

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
    float cp_out;
    if(iteration == 0){
        // pos.x = curand_normal(localState) * 3.9/1.96;
        // pos.y = curand_normal(localState) * 3.9/1.96;
        pos.x = POS_MIN + curand_uniform(localState) * (POS_MAX-POS_MIN);
        pos.y = POS_MIN + curand_uniform(localState) * (POS_MAX-POS_MIN);
        pose_idx = curand(localState) % NUM_POSES;
        std_dev_idx = curand(localState) % NUM_VARIANCES;
    } else {
        n_samplestrue = (int) cps[gidx];
        pos = positions[gidx];
        pose_idx = pose_idxs[gidx];
        std_dev_idx = std_dev_idxs[gidx];
    }

    Pose pose = poses[pose_idx];
    StdDev std_dev = std_devs[std_dev_idx];

    float obstacle[8];
    create_rect(obstacle, pose.width, pose.height);    
    float sampled_obstacle[8];
    float robot[8];

    memcpy(robot, robot_base, sizeof(float) * 8);
    rot_trans_rectangle(robot, pos.x, pos.y, pose.theta);

    for (int i = 0; i < N; i++)
    {
        sample_rectangle(obstacle, sampled_obstacle, std_dev, localState);
        n_samplestrue += convex_collide(robot, sampled_obstacle);
    }
    int n_samples = N * (iteration+1);
    float slack = calcSlack(n_samples, n_samplestrue);

    float p = (float) n_samplestrue / (float) n_samples;
    int d = 0;
    if(iteration+1 == MAX_RESAMPLE){
        printf("n_samplestrue %i, n_samples %i, %f\n", n_samplestrue, n_samples, slack);
        d = 1;
    }
    else {
        for (int i = 0; i < N_ACCURACY_BINS; i++){
            // a bit hacky, but actually results in accuracy_bins[i] <=  p < accuracy_bins[i+1] (which is what we want)
            if(p >= accuracy_bins[i] && p <= accuracy_bins[i+1] && slack < bin_slack[i]){
                d = 1;
            }
        }
    }

    cp_out = n_samplestrue;
    if(d){
        cp_out = p;
    }

    done[gidx] = d;
    cps[gidx] = cp_out;
    positions[gidx] = pos;
    pose_idxs[gidx] = pose_idx;
    std_dev_idxs[gidx] = std_dev_idx;
}

void print_help(int* done, int* d_done, int* index, int* d_index){
    cudaMemcpy(done, d_done, sizeof(int) * NUM_BATCH, cudaMemcpyDeviceToHost);
    cudaMemcpy(index, d_index, sizeof(int) * NUM_BATCH, cudaMemcpyDeviceToHost);
    printf("done:  ");
    for (size_t i = 0; i < NUM_BATCH; i++)
    {
        printf("%2d ", done[i]);
    }
    printf("\n");
    printf("index: ");
    for (size_t i = 0; i < NUM_BATCH; i++)
    {
        printf("%2d ", index[i]);
    }
    printf("\n");
}

int main(){
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

    write_config();

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
    npy::SaveArrayAsNumpy(DATA_DIR + std::string("/poses.npy"), false, 2, poses_shape, (float*) poses);
    npy::SaveArrayAsNumpy(DATA_DIR + std::string("/variances.npy"), false, 2, variances_shape, (float*) variances.data());

    float accuracy_bins[4] = {0, 0.01, 0.1, 1.};
    float bin_slack[4] = {0.0001, 0.005, 0.01, 0};

    float* robot = (float*) malloc(sizeof(float)*4*2);
    Position* positions = (Position*) malloc(sizeof(Position) * NUM_BATCH);
    float* pose_idxs = (float*) malloc(sizeof(float) * NUM_BATCH);
    float* var_idxs = (float*) malloc(sizeof(float) * NUM_BATCH);
    float* cp = (float*) malloc(sizeof(float) * NUM_BATCH);
    
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

    cudaMalloc(&d_accuracy_bins, sizeof(float)*(4));
    cudaMalloc(&d_bin_slack, sizeof(float)*(4));
    cudaMalloc(&d_robot, sizeof(float)*(4*2));
    cudaMalloc(&d_poses, sizeof(Pose)*NUM_POSES);
    cudaMalloc(&d_std_devs, sizeof(StdDev)*NUM_VARIANCES);
    cudaMalloc(&d_positions, sizeof(Position)*NUM_BATCH);
    cudaMalloc(&d_pose_idxs, sizeof(float)*(NUM_BATCH));
    cudaMalloc(&d_var_idxs, sizeof(float)*(NUM_BATCH));
    cudaMalloc(&d_cp, sizeof(float)*(NUM_BATCH));
    cudaMalloc(&d_done, sizeof(int)*(NUM_BATCH));

    DeviceZipIterator d_iter(thrust::make_tuple(thrust::device_pointer_cast(d_positions), 
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
    cudaMemcpy(d_poses, poses, sizeof(Pose)*NUM_POSES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_std_devs, std_devs, sizeof(Variance)*NUM_VARIANCES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_accuracy_bins, accuracy_bins, sizeof(float)*(4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bin_slack, bin_slack, sizeof(float)*(4), cudaMemcpyHostToDevice);
    setup_kernel<<<BLOCKS, THREADS>>>(devStates);
    CUDA_CALL(cudaPeekAtLastError());

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::cout << "Total number of configurations: " << NUM_BATCH << std::endl;
    std::cout << "Begin computation..." << std::endl;
    int counter = 0;
    printf("batches generated: %i/%i\n", counter, NUM_BATCHES);

    for (int batch_index = 0; batch_index < NUM_BATCHES; batch_index++)
    {
        int num_left = NUM_BATCH;
        int batch_done = 0;
        int iteration = 0;
        while(num_left != 0 && iteration < MAX_RESAMPLE){
            numBlocks = ((int) ceil((float) num_left/threadsPerBlock.x));  
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
                cudaMemcpy(positions + num_left, d_positions + num_left, sizeof(Position) * batch_done, cudaMemcpyDeviceToHost);
                cudaMemcpy(cp + num_left, d_cp + num_left, sizeof(float) * batch_done, cudaMemcpyDeviceToHost);
                cudaMemcpy(var_idxs + num_left, d_var_idxs + num_left, sizeof(float) * batch_done, cudaMemcpyDeviceToHost);
                cudaMemcpy(pose_idxs + num_left, d_pose_idxs + num_left, sizeof(float) * batch_done, cudaMemcpyDeviceToHost);
            }
            iteration++;
            printf("num left %i\n", num_left);
        }

        if(num_left > 0){
            printf("copying remaining %i over (0, %i)\n", num_left, num_left);
            cudaMemcpy(positions, d_positions, sizeof(Position) * num_left, cudaMemcpyDeviceToHost);
            cudaMemcpy(cp, d_cp, sizeof(float) * num_left, cudaMemcpyDeviceToHost);
            cudaMemcpy(var_idxs, d_var_idxs, sizeof(float) * num_left, cudaMemcpyDeviceToHost);
            cudaMemcpy(pose_idxs, d_pose_idxs, sizeof(float) * num_left, cudaMemcpyDeviceToHost); 
        }


        CUDA_CALL(cudaDeviceSynchronize());
        printf("\33[2K\r");
        printf("batches generated: %i/%i", ++counter, NUM_BATCHES);
        printf("num left %i", num_left);
        fflush(stdout); 

        // write data
        float* d = dataset.data();
        float* c = cp;

        for (int j = 0; j < NUM_BATCH; j++)
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
        size_t ds_shape[2] = {(size_t) NUM_BATCH, (size_t) 5};

        npy::SaveArrayAsNumpy(DATA_DIR + std::string("/") + std::to_string(batch_index) + ".npy", false, 2, ds_shape, dataset);
    }
    std::cout << std::endl;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Finished computation" << std::endl;
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " [ms]" << std::endl;

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
