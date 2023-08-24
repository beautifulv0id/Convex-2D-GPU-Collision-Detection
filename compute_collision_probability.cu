#include <stdio.h>
#include <string>
#include <iostream> 
#include <fstream>
#include <cmath>
#include <math.h>
#include <random>
#include <chrono>
#include<ctime>
#include <nlohmann/json.hpp>
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

#define UNIFORM_SAMPLING 0

#define N 4000000

// SIMULATION
#define R_WIDTH 4.07 // width of the robot
#define R_HEIGHT 1.74 // height of the robot

// TIAGO
// #define R_WIDTH 0.75 // width of the robot
// #define R_HEIGHT 0.8 // height of the robot


#define R_OFFSET ((R_WIDTH + R_HEIGHT) / 4)
#define POS_MIN -12.0 // minimium x-, y-position of the robot
#define POS_MAX 12.0 // maximum x-, y-position of the robot

#define OBSTACLE_WIDTH_MIN 0.1 // minimum width, height of obstacles
#define OBSTACLE_WIDTH_MAX 5.0 // maximum width, height of obstacles

#define VAR_MIN 0.0 // minimum positional and rotational variance 
#define VAR_MAX 0.3 // maximum positional and rotational variance 

#define N_ACCURACY_BINS 4

#define THREADS 512

float accuracy_bins[N_ACCURACY_BINS+1] = {0, 0.001, 0.01, 0.1, 1};
float bin_slack[N_ACCURACY_BINS+1] = {0.00005,0.0005, 0.001, 0.01, 0};

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
    __host__ __device__ Position(float x, float y) : x(x), y(y) {}
    __host__ __device__ Position() : x(0), y(0) {}
};

struct Variance
{
    float x, y, theta, width, height;
    __host__ __device__ Variance(float x, float y, float theta, float width, float height) : x(x), y(y), theta(theta), width(width), height(height) {}
    __host__ __device__ Variance() : x(0), y(0), theta(0), width(0), height(0) {}
};

struct Pose
{
    float width, height, theta;
    __host__ __device__ Pose(float width, float height, float theta) : width(width), height(height), theta(theta) {}
    __host__ __device__ Pose() : width(0), height(0), theta(0) {}
};


typedef Variance StdDev;
typedef thrust::device_vector<Position>::iterator   DeviceFloatPairIterator;
typedef thrust::device_vector<float>::iterator   DeviceFloatIterator;
typedef thrust::device_vector<int>::iterator   DeviceIntIterator;
typedef thrust::tuple<DeviceFloatPairIterator, DeviceFloatIterator, DeviceFloatIterator, DeviceFloatIterator, DeviceIntIterator> DeviceIteratorTuple;
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
    pos = positions[gidx];
    if(iteration == 0){
        pose_idxs[gidx] = gidx;
        std_dev_idxs[gidx] = gidx;
        std_dev = std_devs[gidx];
        pose = poses[gidx];
    }
    else {
        n_samplestrue = (int) cps[gidx];
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
    // int n_samples = N * (iteration+1);
    float slack = calcSlack(n_samples, n_samplestrue);

    float p = (float) n_samplestrue / (float) n_samples;
    int d = 0;
    for (int i = 0; i < N_ACCURACY_BINS; i++){
        // a bit hacky, but actually results in accuracy_bins[i] <=  p < accuracy_bins[i+1] (which is what we want)
        if(p >= accuracy_bins[i] && p <= accuracy_bins[i+1] && slack <= bin_slack[i]){
            d = 1;
        }
    }
    done[gidx] = d;
    cps[gidx] = n_samplestrue;
}

int main(int argc, char* argv[])
{   
    std::string data_dir = "data";
    if(argc > 1){
        data_dir = std::string(argv[1]);
    }
    
    std::vector<float> positionsLoader;
    std::vector<float> posesLoader;
    std::vector<float> variancesLoader;

    std::vector<npy::ndarray_len_t> poses_shape;
    std::vector<npy::ndarray_len_t> variances_shape;
    std::vector<npy::ndarray_len_t> positions_shape;

    npy::LoadArrayFromNumpy(data_dir + std::string("/poses.npy"), poses_shape, posesLoader);
    npy::LoadArrayFromNumpy(data_dir + std::string("/variances.npy"), variances_shape, variancesLoader);
    npy::LoadArrayFromNumpy(data_dir + std::string("/positions.npy"), positions_shape, positionsLoader);

    int num_data_points = poses_shape[0];
    int* indices = (int*) malloc(sizeof(int) * num_data_points);
    Position* positions = (Position*) malloc(sizeof(Position) * num_data_points);
    StdDev* std_devs = (Variance*) malloc(sizeof(StdDev)*num_data_points);  
    std::vector<Pose> poses;
    std::vector<Variance> variances;  

    for (int i = 0; i < num_data_points; i++)
    {
        positions[i] = Position(positionsLoader[i*2], positionsLoader[i*2+1]);
        poses.push_back(Pose(posesLoader[i*3], posesLoader[i*3+1], posesLoader[i*3+2]));
        variances.push_back(Variance(variancesLoader[i*5], variancesLoader[i*5+1], variancesLoader[i*5+2], variancesLoader[i*5+3], variancesLoader[i*5+4]));
        std_devs[i] = StdDev(sqrt(variances[i].x), sqrt(variances[i].y), sqrt(variances[i].theta), sqrt(variances[i].width), sqrt(variances[i].height));
        indices[i] = i;
    }

    float* robot = (float*) malloc(sizeof(float)*4*2);
    
    float* pose_idxs = (float*) malloc(sizeof(float) * num_data_points);
    float* var_idxs = (float*) malloc(sizeof(float) * num_data_points);
    float* cp = (float*) malloc(sizeof(float) * num_data_points);
    
    int* d_indices;
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

    cudaMalloc(&d_indices, sizeof(int)*num_data_points);
    cudaMalloc(&d_accuracy_bins, sizeof(float)*(N_ACCURACY_BINS+1));
    cudaMalloc(&d_bin_slack, sizeof(float)*(N_ACCURACY_BINS+1));
    cudaMalloc(&d_robot, sizeof(float)*(4*2));
    cudaMalloc(&d_poses, sizeof(Pose)*num_data_points);
    cudaMalloc(&d_std_devs, sizeof(StdDev)*num_data_points);
    cudaMalloc(&d_positions, sizeof(Position)*num_data_points);
    cudaMalloc(&d_pose_idxs, sizeof(float)*(num_data_points));
    cudaMalloc(&d_var_idxs, sizeof(float)*(num_data_points));
    cudaMalloc(&d_cp, sizeof(float)*(num_data_points));
    cudaMalloc(&d_done, sizeof(int)*(num_data_points));

    DeviceZipIterator d_iter(thrust::make_tuple(
                                        thrust::device_pointer_cast(d_positions), 
                                        thrust::device_pointer_cast(d_cp),
                                        thrust::device_pointer_cast(d_var_idxs),
                                        thrust::device_pointer_cast(d_pose_idxs), 
                                        thrust::device_pointer_cast(d_indices)
                                        ));


    curandState *devStates;
    cudaMalloc((void **)&devStates,  num_data_points*sizeof(curandState));

    dim3 threadsPerBlock(THREADS);
    dim3 numBlocks((int) ceil((float) num_data_points/threadsPerBlock.x));  

    // Initialize array
    create_rect(robot, R_WIDTH, R_HEIGHT);

    // Transfer data from host to device memory
    cudaMemcpy(d_indices, indices, sizeof(int)*num_data_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_robot, robot, sizeof(float)*(4*2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_poses, (float*) poses.data(), sizeof(Pose)*num_data_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_std_devs, std_devs, sizeof(Variance)*num_data_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, (float*) positions, sizeof(Position)*num_data_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_accuracy_bins, accuracy_bins, sizeof(float)*(N_ACCURACY_BINS+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bin_slack, bin_slack, sizeof(float)*(N_ACCURACY_BINS+1), cudaMemcpyHostToDevice);
    std::srand(std::time(0));
    numBlocks = ((int) ceil((float) num_data_points/threadsPerBlock.x));
    setup_kernel<<<numBlocks, THREADS>>>(devStates, std::rand());
    CUDA_CALL(cudaPeekAtLastError());


    int num_left = num_data_points;
    int batch_done = 0;
    int iteration = 0;
    int n_samples = 0;
    int n_batch = 0;
    while(num_left != 0 && n_samples < N){
        numBlocks = ((int) ceil((float) num_left/threadsPerBlock.x));  
        if(n_samples < 20000)
            n_batch = 1000;
        else
            n_batch = 10000;
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
    }


    if(num_left > 0){
        numBlocks = ((int) ceil((float) num_left/threadsPerBlock.x));  
        write_collision_probability<<<numBlocks, threadsPerBlock>>>(d_cp, num_left, n_samples);
        gpuErrchk( cudaPeekAtLastError() );
        cudaMemcpy(positions, d_positions, sizeof(Position) * num_left, cudaMemcpyDeviceToHost);
        cudaMemcpy(cp, d_cp, sizeof(float) * num_left, cudaMemcpyDeviceToHost);
        cudaMemcpy(var_idxs, d_var_idxs, sizeof(float) * num_left, cudaMemcpyDeviceToHost);
        cudaMemcpy(pose_idxs, d_pose_idxs, sizeof(float) * num_left, cudaMemcpyDeviceToHost); 
    }

    CUDA_CALL(cudaDeviceSynchronize());
    cudaMemcpy(indices, d_indices, sizeof(int)*num_data_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(positions, d_positions, sizeof(Position) * num_data_points, cudaMemcpyDeviceToHost);

    std::vector<float> cp_sorted(num_data_points);    
    for(int i = 0; i < num_data_points; i++){
        cp_sorted[indices[i]] = cp[i];
    }

    // write dataset
    size_t cp_shape[1] = {(size_t) num_data_points};
    npy::SaveArrayAsNumpy(data_dir + std::string("/") + "cp.npy", false, 1, cp_shape, cp_sorted);
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
}
