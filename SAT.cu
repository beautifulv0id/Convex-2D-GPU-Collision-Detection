#include <stdio.h>
#include <string>
#include <iostream> 
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

#define N 100000

#define R_WIDTH 4.07
#define R_HEIGHT 1.74

#define BORDER 6.0
#define SPATIAL_RESOLUTION 256
#define THETA_RESOLUTION 64

#define NUM_OBSTACLES 2
#define NUM_VARIANCES 2

#define BATCH_SIZE SPATIAL_RESOLUTION*SPATIAL_RESOLUTION*THETA_RESOLUTION
#define SIZE BATCH_SIZE*NUM_OBSTACLES*NUM_VARIANCES

#define NUM_POSES THETA_RESOLUTION*NUM_OBSTACLES

#define THREADS 1024
#define BLOCKS (int) ceil(BATCH_SIZE/(float) THREADS)

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets different seed, a different sequence
       number, no offset */
    curand_init(7+id, id, 0, &state[id]);
}

__device__ __host__ float idx_to_pos(int idx){
    return idx * (2.0*BORDER) / SPATIAL_RESOLUTION - BORDER + BORDER / SPATIAL_RESOLUTION;
}

__device__ __host__ float idx_to_theta(int idx){
    return idx * 2.0 * M_PI / THETA_RESOLUTION;
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


__global__ void monte_carlo_sample_collision_dataset(float* cp, float* ro, float* rr_g, float* std_dev, curandState *state, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int xidx = index / (THETA_RESOLUTION * SPATIAL_RESOLUTION);
    int yidx = (index - (xidx * (THETA_RESOLUTION * SPATIAL_RESOLUTION))) / THETA_RESOLUTION;
    int tidx = index % THETA_RESOLUTION;

    float x = idx_to_pos(xidx);
    float y = idx_to_pos(yidx);
    float theta = idx_to_theta(tidx);

    curandState* localState = &state[index];

    float sr[8];
    float rr[8];
    memcpy(rr, rr_g, sizeof(float) * 8);
    rot_trans_rectangle(rr, x, y, theta);


    int collisions = 0;
    for(int s = 0; s < n; s++){
        sample_rectangle(ro, sr, std_dev, localState);
        collisions += convex_collide(rr, sr);
    }
    cp[index] = (float) collisions / (float) n;
    // printf("%i, %i,  %f, %f, %f\n", index, collisions, x, y, theta);
}

__device__ int grid_to_cp_index(int x, int y){
    return x*THETA_RESOLUTION * SPATIAL_RESOLUTION + y * THETA_RESOLUTION;
}

__global__ void compute_normals(float* cp, float* normals){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < 1 || x >= SPATIAL_RESOLUTION - 1 || y < 1 || y > SPATIAL_RESOLUTION - 1){
        return;
    }

    int index = grid_to_cp_index(x,y);

    int l = grid_to_cp_index(x-1, y);
    int r = grid_to_cp_index(x+1, y);
    int t = grid_to_cp_index(x, y+1);
    int b = grid_to_cp_index(x, y-1);

    float h = 2*(2*BORDER/SPATIAL_RESOLUTION);

    float nx = (cp[r] - cp[l]) / h;
    float ny = (cp[t] - cp[b]) / h;

    float norm = sqrtf(nx*nx+ny*ny);

    normals[2*index] = 0;
    normals[2*index+1] = 0;

    if(norm >= 1e-5)
    {
        normals[2*index] = nx/norm;
        normals[2*index+1] = ny/norm;
    }
}

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

    CUDA_CALL(cudaSetDevice(nDevices-2));

    std::cout << "Allocate data..." << std::endl;

    // float obstacle_dims[NUM_OBSTACLES][2] = {{4.07, 1.74}, {4.9, 1.86}, {2.1, 1.56}};
    // float variances[NUM_VARIANCES][5] = {{0.05, 0.15, 0.8, 0.0, 0.0}, {0.05, 0.05, 0.03, 0.0, 0.0}, {0.10, 0.10, 0.08, 0.0, 0.0}, {0.20, 0.20, 0.01, 0.0, 0.0}};
    float obstacle_dims[NUM_OBSTACLES][2] = {{4.07, 1.74}, {4.9, 1.86}};
    float variances[NUM_VARIANCES][5] = {{0.05, 0.05, 0.03, 0.0, 0.0}, {0.20, 0.20, 0.01, 0.0, 0.0}};

    std::vector<float> collision_probabilities(SIZE);
    std::vector<float> normals(SIZE*2);
    
    thrust::host_vector<float> ro(4*2);
    thrust::host_vector<float> rr(4*2);
    thrust::host_vector<float> std_dev(5);
    std::vector<thrust::host_vector<float>> normals_batch(NUM_OBSTACLES*NUM_VARIANCES, thrust::host_vector<float>(BATCH_SIZE*2));
    std::vector<thrust::host_vector<float>> cp_batch(NUM_OBSTACLES*NUM_VARIANCES, thrust::host_vector<float>(BATCH_SIZE));

    thrust::device_vector<float> d_ro(4*2);
    thrust::device_vector<float> d_rr(4*2);
    thrust::device_vector<float> d_std_dev(5);
    thrust::device_vector<float> d_normals(BATCH_SIZE*2);
    thrust::device_vector<float> d_cp(BATCH_SIZE);

    curandState *devStates;
    cudaMalloc((void **)&devStates, BATCH_SIZE *  sizeof(curandState));

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((int) ceil(THETA_RESOLUTION*SPATIAL_RESOLUTION/threadsPerBlock.x),  /* for instance 512/8 = 64*/
              (int) ceil(SPATIAL_RESOLUTION/threadsPerBlock.y));  

    // Initialize array
    create_rect(rr.data(), R_WIDTH, R_HEIGHT);
    

    // Transfer data from host to device memory
    d_rr = rr;
    setup_kernel<<<BLOCKS, THREADS>>>(devStates);
    CUDA_CALL(cudaPeekAtLastError());

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int counter = 0;
    std::cout << "Total number of configurations: " << SIZE << std::endl;
    std::cout << "Begin computation..." << std::endl;
    printf("poses generated: %i/%i", counter, NUM_OBSTACLES*NUM_VARIANCES);
    fflush(stdout); 

    for(int oidx = 0; oidx < NUM_OBSTACLES; oidx++){
        for(int vidx = 0; vidx < NUM_VARIANCES; vidx++){
            create_rect(ro.data(), obstacle_dims[oidx][0], obstacle_dims[oidx][1]);
            std_dev[0] = sqrtf(variances[vidx][0]);
            std_dev[1] = sqrtf(variances[vidx][1]);
            std_dev[2] = sqrtf(variances[vidx][2]);
            d_std_dev = std_dev;
            d_ro = ro;

            monte_carlo_sample_collision_dataset<<<BLOCKS, THREADS>>>
                                    (   thrust::raw_pointer_cast(d_cp.data()), 
                                        thrust::raw_pointer_cast(d_ro.data()), 
                                        thrust::raw_pointer_cast(d_rr.data()), 
                                        thrust::raw_pointer_cast(d_std_dev.data()), 
                                        devStates,
                                        N);
            CUDA_CALL(cudaDeviceSynchronize());
            compute_normals<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(d_cp.data()), 
                                    thrust::raw_pointer_cast(d_normals.data()) );
            CUDA_CALL(cudaPeekAtLastError());
            cp_batch[counter] = d_cp;
            normals_batch[counter] = d_normals;
            printf("\33[2K\r");
            printf("progress: %i/%i", ++counter, NUM_OBSTACLES*NUM_VARIANCES);
            fflush(stdout); 
        }   
    }

    printf("\33[2K\r");
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Finished computation" << std::endl;
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " [ms]" << std::endl;


    std::cout << "Writing dataset..." << std::endl;
    counter = 0;
    for(int i = 0; i < NUM_OBSTACLES*NUM_VARIANCES; i++){
            memcpy(&collision_probabilities[i*BATCH_SIZE], thrust::raw_pointer_cast(cp_batch[i].data()), BATCH_SIZE*sizeof(float));
            memcpy(&normals[i*BATCH_SIZE*2], thrust::raw_pointer_cast(normals_batch[i].data()), 2*BATCH_SIZE*sizeof(float));
    }

    // allocate arrays for dataset
    std::vector<float> dataset(SIZE*8);
    std::vector<float> poses(NUM_POSES*3);

    // write data
    float* d = dataset.data();
    float* c = collision_probabilities.data();
    float* n = normals.data();

    for(int oidx = 0; oidx < NUM_OBSTACLES; oidx++){
        for(int vidx = 0; vidx < NUM_VARIANCES; vidx++){
            for(int xidx = 0; xidx < SPATIAL_RESOLUTION; xidx++){
                for(int yidx = 0; yidx < SPATIAL_RESOLUTION; yidx++){
                    for(int tidx = 0; tidx < THETA_RESOLUTION; tidx++){
                        d[0] = idx_to_pos(xidx);  // x
                        d[1] = idx_to_pos(yidx);  // y
                        d[2] = n[0]; // dx
                        d[3] = n[1]; // dy
                        d[4] = *c;  // cp
                        d[5] = vidx; // var_idx
                        d[6] = oidx*THETA_RESOLUTION+tidx; // pose_idx
                        d[7] = 1.0; // weight
                        n+=2;
                        c+=1;
                        d+=8;
                    }
                }      
            }
        }
    }
    // write poses
    float* p = poses.data();
    for(int oidx = 0; oidx < NUM_OBSTACLES; oidx++){
        for(int tidx = 0; tidx < THETA_RESOLUTION; tidx++){
            p[0] = obstacle_dims[oidx][0];
            p[1] = obstacle_dims[oidx][1];
            p[2] = idx_to_theta(tidx);
            p+=3;
        }      
    }

    // write dataset
    size_t ds_shape[2] = {(size_t) SIZE, (size_t) 8};
    size_t poses_shape[2] = {(size_t) NUM_POSES, (size_t) 3};
    size_t variances_shape[2] = {(size_t) NUM_VARIANCES, (size_t) 5};

    npy::SaveArrayAsNumpy("data/0.npy", false, 2, ds_shape, dataset);
    npy::SaveArrayAsNumpy("data/poses.npy", false, 2, poses_shape, poses);
    npy::SaveArrayAsNumpy("data/variances.npy", false, 2, variances_shape, (float*) &variances);
    // free memory
    cudaFree(devStates);
    std::cout << "Done." << std::endl;
}
