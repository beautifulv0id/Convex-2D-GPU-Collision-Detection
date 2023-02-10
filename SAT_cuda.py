from numba import vectorize
import numpy as np
from numba import njit
from numba import cuda
from PIL import Image
from numba import cuda, float32
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32

tpb = 1

@cuda.jit('float32(float32[:], int32)', device=True)
def _max(a, n):
    loc_max = a[0]
    for i in range(1, n):
        loc_max = max(loc_max, a[i])
    return loc_max

@cuda.jit('float32(float32[:], int32)', device=True)
def _min(a, n):
    loc_min = a[0]
    for i in range(1, n):
        loc_min = min(loc_min, a[i])
    return loc_min

@cuda.jit
def convex_collide(R1, R2):
    norm = cuda.local.array(shape=(2), dtype=float32)
    p1 = cuda.local.array(shape=(4), dtype=float32)
    p2 = cuda.local.array(shape=(4), dtype=float32)
    min1 = 1.
    min2 = 1.
    max1 = 1.
    max2 = 1.
        
    for i in range(4):
        norm[0] = R1[(i+1)%4,0] - R1[i,0]
        norm[1] = R1[(i+1)%4,1] - R1[i,1]
        for k in range(4):
            p1[k] = norm[0]*R1[k,0]+norm[1]*R1[k,1]
            p2[k] = norm[0]*R2[k,0]+norm[1]*R2[k,1]
        
        min1, max1 = _min(p1, 4), _max(p1, 4)
        min2, max2 = _min(p2,4), _max(p2,4)
        if max1 < min2 or max2 < min1:
            return 0
            
    
    for i in range(4):
        norm[0] = R1[(i+1)%4,0] - R1[i,0]
        norm[1] = R1[(i+1)%4,1] - R1[i,1]
        for k in range(4):
            p1[k] = norm[0]*R1[k,0]+norm[1]*R1[k,1]
            p2[k] = norm[0]*R2[k,0]+norm[1]*R2[k,1]
        
        min1, max1 = _min(p1, 4), _max(p1, 4)
        min2, max2 = _min(p2,4), _max(p2,4)
        if max1 < min2 or max2 < min1:
            return 0
    
    return 1
  
@cuda.jit          
def rot_trans_rectangle(rin, rout, dx, dy, dt):
    c = math.cos(dt)
    s = math.sin(dt)
    for i in range(4):
        x = rin[i,0]
        y = rin[i,1]
        rout[i,0] = c*x-s*y + dx
        rout[i,1] = s*x+c*y + dy
        
@cuda.jit
def copy(dest, source):
    for i, v in enumerate(source):
        dest[i] = v   

@cuda.jit
def test(a, b):
    i = cuda.grid(1)
    print(i)
    # for i in range(a.shape[0]):
    #     copy(b[i], a[i])
        
@cuda.jit
def monte_carlo_sample_collision(state, obstacles, num_obstacles, std_dev, count, rng_states):
    gidx = cuda.grid(1)
    if gidx >= num_samples:
        return
    tidx = cuda.threadIdx.x
    bidx = cuda.blockIdx.x
    sample_obstacle = cuda.local.array(shape=(4, 2), dtype=float32)
    local_count = cuda.shared.array(shape=(threads_per_block), dtype=float32)
    local_count[tidx] = 0.
    
    # generate world
    for i in range(num_obstacles):
        dx = xoroshiro128p_normal_float32(rng_states, gidx) * std_dev[i, 0]
        dy = xoroshiro128p_normal_float32(rng_states, gidx) * std_dev[i, 1]
        dt = xoroshiro128p_normal_float32(rng_states, gidx) * std_dev[i, 2]

        rot_trans_rectangle(obstacles[i], sample_obstacle, dx, dy, dt)
        
        if convex_collide(state, sample_obstacle) == 1:
            local_count[tidx] = 1

    num_collisions = 0.
    cuda.syncthreads()
    if tidx == 0:
        for i in range(threads_per_block):
            num_collisions += local_count[i]
        count[bidx] = num_collisions


num_samples = 10000
threads_per_block = 1024
blocks = (num_samples + (threads_per_block - 1)) // threads_per_block
print(blocks, threads_per_block)
rng_states = create_xoroshiro128p_states(threads_per_block*blocks, seed=1)

# a = np.ones((2,5), dtype=np.int32)
# b = np.zeros((2,5), dtype=np.int32)

# test[2,1024](a,b)
# print(b)

state = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
obstacles = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32)
num_obstacles = len(obstacles)
std_dev = np.array([[np.sqrt(0.3), np.sqrt(0.5), np.sqrt(1)]], dtype=np.float32)
collision_count = np.zeros(blocks, dtype=np.float32)

monte_carlo_sample_collision[blocks,1024](state, obstacles, num_obstacles, std_dev, collision_count, rng_states)
collisions = np.sum(collision_count)
print(collision_count)
collision_probability = collisions / num_samples
print(collision_probability)
