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
    for i in range(a.shape[0]):
        copy(b[i], a[i])
        
@cuda.jit
def monte_carlo_sample_collision(state, obstacles, num_obstacles, std_dev, collision_probability, rng_states):
    sample_obstacle = cuda.local.array(shape=(4, 2), dtype=float32)
    collisions = cuda.shared.array(shape=(num_samples), dtype=float32)
    idx = cuda.grid(1)
    collisions[idx] = 0
    
    dx = 0.
    dy = 0.
    dt = 0.
    
    # generate world
    for i in range(num_obstacles):
        dx = xoroshiro128p_normal_float32(rng_states, idx) * std_dev[i, 0]
        dy = xoroshiro128p_normal_float32(rng_states, idx) * std_dev[i, 1]
        dt = xoroshiro128p_normal_float32(rng_states, idx) * std_dev[i, 2]

        rot_trans_rectangle(obstacles[i], sample_obstacle, dx, dy, dt)
        
        if convex_collide(state, sample_obstacle) == 1:
            collisions[idx] = 1

    num_collisions = 0.
    cuda.syncthreads()
    if idx == 0:
        for i in range(num_samples):
            num_collisions += collisions[i]
        collision_probability[0] = num_collisions / num_samples


collide = np.ones(1, dtype=np.int32)
offset = np.array([[0.9, 1]])
theta = np.pi / 3
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)

r1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
r2 = r1 @ R + offset

threads_per_block = 1024
blocks = 1

rng_states = create_xoroshiro128p_states(threads_per_block*blocks, seed=1)

a = np.ones((2,5), dtype=np.int32)
b = np.zeros((2,5), dtype=np.int32)

state = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
obstacles = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32)
num_obstacles = len(obstacles)
std_dev = np.array([[np.sqrt(0.3), np.sqrt(0.5), np.sqrt(1)]], dtype=np.float32)
num_samples = threads_per_block
collision_probability = np.zeros(1, dtype=np.float32)

monte_carlo_sample_collision[(1),(threads_per_block)](state, obstacles, num_obstacles, std_dev, collision_probability, rng_states)
cuda.synchronize()
print(collision_probability)
