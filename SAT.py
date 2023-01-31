import numpy as np
import matplotlib.pyplot as plt

def plot_rect(rect):
  plt.plot(rect[:,0].tolist() + [rect[0,0]], rect[:,1].tolist() + [rect[0,1]])


offset = np.array([[0.9, 1]])
theta = np.pi / 3
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

r1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
r2 = r1 @ R + offset

collide = True

for shape in [r1, r2]:
  for i in range(len(shape)):
    n = shape[(i+1)%len(shape)] - shape[i]
    n = n / np.linalg.norm(n)
    p1 = r1 @ n.T
    p2 = r2 @ n.T
    min1, max1 = np.min(p1), np.max(p1)
    min2, max2 = np.min(p2), np.max(p2)
    if max1 < min2 or max2 < min1:
      collide = False
      break

print(f"collide = {collide}")

fig, ax = plt.subplots(figsize=(6,6))
plot_rect(r1)
plot_rect(r2)
ax.set_aspect('equal')
fig.savefig("sat.png")
