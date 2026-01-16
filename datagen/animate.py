import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = np.load("runs/20260116_160921_jepamd/val_states.npz")   
positions = data["pos"]      # (T, N, dim)

T, N, dim = positions.shape
assert dim == 2

box_size = 10.0

fig, ax = plt.subplots()
scat = ax.scatter(positions[0, :, 0], positions[0, :, 1], s=50)

ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_aspect("equal")
ax.set_title("Lennard-Jones MD trajectory")


def update(frame):
    scat.set_offsets(positions[frame])
    ax.set_title(f"t = {frame}")
    return scat,

ani = FuncAnimation(
    fig,
    update,
    frames=T,
    interval=30,  
    blit=True
)

plt.show()
