import matplotlib.pyplot as plt
import numpy as np
import imageio


M=128

# Create a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
i=0
j=0
for zeta in [1,4]:
    j=0
    for h in [0.10, 0.30]:
        img = imageio.imread(f'./results/{M}/zeta_{zeta:1.1f}_h_{h:.2f}.png')
        axes[i,j].imshow(img)
        axes[i, j].axis('off')  # Turn off axis
        axes[i, j].text(0.5, -0.1, f'zeta: {zeta}, h:{h}', ha='center', va='top', transform=axes[i, j].transAxes, fontsize=12)
        j = j+1
    i=i+1




plt.tight_layout()

# Show the plot
plt.savefig(f'res_{M}.png')