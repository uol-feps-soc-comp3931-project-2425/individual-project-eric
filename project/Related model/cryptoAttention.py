import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches



plt.figure(figsize=(10, 7.5))
ax = plt.subplot(111)


btc_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
eth_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.55, 0.65, 0.75])
sol_weights = np.array([0.15, 0.25, 0.4, 0.6, 0.7, 0.8, 0.9])

attention_matrix = np.vstack([btc_weights, eth_weights, sol_weights])

time_points = ['t-30', 't-25', 't-20', 't-15', 't-10', 't-5', 't-1']
cryptocurrencies = ['BTC', 'ETH', 'SOL']

colors = ["#d4e8ff", "#a5d1ff", "#76baff", "#4692e9", "#2a7dd0", "#1068b7", "#00539e"]
cmap = LinearSegmentedColormap.from_list("blue_gradient", colors)

im = ax.imshow(attention_matrix, cmap=cmap, aspect='auto')

ax.set_xticks(np.arange(len(time_points)))
ax.set_xticklabels(time_points)

ax.set_yticks(np.arange(len(cryptocurrencies)))
ax.set_yticklabels(cryptocurrencies)

ax.set_xticks(np.arange(-.5, len(time_points), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(cryptocurrencies), 1), minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
ax.tick_params(which="minor", size=0)

ax.text(5, 2, "5.2%", ha="center", va="center", color="white", fontweight="bold")
ax.text(6, 2, "5.4%", ha="center", va="center", color="white", fontweight="bold")

rect = patches.Rectangle((3, 1.5), 2, 1, linewidth=2, edgecolor='#ff5722', 
                         facecolor='none', linestyle='--')
ax.add_patch(rect)
ax.text(4, 2.3, "5-15 min lag period", ha="center", va="center", 
        color="#ff5722", fontweight="bold")

plt.title('Cross-Cryptocurrency Attention Distribution - BTC Price Prediction', fontsize=14, pad=20)
plt.suptitle('Analysis of Attention Weights Across Assets in Transformer Model', 
             fontsize=10, style='italic', y=0.92)
plt.xlabel('Time Points (minutes)', fontsize=12, labelpad=10)
plt.ylabel('Cryptocurrency', fontsize=12, labelpad=10)

cbar = plt.colorbar(im, ticks=[0.1, 0.5, 0.9])
cbar.ax.set_yticklabels(['Low Attention', 'Medium', 'High Attention'])

legend_elements = [patches.Patch(edgecolor='#ff5722', facecolor='none', 
                                 linestyle='--', label='Strongest attention region')]
ax.legend(handles=legend_elements, loc='upper center', 
          bbox_to_anchor=(0.5, -0.15))

plt.tight_layout()
plt.subplots_adjust(top=0.88)

plt.savefig('cross_crypto_attention.png', dpi=300, bbox_inches='tight')

plt.show()
