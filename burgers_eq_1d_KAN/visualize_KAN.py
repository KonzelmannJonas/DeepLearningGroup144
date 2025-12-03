# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_network_architecture():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot 1: MLP Architecture ---
    ax = axes[0]
    ax.set_title("MLP (Multi-Layer Perceptron)", fontsize=16, pad=20)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Nodes coordinates
    layer_x = [0.5, 1.5, 2.5]
    layer_y = [[0.5, 1.5, 2.5], [0.5, 1.5, 2.5], [1.5]]
    
    # Draw Edges (Weights)
    for i, x_start in enumerate(layer_x[:-1]):
        for y_start in layer_y[i]:
            for y_end in layer_y[i+1]:
                ax.plot([x_start, layer_x[i+1]], [y_start, y_end], 'gray', linewidth=1, zorder=1)
                # Annotation for weight
                if i == 0 and y_start == 1.5 and y_end == 1.5:
                    ax.text(1.0, 1.6, r'$w_{ij}$', fontsize=12, color='black', ha='center')

    # Draw Nodes (Activations)
    for x_coord, y_list in zip(layer_x, layer_y):
        for y_coord in y_list:
            circle = patches.Circle((x_coord, y_coord), 0.15, facecolor='lightblue', edgecolor='black', zorder=2)
            ax.add_patch(circle)
            ax.text(x_coord, y_coord, r'$\sigma$', fontsize=14, ha='center', va='center', zorder=3)

    ax.text(1.5, -0.2, "Activació Fixa als Nodes\nPesos Lineals a les Arestes", ha='center', fontsize=12, style='italic')

    # --- Plot 2: KAN Architecture ---
    ax = axes[1]
    ax.set_title("KAN (Kolmogorov-Arnold Network)", fontsize=16, pad=20)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Draw Edges (Learnable Functions)
    for i, x_start in enumerate(layer_x[:-1]):
        for y_start in layer_y[i]:
            for y_end in layer_y[i+1]:
                # Draw a curvy line to represent a function
                ax.plot([x_start, layer_x[i+1]], [y_start, y_end], 'green', linewidth=2, zorder=1)
                
                # Draw simplified function symbol on edge
                if i == 0 and y_start == 1.5 and y_end == 1.5:
                     # Draw a mini sine wave box to symbolize function
                    rect = patches.Rectangle((0.9, 1.4), 0.2, 0.2, facecolor='white', edgecolor='green', zorder=2)
                    ax.add_patch(rect)
                    ax.plot([0.92, 0.96, 1.0, 1.04, 1.08], [1.5, 1.55, 1.5, 1.45, 1.5], 'green', linewidth=1, zorder=3)
                    ax.text(1.0, 1.7, r'$\phi(x)$', fontsize=12, color='green', ha='center')

    # Draw Nodes (Summation)
    for x_coord, y_list in zip(layer_x, layer_y):
        for y_coord in y_list:
            circle = patches.Circle((x_coord, y_coord), 0.15, facecolor='lightgreen', edgecolor='black', zorder=2)
            ax.add_patch(circle)
            ax.text(x_coord, y_coord, r'$\Sigma$', fontsize=14, ha='center', va='center', zorder=3)

    ax.text(1.5, -0.2, "Suma Simple als Nodes\nFuncions Aprenibles a les Arestes", ha='center', fontsize=12, style='italic')

    plt.tight_layout()
    plt.savefig('mlp_vs_kan_architecture.png')
    plt.show()

# %% <-- Opcional, per separar blocs
# Cridem la funció per generar el gràfic
draw_network_architecture()
# %%
