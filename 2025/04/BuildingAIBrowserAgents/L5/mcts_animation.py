import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import HTML
from gridworld import dfs

def compare_exploration_weights(all_results, grid, exploration_weights, animate=False):
    """
    Compare the final results of MCTS with different exploration weights
    
    Args:
        all_results: List of lists of MCTSResult objects for different exploration weights
        grid: The original grid layout
        exploration_weights: List of exploration weights used
        animate: Whether to create an animation of the exploration process
    
    Returns:
        If animate is True, returns an HTML object with the animation
        Otherwise, displays the final state plots
    """
    height, width = len(grid), len(grid[0])
    n_weights = len(exploration_weights)
    
    # Change to vertical layout (one below the other)
    fig, axes = plt.subplots(n_weights, 2, figsize=(12, 4*n_weights))
    if n_weights == 1:
        axes = [axes]
    
    # Find start and end cells
    start_cell = None
    end_cell = None
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 3:
                end_cell = (r, c)
            elif grid[r][c] == 2:
                start_cell = (r, c)
    
    # Get all iterations for plotting rewards
    all_iters = []
    if all_results and all_results[0]:
        all_iters = list(range(len(all_results[0])))
    
    # Initialize lines and heatmap images for animation
    reward_lines = []
    heatmap_images = []
    wall_patches = []
    path_lines = []
    
    for i, (results, weight, ax_row) in enumerate(zip(all_results, exploration_weights, axes)):
        # Left plot: Reward vs Iteration
        reward_ax = ax_row[0]
        # Create empty line for rewards
        line, = reward_ax.plot([], [], 'r-', label=f'Exploration Weight = {weight:.2f}')
        reward_lines.append(line)
        
        reward_ax.set_title(f"Iteration vs Cumulative Reward (w={weight:.2f})")
        reward_ax.set_xlabel("Iteration")
        reward_ax.set_ylabel("Cumulative Reward")
        reward_ax.set_xlim(0, len(all_iters)-1 if all_iters else 100)
        reward_ax.set_ylim(0, 1)
        reward_ax.legend()
        
        # Right plot: Heatmap of Q-values
        heatmap_ax = ax_row[1]
        
        # Initialize empty heatmap
        empty_matrix = np.zeros((height, width))
        img = heatmap_ax.imshow(empty_matrix, cmap="viridis", vmin=0, vmax=1)
        heatmap_images.append(img)
        
        # Add wall patches
        row_wall_patches = []
        for r in range(height):
            for c in range(width):
                cell_value = grid[r][c]
                if cell_value == 1:  # Wall
                    # Add black rectangle for walls
                    rect = plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=True, color='black')
                    heatmap_ax.add_patch(rect)
                    row_wall_patches.append(rect)
                    # Add 'W' text for walls
                    heatmap_ax.text(c, r, 'W', ha='center', va='center', color='white')
        wall_patches.append(row_wall_patches)
        
        # Mark start and end positions with text
        if start_cell:
            heatmap_ax.text(start_cell[1], start_cell[0], "S", 
                         ha='center', va='center', color="black", fontsize=16)
        if end_cell:
            heatmap_ax.text(end_cell[1], end_cell[0], "E", 
                         ha='center', va='center', color="black", fontsize=16)
        
        # Initialize empty path line
        path_line, = heatmap_ax.plot([], [], 'r-', linewidth=2)
        path_lines.append(path_line)
        
        heatmap_ax.set_title(f"Q-values of Each Cell (w={weight:.2f})")
        heatmap_ax.set_ylabel("Row Number")
        heatmap_ax.set_xlabel("Column Number")
        heatmap_ax.set_xticks(np.arange(width))
        heatmap_ax.set_yticks(np.arange(height))
    
    # Add colorbar to each heatmap plot
    for i in range(len(heatmap_images)):
        plt.colorbar(heatmap_images[i], ax=axes[i][1], label='Normalized Q-value')
    
    plt.tight_layout()
    fig.suptitle("Agent Trajectories in Gridworld", fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    if not animate:
        # Just show the final state
        for i, (results, weight) in enumerate(zip(all_results, exploration_weights)):
            # Update reward line with final data
            rewards = [max(result.cum_reward, 0) for result in results]
            reward_lines[i].set_data(all_iters, rewards)
            
            # Update heatmap with final data
            final_result = results[-1] if results else None
            visit_matrix = np.zeros((height, width))
            if final_result and final_result.tree_state:
                dfs(final_result.tree_state, visit_matrix)
            
            # Normalize
            if np.max(visit_matrix) > 0:
                visit_matrix = visit_matrix / np.max(visit_matrix)
            
            heatmap_images[i].set_array(visit_matrix)
            
            # Update path if available
            if final_result and final_result.trace and final_result.trace[0]:
                path_x = [state.position[1] for state in final_result.trace[0]]
                path_y = [state.position[0] for state in final_result.trace[0]]
                path_lines[i].set_data(path_x, path_y)
        
        plt.show()
        return None
    else:
        # Animation update function
        def update(frame):
            updated_artists = []
            
            for i, (results, weight) in enumerate(zip(all_results, exploration_weights)):
                # Update reward line
                current_frame = min(frame, len(results)-1)
                current_iters = all_iters[:current_frame+1]
                current_rewards = [max(result.cum_reward, 0) for result in results[:current_frame+1]]
                reward_lines[i].set_data(current_iters, current_rewards)
                updated_artists.append(reward_lines[i])
                 
                # Update heatmap
                current_result = results[current_frame]
                visit_matrix = np.zeros((height, width))
                if current_result.tree_state:
                    dfs(current_result.tree_state, visit_matrix)
                
                # Normalize
                if np.max(visit_matrix) > 0:
                    visit_matrix = visit_matrix / np.max(visit_matrix)
                
                heatmap_images[i].set_array(visit_matrix)
                updated_artists.append(heatmap_images[i])
                
                # Update path if available
                if current_result.trace and current_result.trace[0]:
                    path_x = [state.position[1] for state in current_result.trace[0]]
                    path_y = [state.position[0] for state in current_result.trace[0]]
                    path_lines[i].set_data(path_x, path_y)
                else:
                    path_lines[i].set_data([], [])
                updated_artists.append(path_lines[i])
            
            return updated_artists
        
        # Create animation
        frames = max([len(results) for results in all_results]) if all_results else 1
        ani = FuncAnimation(fig, update, frames=frames, interval=200, blit=True)
        
        # Return HTML for display in notebook
        html_video = ani.to_jshtml()
        plt.close()
        return HTML(html_video)