import igraph as ig 
import plotly.graph_objects as go

def plot_tree(num_vertices, node_url, node_message, node_action, node_weight, node_critic_response, node_q_value, edges):
    G = ig.Graph()
    G.add_vertices(num_vertices)
    G.add_edges(edges)

    lay = G.layout_reingold_tilford(root=[0])
    position = {i: lay[i] for i in range(num_vertices)}
    Xn = [position[k][0] for k in range(num_vertices)]
    Yn = [-position[k][1] for k in range(num_vertices)]

    Xe = []
    Ye = []
    for edge in edges:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]  # X-coordinates
        Ye += [-position[edge[0]][1], -position[edge[1]][1], None]  # Y-coordinates

    hover_text = []
    for n in range(num_vertices):
        text = f"<b>Node {n}</b><br>"
        if n in node_url:
            url_text = node_url[n]
            if len(url_text) > 100:
                url_text = url_text[:100] + "..."
            text += f"<b>URL:</b> {url_text}<br>"
        if n in node_q_value:
            text += f"<b>Q-value:</b> {node_q_value[n]}<br>"
        if n in node_message:
            msg = node_message[n]
            if len(msg) > 100:
                msg = msg[:100] + "..."
            text += f"<b>Message:</b> {msg}<br>"
        if n in node_action:
            action_text = str(node_action[n])
            if len(action_text) > 100:
                action_text = action_text[:100] + "..."
            text += f"<b>Action:</b> {action_text}<br>"
        if n in node_weight:
            text += f"<b>Weight:</b> {node_weight[n]}<br>"
        if n in node_critic_response:
            resp = node_critic_response[n]
            if len(resp) > 100:
                resp = resp[:100] + "..."
            text += f"<b>Critic Response:</b> {resp}<br>"
        hover_text.append(text)

    fig = go.Figure()

    # Draw edges
    fig.add_trace(go.Scatter(
        x=Xe, y=Ye, mode="lines",
        line=dict(color="gray", width=1), hoverinfo="none"
    ))

    # Create a color scale based on Q-values
    node_colors = []
    for n in range(num_vertices):
        q_val = node_q_value.get(n, 0)
        # Map Q-values to a color scale (blue to red)
        node_colors.append(q_val)
    
    # Normalize colors if we have values
    if node_colors and max(node_colors) != min(node_colors):
        node_colors = [(c - min(node_colors)) / (max(node_colors) - min(node_colors)) for c in node_colors]
    
    fig.add_trace(go.Scatter(
        x=Xn, y=Yn, mode="markers+text",
        marker=dict(
            symbol="circle-dot", 
            size=18, 
            color=node_colors,
            colorscale="Viridis",
            line=dict(color="black", width=1)
        ),
        text=[str(n) for n in range(num_vertices)],
        textposition="middle center",
        hovertext=hover_text,
        hoverinfo="text"
    ))

    fig.update_layout(
        title="MCTS Tree Visualization",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=600,
        width=900
    )

    return fig



def dfs_browser_nodes(current_node, urls, messages, actions, weights, critic_responses, AL, EL, Q, objectives, rewards, is_terminals, completed_tasks):
    if hasattr(current_node, 'state') and current_node.state:  # Store basic node information
        urls[current_node.id] = current_node.state.url if hasattr(current_node.state, 'url') else "No URL"
       
        if hasattr(current_node.state, 'objective'):  # Store objective if available
            objectives[current_node.id] = current_node.state.objective
        
        if hasattr(current_node.state, 'completed_tasks'):  # Store completed tasks if available
            completed_tasks[current_node.id] = current_node.state.completed_tasks
    
    if hasattr(current_node, 'agent_message'):  # Store agent messages, actions, weights, and critic responses if available
        messages[current_node.id] = current_node.agent_message
    if hasattr(current_node, 'action'):
        actions[current_node.id] = current_node.action
    if hasattr(current_node, 'weight'):
        weights[current_node.id] = current_node.weight
    
    Q[current_node.id] = current_node.Q if hasattr(current_node, 'Q') else 0  # Store Q-values, rewards, and terminal state info
    if hasattr(current_node, 'reward'):
        rewards[current_node.id] = current_node.reward
    if hasattr(current_node, 'is_terminal'):
        is_terminals[current_node.id] = current_node.is_terminal
    
    if hasattr(current_node, 'is_terminal') and not current_node.is_terminal and hasattr(current_node, 'children') and current_node.children:
        for child in current_node.children:
            EL.append((current_node.id, child.id))
            if current_node.id not in AL:
                AL[current_node.id] = []
            AL[current_node.id].append(child.id)
            dfs_browser_nodes(child, urls, messages, actions, weights, critic_responses, AL, EL, Q, objectives, rewards, is_terminals, completed_tasks)