import random
from app.database import db

def run_flow_simulation():
    # Reset activations
    db.execute_query("MATCH (n:CoreNode) SET n.activations = 0")
    
    # Initialize
    activations = {f"{record['n.system']}-{record['n.node']}": 0 
                  for record in db.execute_query("MATCH (n:CoreNode) RETURN n.system, n.node")}
    current_node = "system_1-N0"
    path = [current_node]
    
    # Update first node
    system, node = current_node.split('-')
    db.execute_query(
        "MATCH (n:CoreNode {system: $system, node: $node}) SET n.activations = coalesce(n.activations, 0) + 1",
        {"system": system, "node": node}
    )
    activations[current_node] += 1
    
    while not all(count >= 2 for count in activations.values()):
        # Determine next node based on flow rules
        if node == 'N0':
            next_node = 'blueprint-Centroid'
        elif node == 'N1':
            next_node = f"{system}-N2"
        elif node == 'N3':
            next_node = 'system_2-N0' if system == 'system_1' else 'system_1-N0'
        elif current_node == 'blueprint-Centroid':
            next_node = random.choice(['system_1-N1', 'system_2-N1'])
        else:  # Default progression
            next_node = f"{system}-N3"
        
        # Update activations
        system, node = next_node.split('-') if '-' in next_node else ('blueprint', 'Centroid')
        db.execute_query(
            "MATCH (n:CoreNode {system: $system, node: $node}) SET n.activations = coalesce(n.activations, 0) + 1",
            {"system": system, "node": node}
        )
        activations[next_node] += 1
        path.append(next_node)
        current_node = next_node
    
    return {"path": path, "activations": activations}