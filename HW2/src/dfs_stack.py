import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
    """
    I first use csv.reader to read the file and convert it into a list of rows of data. For each row of data, I store it in a two-dimensional dictionary representing the graph, 
    where the keys are the starting nodes and the values are dictionaries containing the adjacent nodes and their distances.

    The DFS algorithm starts by pushing the start node onto the stack. Then, in each iteration, it pops the top element from the stack and explores its neighbors. 
    If a neighbor has not been visited yet, it is pushed onto the top of the stack. This process continues until either the stack is empty or the algorithm detects the end node.

    Once the end node is detected, the algorithm finishes. Using the information stored in the parent dictionary, 
    the algorithm reconstructs the path from the start node to the end node. 
    Additionally, it computes the total distance of the path by summing up the distances between consecutive nodes along the path.

    Overall, the DFS algorithm explores the graph in a depth-first manner, prioritizing exploration of nodes that are deeper in the graph before backtracking. 
    This can be useful for certain types of graph traversal problems, such as finding paths or cycles.
    """
    edges = {}
    with open(edgeFile, newline='') as csvfile:
        rows = csv.reader(csvfile)
        next(rows)  # Skip header
        for row in rows:
            start_node, end_node = map(int, row[:2])
            distance = float(row[2])  # Convert distance to float
            if start_node not in edges:
                edges[start_node] = {}
            edges[start_node][end_node] = distance

    visited = set()
    stack = [start]  # Stack to keep track of nodes to visit
    parent = {start: None}
    path = []
    dist = 0
    num_visited = 0

    while stack:
        current = stack.pop()  # Pop the top node from the stack
        num_visited += 1
        if current == end:
            # Reconstruct the path
            node = end
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            # Compute distance
            for i in range(len(path) - 1):
                dist += edges[path[i]][path[i+1]]
            return path, dist, num_visited

        visited.add(current)

        # Push unvisited neighbors onto the stack
        for neighbor, distance in edges.get(current, {}).items():
            if neighbor not in visited:
                stack.append(neighbor)
                parent[neighbor] = current

    # If end node is not reachable
    return [], 0, num_visited
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
