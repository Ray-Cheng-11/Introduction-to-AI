import csv
import heapq
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def load_heuristic():
    """
    I use csv.reader to read the heuristic.csv file and convert it into a dictionary.
    For each row of data, I store the node ID and its heuristic value.
    The heuristic values represent the estimated cost from each node to the goal node.
    """
    heuristic = {}
    with open(heuristicFile, newline='') as csvfile:
        rows = csv.reader(csvfile)
        next(rows)  # Skip header
        for row in rows:
            node = int(row[0])
            h_value = float(row[1])
            heuristic[node] = h_value
    return heuristic


def astar(start, end):
    # begin your code (Part4)
    """
    I first read the edge data from the edges.csv file and construct the graph.
    Then, I load the heuristic values from the heuristic.csv file using the load_heuristic function.
    I use a priority_queue to implement the A* algorithm, where each element contains the heuristic value,
    node ID, ID of the node's parent, and true distance from the start node.
    The heuristic value is calculated using the straight distance to the destination plus the true distance from the start point.
    For each iteration, I extract the most prioritized element (the one with the least heuristic value) from the priority_queue.
    I explore the neighbors of the current node, updating their true distances if a shorter path is found.
    I then calculate the new heuristic value for each neighbor and add it to the priority queue if it's not already visited.
    The algorithm continues until the priority queue is empty or the end node is detected.
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

    heuristic_values = load_heuristic()

    # Priority queue to store nodes based on cumulative distance + heuristic
    priority_queue = [(0, start)]
    parent = {start: None}
    # Dictionary to store the cumulative distance for each node
    distances = {start: 0}
    path = []
    num_visited = 0

    while priority_queue:
        # Pop node with minimum cumulative distance + heuristic
        _, current = heapq.heappop(priority_queue)
        num_visited += 1
        if current == end:
            # Reconstruct the path
            node = end
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            # Total distance is the cumulative distance to the end node
            dist = distances[end]
            return path, dist, num_visited

        # Explore neighbors of the current node
        for neighbor, distance in edges.get(current, {}).items():
            cumulative_distance_to_neighbor = distances[current] + distance
            # Update the distance if a shorter path to the neighbor is found
            if neighbor not in distances or cumulative_distance_to_neighbor < distances[neighbor]:
                distances[neighbor] = cumulative_distance_to_neighbor
                priority = cumulative_distance_to_neighbor + \
                    heuristic_values.get(neighbor, 0)
                heapq.heappush(priority_queue, (priority, neighbor))
                parent[neighbor] = current

    # If end node is not reachable
    return [], 0, num_visited
    # end your code (Part4)


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
