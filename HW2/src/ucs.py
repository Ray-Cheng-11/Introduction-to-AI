import csv
import heapq
edgeFile = 'edges.csv'


def ucs(start, end):
    # begin your code (Part 3)
    """
    I first use csv.reader to read the file and convert it into a list of rows of data.
    For each row of data, I store it in a two-dimensional dictionary representing the graph,
    where the keys are the starting nodes and the values are dictionaries containing the adjacent nodes and their distances.

    I use a priority_queue (heapq) to implement the Uniform Cost Search (UCS) algorithm,
    where each element in the queue consists of a tuple containing the distance from the start point, node ID, and ID of the node's parent.
    When the UCS algorithm detects the end node, it finishes.

    Using the information from the node's parent, I can reconstruct the path from the start to the end node,
    as well as calculate the total distance of the road.

    * Uniform Cost Search algorithm:
    First, I put the tuple (0, start node) into the priority_queue.
    For every iteration, I extract the most prioritized element (the closest) from the priority_queue,
    and push its neighbors (along with the cumulative distance) that haven't been visited into the priority queue,
    until the priority queue is empty or the end node is detected.
    Additionally, I use a dictionary "distances" to record the distance of nodes that are in the priority queue.
    If the distance of the newly explored node is larger than the previously explored distance,
    I do not add it to the priority queue.
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

    # Priority queue to store nodes based on cumulative distance
    priority_queue = [(0, start)]
    parent = {start: None}
    # Dictionary to store the cumulative distance for each node
    distances = {start: 0}
    path = []
    num_visited = 0

    while priority_queue:
        # Pop node with minimum cumulative distance
        cumulative_distance, current = heapq.heappop(priority_queue)
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
            cumulative_distance_to_neighbor = cumulative_distance + distance
            if neighbor not in distances or cumulative_distance_to_neighbor < distances[neighbor]:
                distances[neighbor] = cumulative_distance_to_neighbor
                heapq.heappush(
                    priority_queue, (cumulative_distance_to_neighbor, neighbor))
                parent[neighbor] = current

    # If end node is not reachable
    return [], 0, num_visited
    # end your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
