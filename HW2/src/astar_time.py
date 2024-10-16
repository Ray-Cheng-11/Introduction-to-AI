import csv
import heapq
import math

# File paths
edge_file = 'edges.csv'
heuristic_file = 'heuristic.csv'


def load_heuristic():
    """
    Load heuristic values from the heuristic.csv file into a dictionary.
    """
    heuristic = {}
    with open(heuristic_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            node = int(row[0])
            h_value = float(row[1])
            heuristic[node] = h_value
    return heuristic


def astar_time(start, end):
    # begin your code (Part 6)
    """
    I first use csv.reader to read the files, and convert it to a list of rows of data. For each data, I store it into a two-dimensional dictionary, including edges (where I convert the speed from km/h to m/sec) and heuristic values.

    I use a priority_queue to implement the A* algorithm, and store every node's parent into "From". When the A* algorithm detects the end node, it finishes.

    Using the information from the node's parent, I can get the path from start to end, and the distance of the road.

    In the A* algorithm:
        - First, I put the (float(heur[start][end])/(60/3.6), start, -1, 0) into the priority_queue. The structure in the priority_queue is (heuristic value, nodeID, ID of the node's parent, true distance). The heuristic function is the straight distance to the destination divided by the average speed (converted to m/sec) plus the true time from the start point.
        - For every iteration, I get the most prioritized (the least heuristic value) element out and push its neighbors (heuristic value and distance) which haven't been detected into the priority queue until the priority queue is empty or the end node is detected.
        - Additionally, I add "dis" (a dictionary) to record the heuristic value of nodes which are in the priority queue. If the heuristic value of a newly explored node is larger than the previous explore, I do not add it into the priority queue.
        """
    # Load edge data from CSV
    edges = {}
    max_speed_limit = 0  # Initialize maximum speed limit
    with open(edge_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            # Convert node IDs to integers
            start_node, end_node = map(int, row[:2])
            distance = float(row[2])
            speed_limit = float(row[3])
            # Convert speed limit from km/hr to m/s
            speed_limit = speed_limit * 1000 / 3600

            # Update maximum speed limit
            max_speed_limit = max(max_speed_limit, speed_limit)
            # Calculate edge cost based on speed limit
            edge_cost = distance / (speed_limit)
            edges.setdefault(start_node, {})[end_node] = edge_cost

    # Load heuristic values
    heuristic_values = load_heuristic()

    # Update heuristic values based on maximal speed limit
    for node, h_value in heuristic_values.items():
        # Convert speed limit from km/hr to m/s
        heuristic_values[node] = h_value / (max_speed_limit * 1000 / 3600)

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
            # Total time is the cumulative distance to the end node
            time = distances[end]
            return path, time, num_visited

        # Explore neighbors of the current node
        for neighbor, edge_cost in edges.get(current, {}).items():
            cumulative_distance_to_neighbor = distances[current] + edge_cost
            # Update the distance if a shorter path to the neighbor is found
            if neighbor not in distances or cumulative_distance_to_neighbor < distances[neighbor]:
                distances[neighbor] = cumulative_distance_to_neighbor
                priority = cumulative_distance_to_neighbor + \
                    heuristic_values.get(neighbor, 0)
                heapq.heappush(priority_queue, (priority, neighbor))
                parent[neighbor] = current

    # If end node is not reachable
    return [], 0, num_visited
    # end your code (Part 6)


if __name__ == '__main__':
    path, time, num_visited = astar_time(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')
