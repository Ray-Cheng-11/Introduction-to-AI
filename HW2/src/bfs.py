import csv
from queue import Queue

edgeFile = 'edges.csv'


def bfs(start, end):
    # Begin your code (Part 1)
    """
    I utilize the `csv.reader` to parse the file, converting it into a list of rows of data.
    For each row, I construct a two-dimensional dictionary to store the edges.
    """
    """
    Implement BFS.
    I employ a queue and Initialize an empty dictionary called 'parent', a list with start ID called 'queue', and set to mark the visited node called 'visited', and 'num_visited' to 0
    Upon reaching the end node, the algorithm terminates.
    Leveraging the parent information, I retrieve the path from the start to the end node, along with the distance.

    *BFS Algorithm:
    1. Enqueue the start node into the queue.
    2. While the queue is not empty:
        - Dequeue the front element.
        - Enqueue its unvisited neighbors and mark them with their parent.
        - Continue until the queue is empty or the end node is detected.
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
    q = Queue()
    q.put(start)
    visited.add(start)
    parent = {start: None}
    path = []
    dist = 0
    num_visited = 0

    while not q.empty():
        current = q.get()
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

        for neighbor, distance in edges.get(current, {}).items():
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                q.put(neighbor)

    # If end node is not reachable
    return [], 0, num_visited
    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
