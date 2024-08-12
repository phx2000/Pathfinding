import time

import numpy
import pygame
import sys
import random
import math
from pygame.locals import *
import heapq
from collections import deque
import matplotlib.pyplot as plt

pygame.init()
pygame.display.set_caption('Pathfinding')
screen_size = (1400, 900)
is_fullScreen = False
screen = pygame.display.set_mode(screen_size)

clock = pygame.time.Clock()

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0 ,0)
map_size = 100

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # 距离起点的代价
        self.h = 0  # 距离终点的估算代价
        self.f = 0  # 总代价

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)

def heuristic(current, goal):
    # 使用欧几里得距离作为启发式函数
    return math.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)


def astar(grid, start, end):
    if grid is None:
        raise ValueError("Grid is None. Please provide a valid grid.")

    # 创建起始节点和终点节点
    start_node = Node(start)
    end_node = Node(end)

    open_list = []
    closed_list = set()

    # 将起始节点放入打开列表（优先队列）
    heapq.heappush(open_list, start_node)

    while open_list:
        # 从打开列表中获取代价最小的节点
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node)

        # 找到目标节点，重建路径
        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        # 获取当前节点的邻居
        neighbors = [
            (0, -1), (0, 1), (-1, 0), (1, 0),  # 上下左右四个方向
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 四个对角方向
        ]

        for dx, dy in neighbors:
            neighbor_position = (current_node.position[0] + dx, current_node.position[1] + dy)

            # 检查是否越界或碰到障碍物
            if (0 <= neighbor_position[0] < len(grid) and
                    0 <= neighbor_position[1] < len(grid[0]) and
                    grid[neighbor_position[0]][neighbor_position[1]] == 0):
                neighbor_node = Node(neighbor_position, current_node)

                if neighbor_node in closed_list:
                    continue

                # 计算步长
                if dx == 0 or dy == 0:
                    step_cost = 1
                else:
                    step_cost = math.sqrt(2)

                # 计算G, H, F值
                tentative_g = current_node.g + step_cost
                if neighbor_node not in open_list:
                    neighbor_node.g = tentative_g
                    neighbor_node.h = heuristic(neighbor_node.position, end_node.position)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h
                    heapq.heappush(open_list, neighbor_node)
                elif tentative_g < neighbor_node.g:
                    # 更新现有节点的 g 值
                    neighbor_node.g = tentative_g
                    neighbor_node.f = neighbor_node.g + neighbor_node.h
                    neighbor_node.parent = current_node
                    # 由于 heapq 没有 decrease-key 操作，需要重建堆
                    open_list = [node for node in open_list if node != neighbor_node]
                    heapq.heapify(open_list)
                    heapq.heappush(open_list, neighbor_node)

    # 未找到路径
    return None


def dijkstra(grid, start, end):
    rows, cols = len(grid), len(grid[0])
    distances = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    distances[start] = 0
    priority_queue = [(0, start)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 四个方向加对角线
    came_from = {start: None}

    while priority_queue:
        current_distance, current_position = heapq.heappop(priority_queue)
        if current_position == end:
            break

        for direction in directions:
            neighbor = (current_position[0] + direction[0], current_position[1] + direction[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0]][neighbor[1]] == 0:
                distance = current_distance + (1 if direction[0] == 0 or direction[1] == 0 else 1.414)  # 对角线距离是√2
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    came_from[neighbor] = current_position
                    heapq.heappush(priority_queue, (distance, neighbor))

    if distances[end] == float('inf'):
        return None

    path = []
    current = end
    while current:
        path.append(current)
        current = came_from[current]
    return path[::-1]


def dfs(grid, start, end):
    stack = [(start, [start])]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 四个方向加对角线
    visited = set()

    while stack:
        current_position, path = stack.pop()
        if current_position == end:
            return path

        if current_position in visited:
            continue

        visited.add(current_position)
        for direction in directions:
            neighbor = (current_position[0] + direction[0], current_position[1] + direction[1])
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0:
                stack.append((neighbor, path + [neighbor]))

    return None


def bfs(grid, start, end):
    queue = deque([(start, [start])])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 四个方向加对角线
    visited = set()

    while queue:
        position, path = queue.popleft()
        if position == end:
            return path

        if position in visited:
            continue

        visited.add(position)
        for direction in directions:
            neighbor = (position[0] + direction[0], position[1] + direction[1])
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0:
                queue.append((neighbor, path + [neighbor]))

    return None


def easy_path(grid, starts, end):
    start = [starts[0], starts[1]]
    path = [(start[0], start[1])]
    has_found = False
    while True:
        sign_x = (end[0] > start[0]) - (end[0] < start[0])
        sign_y = (end[1] > start[1]) - (end[1] < start[1])
        # 是否到终点？
        if sign_x == 0 and sign_y == 0:
            has_found = True
            break
        # 斜着走划算的情况
        elif sign_x != 0 and sign_y != 0:
            # 能斜着走就先斜着走
            if grid[start[0]+sign_x][start[1]+sign_y] == 0:
                start[0] += sign_x
                start[1] += sign_y
                path.append((start[0], start[1]))
            # 然后考虑横着走和竖着走
            else:
                # 横着长就先走横着试试
                if abs(end[0] - start[0]) >= abs(end[1] - start[1]):
                    if grid[start[0] + sign_x][start[1]] == 0:
                        start[0] += sign_x
                        path.append((start[0], start[1]))
                    # 走不了就只能竖着走了
                    else:
                        start[1] += sign_y
                        path.append((start[0], start[1]))
                # 否则就先竖着走
                else:
                    if grid[start[0]][start[1]+sign_y] == 0:
                        start[1] += sign_y
                        path.append((start[0], start[1]))
                    # 走不了就只能竖着走了
                    else:
                        start[0] += sign_x
                        path.append((start[0], start[1]))
        else:
            if sign_x == 0:
                if grid[start[0]][start[1] + sign_y] == 0:
                    start[1] += sign_y
                    path.append((start[0], start[1]))
                else:
                    break
            if sign_y == 0:
                if grid[start[0] + sign_x][start[1]] == 0:
                    start[0] += sign_x
                    path.append((start[0], start[1]))
                else:
                    break
    if has_found:
        return path
    else:
        path = path[:-8]
        path2 = bfs(grid, (path[-1][0], path[-1][1]), end)
        if path2:
            for i in path2:
                path.append(i)
            return path
        else:
            return None


def stepwise(grid, starts, end):
    start = [starts[0], starts[1]]
    path = [(start[0], start[1])]
    has_found = False
    while True:
        sign_x = (end[0] > start[0]) - (end[0] < start[0])
        sign_y = (end[1] > start[1]) - (end[1] < start[1])
        # 是否到终点？
        if sign_x == 0 and sign_y == 0:
            has_found = True
            break
        # 斜着走划算的情况
        elif sign_x != 0 and sign_y != 0:
            # 能斜着走就先斜着走
            if grid[start[0]+sign_x][start[1]+sign_y] == 0:
                start[0] += sign_x
                start[1] += sign_y
                path.append((start[0], start[1]))
            # 然后考虑横着走和竖着走
            else:
                # 横着长就先走横着试试
                if abs(end[0] - start[0]) >= abs(end[1] - start[1]):
                    if grid[start[0] + sign_x][start[1]] == 0:
                        start[0] += sign_x
                        path.append((start[0], start[1]))
                    # 走不了就只能竖着走了
                    elif grid[start[0]][start[1]+sign_y] == 0:
                        start[1] += sign_y
                        path.append((start[0], start[1]))
                    # 竖着也走不了
                    elif len(path) >= 2:
                        grid[path[-1][0]][path[-1][1]] = 2
                        path.pop()
                        start = [path[-1][0], path[-1][1]]
                    else:
                        break

                # 否则就先竖着走
                else:
                    if grid[start[0]][start[1]+sign_y] == 0:
                        start[1] += sign_y
                        path.append((start[0], start[1]))
                    # 走不了就只能横着走了
                    elif grid[start[0] + sign_x][start[1]] == 0:
                        start[0] += sign_x
                        path.append((start[0], start[1]))
                    # 横着也走不了
                    elif len(path) >= 2:
                        grid[path[-1][0]][path[-1][1]] = 2
                        path.pop()
                        start = [path[-1][0], path[-1][1]]
                    else:
                        break
        else:
            if sign_x == 0:
                if grid[start[0]][start[1] + sign_y] == 0:
                    start[1] += sign_y
                    path.append((start[0], start[1]))
                elif len(path) >= 2:
                    grid[path[-1][0]][path[-1][1]] = 2
                    path = path[:-1]
                    start = [path[-1][0], path[-1][1]]
                else:
                    break
            if sign_y == 0:
                if grid[start[0] + sign_x][start[1]] == 0:
                    start[0] += sign_x
                    path.append((start[0], start[1]))
                elif len(path) >= 2:
                    grid[path[-1][0]][path[-1][1]] = 2
                    path = path[:-1]
                    start = [path[-1][0], path[-1][1]]
                else:
                    break
    if has_found:
        return path
    else:
        return None


def generate_map1(map_size, start, end):
    meshes_list = []
    for i in range(map_size):
        meshes_list_row = []
        for j in range(map_size):
            temp = random.randint(0, 6)
            if temp <= 2:
                meshes_list_row.append(1)
            else:
                meshes_list_row.append(0)
        meshes_list.append(meshes_list_row)

    rang = 4
    for i in range(start[0]-rang, start[0]+rang):
        for j in range(start[1]-rang, start[1]+rang):
            if 0 <= i < map_size and 0 <= j < map_size:
                meshes_list[i][j] = 0
    for i in range(end[0]-rang, end[0]+rang):
        for j in range(end[1]-rang, end[1]+rang):
            if 0 <= i < map_size and 0 <= j < map_size:
                meshes_list[i][j] = 0
    return  meshes_list


def generate_map2(map_size, start, end):
    meshes_list = []
    for i in range(map_size):
        meshes_list_row = []
        for j in range(map_size):
            temp = random.randint(0, 10)
            if temp <= 0:
                meshes_list_row.append(2)
            else:
                meshes_list_row.append(0)
        meshes_list.append(meshes_list_row)

    for i in range(map_size):
        for j in range(map_size):
            if meshes_list[i][j] == 2:
                meshes_list[i][j] = 1
                all = 0
                # 1
                temp = random.randint(-10, 6)
                if temp > 0 and all < 2:
                    for e in range(temp):
                        meshes_list[max(0, i-e)][j] = 1
                    all += 1
                # 2
                temp = random.randint(-10, 6)
                if temp > 0 and all < 2:
                    for e in range(temp):
                        meshes_list[i][max(0, j-e)] = 1
                    all += 1
                # 3
                temp = random.randint(-10, 6)
                if temp > 0 and all < 2:
                    for e in range(temp):
                        meshes_list[min(map_size-1, i+e)][j] = 1
                    all += 1
                # 4
                temp = random.randint(-10, 6)
                if temp > 0 and all < 2:
                    for e in range(temp):
                        meshes_list[i][min(map_size-1, j+e)] = 1
                    all += 1

    rang = 4
    for i in range(start[0] - rang, start[0] + rang):
        for j in range(start[1] - rang, start[1] + rang):
            if 0 <= i < map_size and 0 <= j < map_size:
                meshes_list[i][j] = 0
    for i in range(end[0] - rang, end[0] + rang):
        for j in range(end[1] - rang, end[1] + rang):
            if 0 <= i < map_size and 0 <= j < map_size:
                meshes_list[i][j] = 0
    return meshes_list


def generate_map3(map_size, start, end):
    meshes_list = []
    map_over = 20
    for i in range(map_size+map_over):
        meshes_list_row = []
        for j in range(map_size+map_over):
            meshes_list_row.append(0)
        meshes_list.append(meshes_list_row)

    rect_size = [random.randint(3, 8), random.randint(3, 8)]
    for i in range(map_size+map_over):
        for j in range(map_size+map_over):
            # in the map, 4 corner == 0
            can_build = True
            a = 1 # width of road
            if a <= i < map_size+map_over - rect_size[0] - a and a <= j < map_size+map_over - rect_size[1] - a:
                for m in range(i-a, i + rect_size[0]+a):
                    for n in range(j-a, j + rect_size[1]+a):
                        if meshes_list[m][n] == 1:
                            can_build = False
                            break
                if can_build and random.randint(0, 10) < 5:
                    for m in range(i, i+rect_size[0]):
                        for n in range(j, j+rect_size[1]):
                            meshes_list[m][n] = 1
                    if random.randint(0, 10) < 5:
                        meshes_list[i][j] = 0
                    if random.randint(0, 10) < 5:
                        meshes_list[i][j+rect_size[1]-1] = 0
                    if random.randint(0, 10) < 5:
                        meshes_list[i+rect_size[0]-1][j] = 0
                    if random.randint(0, 10) < 5:
                        meshes_list[i+rect_size[0]-1][j+rect_size[1]-1] = 0
                    rect_size = [random.randint(3, 8), random.randint(3, 8)]

    meshes_list2 = []
    for i in range(map_size):
        meshes_list_row = []
        for j in range(map_size):
            num = meshes_list[i+10][j+10]
            meshes_list_row.append(num)
        meshes_list2.append(meshes_list_row)

    rang = 4
    for i in range(start[0]-rang, start[0]+rang):
        for j in range(start[1]-rang, start[1]+rang):
            if 0 <= i < map_size and 0 <= j < map_size:
                meshes_list2[i][j] = 0
    for i in range(end[0]-rang, end[0]+rang):
        for j in range(end[1]-rang, end[1]+rang):
            if 0 <= i < map_size and 0 <= j < map_size:
                meshes_list2[i][j] = 0
    return  meshes_list2


def generate_map4(map_size, start, end):
    meshes_list = []
    map_over = 10
    for i in range(map_size+map_over):
        meshes_list_row = []
        for j in range(map_size+map_over):
            meshes_list_row.append(0)
        meshes_list.append(meshes_list_row)

    rect_size = [random.randint(2, 8), random.randint(2, 8)]
    for i in range(map_size+map_over):
        for j in range(map_size+map_over):
            # in the map, 4 corner == 0
            if 0 <= i-2 and i+rect_size[0]+2 < map_size+map_over and 0 <= j-2 and j+rect_size[1]+2 < map_size+map_over and \
                meshes_list[i-2][j-2] == 0 and meshes_list[i-2][j+rect_size[1]+2] == 0 and \
                meshes_list[i+rect_size[0]+2][j-2] == 0 and meshes_list[i+rect_size[0]+2][j+rect_size[1]+2] == 0:
                if random.randint(0, 10) < 7:
                    for m in range(i, i+rect_size[0]):
                        for n in range(j, j+rect_size[1]):
                            meshes_list[m][n] = 1
                    rect_size = [random.randint(2, 8), random.randint(2, 8)]

    meshes_list2 = []
    for i in range(map_size):
        meshes_list_row = []
        for j in range(map_size):
            num = meshes_list[i+2][j+2]
            meshes_list_row.append(num)
        meshes_list2.append(meshes_list_row)

    rang = 10
    for i in range(start[0]-rang, start[0]+rang):
        for j in range(start[1]-rang, start[1]+rang):
            if 0 <= i < map_size and 0 <= j < map_size:
                meshes_list2[i][j] = 0
    for i in range(end[0]-rang, end[0]+rang):
        for j in range(end[1]-rang, end[1]+rang):
            if 0 <= i < map_size and 0 <= j < map_size:
                meshes_list2[i][j] = 0
    return  meshes_list2


def generate_map5(map_size, start, end):
    meshes_list = []
    map_over = 20
    for i in range(map_size+map_over):
        meshes_list_row = []
        for j in range(map_size+map_over):
            meshes_list_row.append(0)
        meshes_list.append(meshes_list_row)

    rect_size = [random.randint(3, 8), random.randint(3, 8)]
    for i in range(map_size+map_over):
        for j in range(map_size+map_over):
            # in the map, 4 corner == 0
            can_build = True
            a = 1 # width of road
            if a <= i < map_size+map_over - rect_size[0] - a and a <= j < map_size+map_over - rect_size[1] - a:
                for m in range(i-a, i + rect_size[0]+a):
                    for n in range(j-a, j + rect_size[1]+a):
                        if meshes_list[m][n] == 1:
                            can_build = False
                            break
                if can_build and random.randint(0, 10) < 5:
                    for m in range(i, i+rect_size[0]):
                        for n in range(j, j+rect_size[1]):
                            meshes_list[m][n] = 1
                    if random.randint(0, 10) < -5:
                        meshes_list[i][j] = 0
                    if random.randint(0, 10) < -5:
                        meshes_list[i][j+rect_size[1]-1] = 0
                    if random.randint(0, 10) < -5:
                        meshes_list[i+rect_size[0]-1][j] = 0
                    if random.randint(0, 10) < -5:
                        meshes_list[i+rect_size[0]-1][j+rect_size[1]-1] = 0
                    rect_size = [random.randint(3, 8), random.randint(3, 8)]

    meshes_list2 = []
    for i in range(map_size):
        meshes_list_row = []
        for j in range(map_size):
            num = meshes_list[i+10][j+10]
            meshes_list_row.append(num)
        meshes_list2.append(meshes_list_row)

    rang = 4
    for i in range(start[0]-rang, start[0]+rang):
        for j in range(start[1]-rang, start[1]+rang):
            if 0 <= i < map_size and 0 <= j < map_size:
                meshes_list2[i][j] = 0
    for i in range(end[0]-rang, end[0]+rang):
        for j in range(end[1]-rang, end[1]+rang):
            if 0 <= i < map_size and 0 <= j < map_size:
                meshes_list2[i][j] = 0
    return  meshes_list2


def test_algorithm(algori):
    x = []
    y = []
    all_breakdown = []
    overtime = False
    mapsizes = []
    for i in range(10, 100, 10):
        mapsizes.append(i)
    for i in range(100, 1601, 100):
        mapsizes.append(i)
    for mapsize in mapsizes:
        time_all = 0
        itera = 100
        breakdown = 0
        for j in range(itera):

            # generate map
            start = (0, 0)
            end = (mapsize - 1, mapsize - 1)
            if not overtime:
                grid = generate_map3(mapsize, start, end)

            # test algorithm
            if overtime:
                elapsed_time = end_time - start_time
            elif algori == "astar":
                start_time = time.perf_counter()
                path = astar(grid, start, end)  # select an algorithm
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
            elif algori == "dijkstra":
                start_time = time.perf_counter()
                path = dijkstra(grid, start, end)  # select an algorithm
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
            elif algori == "dfs":
                start_time = time.perf_counter()
                path = dfs(grid, start, end)  # select an algorithm
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
            elif algori == "bfs":
                start_time = time.perf_counter()
                path = bfs(grid, start, end)  # select an algorithm
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
            elif algori == "stepwise":
                start_time = time.perf_counter()
                path = stepwise(grid, start, end)  # select an algorithm
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

            # calculate cost
            if path and not overtime:
                if elapsed_time > 1:
                    elapsed_time = 1
                    overtime = True
                time_all += elapsed_time
                print("map size: ", mapsize, "    iteration: ", j, "    time: ", elapsed_time)
            elif not overtime:
                breakdown += 1
                print("map size: ", mapsize, "    iteration: ", j, "    not found, time: ", elapsed_time)
            else:
                elapsed_time = 1
                time_all += 1

        if itera != breakdown:
            time_average = time_all / (itera - breakdown)
        else:
            print("not good")
            time_average = time_all
        print("# map size: ", mapsize, "    average time: ", time_average)
        x.append(mapsize)
        y.append(time_average)
        all_breakdown.append(breakdown / itera)
    return x, y, all_breakdown


if True:
    start = (0, 0)
    end = (map_size - 1, map_size - 1)
    grid = generate_map3(map_size, start, end)

    start_time = time.perf_counter()
    path = bfs(grid, start, end) # select an algorithm
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.8f} seconds")
    time_text = f"Time cost: {elapsed_time:.8f} seconds"
    font = pygame.font.SysFont(None, 36)
    if path:
        print("Path found")
    else:
        print("No path found")


if False:
    x, astar_l, astar_breakdown = test_algorithm("astar")
    x, dijkstra_l, dijkstra_breakdown = test_algorithm("dijkstra")
    x, dfs_l, dfs_breakdown = test_algorithm("dfs")
    x, bfs_l, bfs_breakdown = test_algorithm("bfs")
    x, stepwise_l, stepwise_breakdown = test_algorithm("stepwise")
    print(x, astar_l, dijkstra_l, dfs_l, bfs_l, stepwise_l,
          astar_breakdown, dijkstra_breakdown, dfs_breakdown, bfs_breakdown, stepwise_breakdown)

    plt.plot(x, astar_l, label='astar')
    plt.plot(x, dijkstra_l, label='dijkstra')
    plt.plot(x, dfs_l, label='dfs')
    plt.plot(x, bfs_l, label='bfs')
    plt.plot(x, stepwise_l, label='stepwise')

    plt.title('Algorithms Comparison')
    plt.xlabel('map size')
    plt.ylabel('time(seconds)')

    plt.legend()
    plt.show()

    plt.plot(x, astar_breakdown, label='astar')
    plt.plot(x, dijkstra_breakdown, label='dijkstra')
    plt.plot(x, dfs_breakdown, label='dfs')
    plt.plot(x, bfs_breakdown, label='bfs')
    plt.plot(x, stepwise_breakdown, label='stepwise')

    plt.title('Algorithm Comparison')
    plt.xlabel('map size')
    plt.ylabel('path found rate')

    plt.legend()
    plt.show()

while True:
    # get position of mouse
    pos = pygame.mouse.get_pos()

    # event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == KEYDOWN:
            if event.key == K_SPACE:
                is_fullScreen = not is_fullScreen
                if is_fullScreen:
                    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF)
                else:
                    screen = pygame.display.set_mode(screen_size, pygame.RESIZABLE | pygame.DOUBLEBUF)

    screen.fill((150, 150, 150))
    pygame.draw.rect(screen, (20, 20, 40), (300-2, 50-2, 800+2, 800+2), 4)
    for i in range(map_size):
        for j in range(map_size):
            if grid[i][j] == 1:
                pygame.draw.rect(screen, BLACK, (300+int(800*i/map_size+0.5), 50+int(800*j/map_size+0.5),
                                                 1+int(800/map_size-1), 1+int(800/map_size-1)), 0)
            elif grid[i][j] == 2:
                pygame.draw.rect(screen, BLUE, (300+int(800*i/map_size+0.5), 50+int(800*j/map_size+0.5),
                                                 1+int(800/map_size-1), 1+int(800/map_size-1)), 0)
            if path and False:
                if (i, j) in path:
                    pygame.draw.rect(screen, RED, (300 + int(800 * i / map_size + 0.5), 50 + int(800 * j / map_size + 0.5),
                                                 1+int(800 / map_size - 1), 1+int(800 / map_size - 1)), 0)
    starts = (0, 0)
    ends = (0, 0)
    if path and path[-1][0] != -10086:
        for i in range(len(path)-1):
            starts = (300 + int(800 * path[i][0] / map_size + 0.5), 50 + int(800 * path[i][1] / map_size + 0.5))
            ends = (300 + int(800 * path[i+1][0] / map_size + 0.5), 50 + int(800 * path[i+1][1] / map_size + 0.5))
            pygame.draw.line(screen, RED, starts, ends, 1+int(800/map_size-1))
        start = (300 + int(800 * path[0][0] / map_size + 0.5), 50 + int(800 * path[0][1] / map_size + 0.5))
        end = (300 + int(800 * path[-1][0] / map_size + 0.5), 50 + int(800 * path[-1][1] / map_size + 0.5))
        pygame.draw.circle(screen, (0, 255, 0), start, 2+int(800/map_size-1))
        pygame.draw.circle(screen, (0, 255, 0), end, 2+int(800/map_size-1))
    text = font.render(time_text, True, (0, 0, 0))
    text_rect = text.get_rect(center=(700, 870))
    screen.blit(text, text_rect)
    pygame.display.flip()
    clock.tick(2)
