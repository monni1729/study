def solution(land, height):
    r = len(land)
    cost_list = []
    for i in range(r - 1):
        for j in range(r - 1):
            if abs(land[i][j] - land[i][j + 1]) <= height:
                cost_list.append([i * r + j, i * r + j + 1, 0])
            else:
                cost_list.append([i * r + j, i * r + j + 1, abs(land[i][j] - land[i][j + 1])])
            if abs(land[i][j] - land[i + 1][j]) <= height:
                cost_list.append([i * r + j, (i + 1) * r + j, 0])
            else:
                cost_list.append([i * r + j, (i + 1) * r + j, abs(land[i][j] - land[i + 1][j])])
        if abs(land[i][r - 1] - land[i + 1][r - 1]) <= height:
            cost_list.append([i * r + r - 1, (i + 1) * r + r - 1, 0])
        else:
            cost_list.append([i * r + r - 1, (i + 1) * r + r - 1, abs(land[i][r - 1] - land[i + 1][r - 1])])
    for j in range(r - 1):
        if abs(land[r - 1][j] - land[r - 1][j + 1]) <= height:
            cost_list.append([(r - 1) * r + j, (r - 1) * r + j + 1, 0])
        else:
            cost_list.append([(r - 1) * r + j, (r - 1) * r + j + 1, abs(land[r - 1][j] - land[r - 1][j + 1])])

    cost_list.sort(key=lambda x: x[2])

    return solution1(len(cost_list), cost_list)


parent = {}
rank = {}


def make_set(v):
    parent[v] = v
    rank[v] = 0


def find(v):
    if parent[v] != v:
        parent[v] = find(parent[v])

    return parent[v]


def union(v, u):
    root1 = find(v)
    root2 = find(u)

    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2

            if rank[root1] == rank[root2]:
                rank[root2] += 1


def kruskal(graph):
    for v in graph['vertices']:
        make_set(v)

    mst = []

    edges = graph['edges']
    edges.sort()

    for edge in edges:
        weight, v, u = edge

        if find(v) != find(u):
            union(v, u)
            mst.append(edge)

    return mst


def solution1(n, costs):
    answer = 0
    graph = {'vertices': [i for i in range(n)],
             'edges': [(costs[i][2], costs[i][0], costs[i][1]) for i in range(len(costs))]}
    temp = kruskal(graph)
    for i in range(len(temp)):
        answer += temp[i][0]
    return answer
