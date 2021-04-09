from collections import deque


def solution(n, edge):
    graph = [[] for _ in range(n + 1)]
    distances = [0 for _ in range(n)]
    is_visit = [False for _ in range(n)]
    queue = deque()
    queue.appendleft(0)
    is_visit[0] = True

    for (a, b) in edge:
        graph[a - 1].append(b - 1)
        graph[b - 1].append(a - 1)

    while queue:
        i = queue.popleft()

        for j in graph[i]:
            if not is_visit[j]:
                is_visit[j] = True
                queue.append(j)
                distances[j] = distances[i] + 1

    distances.sort(reverse=True)
    answer = distances.count(distances[0])

    return answer
