class Node:
    def __init__(self, n):
        self.link = []
        self.n = n


def solution(n, computers):
    node_list = [Node(i) for i in range(n)]
    for i in range(n):
        for j in range(n):
            if computers[i][j] and i != j:
                node_list[i].link.append(j)

    node_cnt = 0
    remain = set(range(n))

    while remain:
        node_cnt += 1
        queue = [remain.pop()]
        while queue:
            curr_node = queue.pop(0)
            for tt in node_list[curr_node].link:
                if tt in remain:
                    remain.remove(tt)
                    queue += node_list[tt].link
                    queue.remove(curr_node)

    return node_cnt
