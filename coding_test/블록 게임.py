class Block:
    def __init__(self):
        self.refp = []
        self.shape = -1
        self.element = []
        self.remnant = []

    def find_shape_remant(self):
        if self.element == [[0, 1], [0, 2], [1, 2]]:
            self.shape = 1
            self.remnant = False
        elif self.element == [[0, 1], [1, 0], [2, 0]]:
            self.shape = 2
            self.remnant = False
        elif self.element == [[1, 0], [1, 1], [1, 2]]:
            self.shape = 3
            self.remnant = [[self.refp[0], self.refp[1] + 1], [self.refp[0], self.refp[1] + 2]]
        elif self.element == [[1, 0], [2, -1], [2, 0]]:
            self.shape = 4
            self.remnant = [[self.refp[0] + 1, self.refp[1] - 1], [self.refp[0] + 1, self.refp[1] - 1]]
        elif self.element == [[0, 1], [0, 2], [1, 0]]:
            self.shape = 5
            self.remnant = False
        elif self.element == [[1, 0], [2, 0], [2, 1]]:
            self.shape = 6
            self.remnant = [[self.refp[0] + 1, self.refp[1] + 1], [self.refp[0] + 1, self.refp[1] + 1]]
        elif self.element == [[1, -2], [1, -1], [1, 0]]:
            self.shape = 7
            self.remnant = [[self.refp[0], self.refp[1] - 2], [self.refp[0], self.refp[1] - 1]]
        elif self.element == [[0, 1], [1, 1], [2, 1]]:
            self.shape = 8
            self.remnant = False
        elif self.element == [[1, -1], [1, 0], [1, 1]]:
            self.shape = 9
            self.remnant = [[self.refp[0], self.refp[1] - 1], [self.refp[0], self.refp[1] + 1]]
        elif self.element == [[1, 0], [1, 1], [2, 0]]:
            self.shape = 10
            self.remnant = False
        elif self.element == [[0, 1], [0, 2], [1, 1]]:
            self.shape = 11
            self.remnant = False
        elif self.element == [[1, -1], [1, 0], [2, 0]]:
            self.shape = 12
            self.remnant = False
        else:
            raise KeyError


def solution(board):
    r = len(board)
    blocks_dict_list = []
    for i in range(r):
        for j in range(r):
            if board[i][j] > 0:
                if board[i][j] not in blocks_dict_list:
                    blocks_dict_list.append(board[i][j])

    blocks_dict_list.sort()
    num_blocks = len(blocks_dict_list)
    blocks_dict = {}
    for i in range(num_blocks):
        blocks_dict[blocks_dict_list[i]] = i + 1

    for i in range(r):
        for j in range(r):
            if board[i][j] > 0:
                board[i][j] = blocks_dict[board[i][j]]

    blocks_list = [Block() for _ in range(num_blocks)]

    for i in range(r):
        for j in range(r):
            if board[i][j] > 0:
                if blocks_list[board[i][j] - 1].refp:
                    blocks_list[board[i][j] - 1].element.append(
                        [i - blocks_list[board[i][j] - 1].refp[0], j - blocks_list[board[i][j] - 1].refp[1]])
                else:
                    blocks_list[board[i][j] - 1].refp = [i, j]

    for b in blocks_list:
        b.find_shape_remant()
    idx = 0
    answer = 0
    while idx < len(blocks_list):
        b = blocks_list[idx]

        disable = False
        if b.remnant:
            for i in range(b.remnant[0][0] + 1):
                if board[i][b.remnant[0][1]] is not 0:
                    disable = True
                    break
            for i in range(b.remnant[1][0] + 1):
                if board[i][b.remnant[1][1]] is not 0:
                    disable = True
                    break
            if disable:
                idx += 1
            else:
                answer += 1
                board[blocks_list[idx].refp[0]][blocks_list[idx].refp[1]] = 0
                board[blocks_list[idx].refp[0] + blocks_list[idx].element[0][0]][
                    blocks_list[idx].refp[1] + +blocks_list[idx].element[0][1]] = 0
                board[blocks_list[idx].refp[0] + blocks_list[idx].element[1][0]][
                    blocks_list[idx].refp[1] + +blocks_list[idx].element[1][1]] = 0
                board[blocks_list[idx].refp[0] + blocks_list[idx].element[2][0]][
                    blocks_list[idx].refp[1] + +blocks_list[idx].element[2][1]] = 0
                blocks_list.pop(idx)
                idx = 0
        else:
            idx += 1

    return answer
