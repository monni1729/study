def solution(arr):
    mx = [[None for i in range(len(arr))] for j in range(len(arr))]
    mi = [[None for i in range(len(arr))] for j in range(len(arr))]

    def mxdp(a, b):
        if a == b:
            mx[a][b] = int(arr[a])
            mi[a][b] = int(arr[a])
        if mx[a][b] != None:
            return mx[a][b]

        tm = []
        for i in range(a + 1, b, 2):
            op = arr[i]
            if op == "+":
                tm.append(mxdp(a, i - 1) + mxdp(i + 1, b))
            elif op == "-":
                tm.append(mxdp(a, i - 1) - midp(i + 1, b))
        mx[a][b] = max(tm)
        return mx[a][b]

    def midp(a, b):
        if a == b:
            mx[a][b] = int(arr[a])
            mi[a][b] = int(arr[a])
        if mi[a][b] != None:
            return mi[a][b]

        tm = []
        for i in range(a + 1, b, 2):
            op = arr[i]
            if op == "+":
                tm.append(midp(a, i - 1) + midp(i + 1, b))
            elif op == "-":
                tm.append(midp(a, i - 1) - mxdp(i + 1, b))
        mi[a][b] = min(tm)
        return mi[a][b]

    return mxdp(0, len(arr) - 1)
