def solution(distance, rocks, n):
    rocks.sort()
    rocks = [0] + rocks + [distance]
    min_ans = 1
    max_ans = distance // (len(rocks) - n - 1) + 1

    def check(c):
        local_rocks = rocks.copy()
        cnt = 0

        idx = 0
        while idx < len(local_rocks) - 1:
            if local_rocks[idx + 1] - local_rocks[idx] < c:
                local_rocks.pop(idx + 1)
                cnt += 1
            else:
                idx += 1

            if cnt > n:
                return False

        return True

    while True:
        center = (min_ans + max_ans) // 2
        if check(center):
            min_ans = center
        else:
            max_ans = center

        if max_ans - min_ans == 1:
            return min_ans