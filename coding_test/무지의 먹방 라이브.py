class Food:
    def __init__(self, order, time):
        self.order = order
        self.time = time


def solution(food_times, k):
    ftc_list = []
    for idx, f in enumerate(food_times):
        ftc_list.append(Food(idx + 1, f))

    ftc_list.sort(key=lambda x: x.time, reverse=True)
    prev_line = 0
    curr_line = ftc_list[-1].time
    while True:
        rec = len(ftc_list) * (curr_line - prev_line)
        if k >= rec:
            k -= rec
            while True:
                ftc_list.pop()
                if not ftc_list:
                    return -1
                prev_line = int(curr_line)
                curr_line = ftc_list[-1].time
                if prev_line != curr_line:
                    break
        else:
            r = k % len(ftc_list)
            ftc_list.sort(key=lambda x: x.order)
            return ftc_list[r].order
