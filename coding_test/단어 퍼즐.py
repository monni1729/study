class StrResult:
    def __init__(self, cnt, rem):
        self.cnt = cnt
        self.rem = rem


def solution(strs, target_word):
    str_set = set(strs)
    cursor = len(target_word)
    cls_list = [StrResult(0, target_word), None, None, None, None]

    if cursor > 5:
        while True:
            longest = cls_list.pop(0)
            cls_list.append(None)
            cursor -= 1
            if longest:
                for i in [1, 2, 3, 4, 5]:
                    temp_str = longest.rem[:i]
                    if temp_str in str_set:
                        if cls_list[i - 1]:
                            if cls_list[i - 1].cnt > longest.cnt + 1:
                                cls_list[i - 1].cnt = longest.cnt + 1
                        else:
                            cls_list[i - 1] = StrResult(longest.cnt + 1, longest.rem[i:])

            if cls_list == [None, None, None, None, None]:
                return -1

            if cursor == 5:
                break

    cls_list.append(StrResult(20001, ""))
    for j in [5, 4, 3, 2, 1]:
        longest = cls_list.pop(0)
        if longest:
            for i in range(1, j + 1):
                temp_str = longest.rem[:i]
                if temp_str in str_set:
                    if cls_list[i - 1]:
                        if cls_list[i - 1].cnt > longest.cnt + 1:
                            cls_list[i - 1].cnt = longest.cnt + 1
                    else:
                        cls_list[i - 1] = StrResult(longest.cnt + 1, longest.rem[i:])

    if cls_list[0].cnt > 20000:
        return -1
    else:
        return cls_list[0].cnt
