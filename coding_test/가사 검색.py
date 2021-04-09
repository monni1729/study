def solution(words, queries):
    queries_head = []
    queries_tail = []
    queries_all = []
    for q in queries:
        if q[0] == '?':
            if q[-1] == '?':
                queries_all.append(q)
            else:
                queries_head.append(q[::-1])
        else:
            queries_tail.append(q)

    words_reversed = []
    num_by_len = [0 for _ in range(10001)]
    for w in words:
        words_reversed.append(w[::-1])
        num_by_len[len(w)] += 1

    dict_tail = make_dict(words, queries_tail)
    dict_head = make_dict(words_reversed, queries_head)
    answer = []
    for q in queries:
        if q[0] == '?':
            if q[-1] == '?':
                answer.append(num_by_len[len(q)])
            else:
                qq = remove_qm(q[::-1])
                s = dict_head[len(q)][qq]
                e = dict_head[len(q)][next_chr(qq)]
                answer.append(e - s)
        else:
            qq = remove_qm(q)
            s = dict_tail[len(q)][qq]
            e = dict_tail[len(q)][next_chr(qq)]
            answer.append(e - s)

    return answer


def remove_qm(q):
    for i in range(0, len(q)):
        if q[i] == '?':
            return q[:i]

    return q


def make_dict(words, queries):
    wq_by_len = [[] for _ in range(10001)]
    for w in words:
        wq_by_len[len(w)].append(w)
    for q in queries:
        wq_by_len[len(q)].append(remove_qm(q))
        wq_by_len[len(q)].append(next_chr(remove_qm(q)))

    dict_by_len = [{} for _ in range(10001)]
    for i in range(1, 10001):
        wq_by_len[i].sort()
        cnt = 0
        for wq in wq_by_len[i]:
            wq = remove_qm(wq)
            if len(wq) == i:
                dict_by_len[i][wq] = cnt
                cnt += 1
            else:
                dict_by_len[i][wq] = cnt

    return dict_by_len


def next_chr(word):
    if word == 'z' * len(word):
        return chr(ord('z') + 1)
    else:
        temp = list(word)
        for c_idx in range(len(word) - 1, -1, -1):
            if word[c_idx] != 'z':
                temp[c_idx] = chr(ord(temp[c_idx]) + 1)
                break
            else:
                temp.pop(c_idx)
        rw = ''
        for c in temp:
            rw += c
        return rw