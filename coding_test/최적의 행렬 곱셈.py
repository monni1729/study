def solution(matrix):
    mat = tuple(j for i in matrix for j in i)
    result_dict = {}
    lenmat = len(mat)
    for i in range(0, lenmat, 2):
        result_dict[(i, i + 2)] = 0

    for i in range(0, lenmat - 2, 2):
        result_dict[(i, i + 4)] = mat[i] * mat[i + 1] * mat[i + 3]

    for pcl in range(6, lenmat + 1, 2):
        for start_idx in range(0, lenmat - pcl + 1, 2):
            temp = set()
            for part in range(2, pcl, 2):
                temp.add(
                    result_dict[(start_idx, start_idx + part)] + result_dict[(start_idx + part, start_idx + pcl)] + mat[
                        start_idx] * mat[start_idx + part] * mat[start_idx + pcl - 1])
            result_dict[(start_idx, start_idx + pcl)] = min(temp)

    return result_dict[(0, lenmat)]