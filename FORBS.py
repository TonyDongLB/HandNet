# import sys
# for line in sys.stdin:
#     a = line.split()
#     print(int(a[0]) + int(a[1]))
#
# import sys
# if __name__ == "__main__":
#     # 读取第一行的n
#     n = int(sys.stdin.readline().strip())
#     ans = 0
#     for i in range(n):
#         # 读取每一行
#         line = sys.stdin.readline().strip()
#         # 把每一行的数字分隔后转化成int列表
#         values = list(map(int, line.split()))
#         for v in values:
#             ans += v
#     print(ans)

# import sys
#
#
# def whether_to_append(matrix, location, all_tuanti):
#     i, j = location
#     m = len(matrix)
#     n = len(matrix[0])
#     has = False
#     for tuanti in all_tuanti:
#         if tuanti is None:
#             break
#         for row in tuanti:
#             if (i, j) in row:
#                 has = True
#                 break
#         if has:
#             break
#     if has:
#         return
#     new_tuanti = []
#     row = [(i, j)]
#     next_j = j + 1
#     while next_j < n:
#         if matrix[i][next_j] == 1:
#             row.append((i, next_j))
#             next_j += 1
#         else:
#             break
#     new_tuanti.append(row)
#     for x in range(1, len(matrix)):
#         row = []
#         last_row = new_tuanti[-1]
#         for (ii, jj) in last_row:
#             ii_b = ii + 1
#             jj_b = jj
#             if ii_b == m:
#                 break
#             # if matrix[ii_b][jj_b] == 0:
#             #     continue
#             while matrix[ii_b][jj_b] == 1:
#                 if (ii_b, jj_b) not in row:
#                     row.append((ii_b, jj_b))
#                     jj_b -= 1
#                     if jj_b < 0:
#                         break
#                 else:
#                     break
#             ii_b = ii + 1
#             jj_b = jj
#             while jj_b < n - 1 and matrix[ii_b][jj_b + 1] == 1:
#                 if (ii_b, jj_b + 1) not in row:
#                     row.append((ii_b, jj_b + 1))
#                     jj_b += 1
#                     if jj_b == n -1:
#                         break
#                 else:
#                     break
#         if len(row) == 0:
#             break
#         new_tuanti.append(row)
#     return new_tuanti
#
#
# if __name__ == "__main__":
#     # 读取第一行的n
#     line = sys.stdin.readline().strip()
#     n = list(map(int, line.split(',')))[0]
#     m = list(map(int, line.split(',')))[1]
#     matrix = []
#     for i in range(n):
#         # 读取每一行
#         line = sys.stdin.readline().strip()
#         # 把每一行的数字分隔后转化成int列表
#         matrix.append(list(map(int, line.split(','))))
#     # print(matrix)
#     all_tuanti = []
#     for i in range(n):
#         for j in range(m):
#             if matrix[i][j] == 1:
#                 result = whether_to_append(matrix, (i, j), all_tuanti)
#                 if result is not None:
#                     all_tuanti.append(result)
#     maxzise = 0
#     tuanti_num = 0
#     for tuanti in all_tuanti:
#         size = 0
#         if tuanti is None:
#             break
#         tuanti_num += 1
#         for row in tuanti:
#             size += len(row)
#         if size > maxzise:
#             maxzise = size
#     print('{},{}'.format(tuanti_num, maxzise))

import sys
if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    total_lines = []
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        lines = list(line.split(';'))
        for dot in lines:
            total_lines.append(dot)
    new_lines = []
    for line in total_lines:
        new_line = []
        new_line.append(int(line.split(',')[0]))
        new_line.append(int(line.split(',')[1]))
        new_lines.append(new_line)
    should_delete = []
    total_lines = new_lines
    new_lines = []
    have = True
    if have:
        have = False
        for i in range(len(total_lines)):
            if i in should_delete:
                continue
            for j in range(i + 1, len(total_lines)):
                if j in should_delete:
                    continue
                a = total_lines[i][0]
                b = total_lines[i][1]
                c = total_lines[j][0]
                d = total_lines[j][1]
                if (b >= c and b <= d) or (a >= c and a <= d) or (a <= c and b >= d):
                    have = True
                    new_a = min(a, c)
                    new_b = max(b, d)
                    total_lines[i][0] = new_a
                    total_lines[i][1] = new_b
                    should_delete.append(j)
                    continue
        for i in range(len(total_lines)):
            if i not in should_delete:
                new_lines.append(total_lines[i])


