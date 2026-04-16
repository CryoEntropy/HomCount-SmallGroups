import csv
import numpy as np


def load_hom_data(filename):
    """加载CSV文件中的同态计数数据"""
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = [list(map(int, row)) for row in reader]
    return np.array(data)


def find_equal_hom_pairs(matrix):
    """查找满足条件的群对，排除同一群自身的情况"""
    n = len(matrix)  # 群的数量（51）
    results = []

    # 检查所有不同的群对 (i, j), i ≠ j
    for i in range(n):
        for j in range(i + 1, n):  # 注意：从i+1开始，确保i≠j
            # 获取四个同态计数
            hom_GG = matrix[i][i]  # |Hom(G,G)|
            hom_HG = matrix[j][i]  # |Hom(H,G)|
            hom_GH = matrix[i][j]  # |Hom(G,H)|
            hom_HH = matrix[j][j]  # |Hom(H,H)|

            # 检查是否全部相等
            if hom_GG == hom_HG == hom_GH == hom_HH:
                results.append((i, j, hom_GG))

    return results


def main():
    # 加载数据
    filename = '32小阶群_Hom计数_51x51.csv'
    matrix = load_hom_data(filename)

    print(f"数据矩阵形状: {matrix.shape}")
    print(f"群的数量: {len(matrix)}")

    # 查找满足条件的群对（排除i=j）
    results = find_equal_hom_pairs(matrix)

    print("\n查找结果（排除同一群自身的情况）:")
    if results:
        for i, j, value in results:
            print(f"群 {i} 和群 {j} 满足条件:")
            print(f"  |Hom(G{i},G{i})| = {matrix[i][i]}")
            print(f"  |Hom(G{j},G{i})| = {matrix[j][i]}")
            print(f"  |Hom(G{i},G{j})| = {matrix[i][j]}")
            print(f"  |Hom(G{j},G{j})| = {matrix[j][j]}")
            print(f"  所有值均为: {value}")
            print()
    else:
        print("未找到满足条件的两个不同群")

    # 显示一些统计信息
    print(f"总共检查了 {len(matrix) * (len(matrix) - 1) // 2} 对不同的群")

    # 统计矩阵中的不同值分布
    all_values = matrix.flatten()
    unique_values = np.unique(all_values)
    print(f"\n矩阵中共有 {len(unique_values)} 个不同的同态计数值")
    print(f"最小值: {min(all_values)}, 最大值: {max(all_values)}")


if __name__ == "__main__":
    main()