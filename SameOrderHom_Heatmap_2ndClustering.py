import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import matplotlib.patches as patches

# ============================== 配置选项 ==============================
# 第一次聚类依据（主聚类）：AbelRank
PRIMARY_CLUSTER = 'AbelRank'  # 注意：原数据中没有AbelRank列，需要计算

# 第二次聚类依据（次聚类）：AbelianizationOrder
SECONDARY_CLUSTER = 'AbelianizationOrder'

# ============================== 主程序 ==============================
# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("正在读取数据...")

# 读取同态计数表（带标签）
hom_df = pd.read_csv('Hom_counts_of_cluster_32_to_32.csv', index_col=0)

# 将列名也转换为整数类型，以保持一致性
hom_df.columns = hom_df.columns.astype(int)

# 读取群信息表
summary_df = pd.read_csv('groups_order_of_cluster_32_summary.csv')

print(f"同态计数表形状: {hom_df.shape}")
print(f"群信息表形状: {summary_df.shape}")

# 确保Id列是整数类型
summary_df['Id'] = summary_df['Id'].astype(int)

# ============================== 计算AbelRank ==============================
print("\n正在计算AbelRank...")


# AbelRank定义为阿贝尔不变因子的个数
def calculate_abel_rank(abelian_invariants_str):
    """
    从阿贝尔不变因子字符串计算AbelRank（秩）
    例如：
    "[32]" -> 1
    "[4, 8]" -> 2
    "[2, 2, 4]" -> 3
    "[2, 2, 2, 2]" -> 4
    """
    if pd.isna(abelian_invariants_str):
        return 0

    # 去除方括号，按逗号分割
    try:
        # 处理字符串，例如"[2, 2, 4]"
        cleaned = abelian_invariants_str.strip('[]')
        if cleaned == '':
            return 0
        # 分割并计算元素个数
        factors = cleaned.split(',')
        return len(factors)
    except:
        # 如果解析失败，尝试其他方法
        return 0


# 计算AbelRank
summary_df['AbelRank'] = summary_df['AbelianInvariants'].apply(calculate_abel_rank)

print(f"AbelRank分布:")
abel_rank_counts = summary_df['AbelRank'].value_counts().sort_index()
for rank, count in abel_rank_counts.items():
    print(f"  AbelRank={rank}: {count}个群")

print(f"\n聚类依据:")
print(f"  主聚类: {PRIMARY_CLUSTER}")
print(f"  次聚类: {SECONDARY_CLUSTER}")


# ============================== 实现二级聚类 ==============================
def get_hierarchical_cluster_info(summary_df, primary_column, secondary_column):
    """实现两次聚类：先按primary_column聚类，再按secondary_column聚类
       对于某些特定列（CenterOrder, AbelianizationOrder），次聚类按从大到小排序"""

    # 复制数据，避免修改原始数据
    df = summary_df.copy()

    # 检查次聚类依据是否为需要降序排列的列
    # 对于AbelianizationOrder，值越大表示阿贝尔化后群越大，所以从大到小排列
    #descending_secondary_columns = ['CenterOrder', 'AbelianizationOrder']
    descending_secondary_columns = ['CenterOrder']

    if secondary_column in descending_secondary_columns:
        # 对于需要降序的列，我们创建一个临时列用于排序
        # 先按主聚类升序，然后按次聚类降序，最后按Id升序
        df_sorted = df.sort_values([primary_column, secondary_column, 'Id'],
                                   ascending=[True, False, True])
    else:
        # 对于其他列，按常规方式排序：主聚类升序，次聚类升序，Id升序
        df_sorted = df.sort_values([primary_column, secondary_column, 'Id'])

    # 获取排序后的Id列表
    sorted_ids = df_sorted['Id'].tolist()

    # 记录聚类信息
    primary_boundaries = []
    secondary_boundaries = []

    # 记录每个聚类的详细信息
    cluster_info = []

    # 获取所有主聚类值
    primary_values = df_sorted[primary_column].unique()

    primary_start = 0
    for i, primary_val in enumerate(primary_values):
        # 获取主聚类对应的数据
        primary_mask = df_sorted[primary_column] == primary_val
        primary_subset = df_sorted[primary_mask]

        # 记录主聚类边界
        primary_boundaries.append(primary_start)

        # 获取次聚类值（已经按正确的顺序排序）
        secondary_values = primary_subset[secondary_column].unique()

        secondary_start = 0
        for j, secondary_val in enumerate(secondary_values):
            # 获取次聚类对应的数据
            secondary_mask = primary_subset[secondary_column] == secondary_val
            secondary_subset = primary_subset[secondary_mask]

            # 记录次聚类边界
            secondary_boundaries.append(primary_start + secondary_start)

            # 保存聚类信息
            cluster_ids = secondary_subset['Id'].tolist()
            cluster_size = len(cluster_ids)

            cluster_info.append({
                'primary_value': primary_val,
                'secondary_value': secondary_val,
                'primary_index': i,
                'secondary_index': j,
                'start': primary_start + secondary_start,
                'end': primary_start + secondary_start + cluster_size - 1,
                'size': cluster_size,
                'ids': cluster_ids
            })

            secondary_start += cluster_size

        primary_start += len(primary_subset)

    # 添加最终边界
    primary_boundaries.append(len(df_sorted))
    secondary_boundaries.append(len(df_sorted))

    return df_sorted, sorted_ids, cluster_info, primary_boundaries, secondary_boundaries


# 获取二级聚类信息
print(f"\n正在获取二级聚类信息（主聚类: {PRIMARY_CLUSTER}, 次聚类: {SECONDARY_CLUSTER}）...")
summary_df_sorted, sorted_ids, cluster_info, primary_boundaries, secondary_boundaries = get_hierarchical_cluster_info(
    summary_df, PRIMARY_CLUSTER, SECONDARY_CLUSTER
)

# 打印聚类信息
print(f"\n=== 按{PRIMARY_CLUSTER}→{SECONDARY_CLUSTER}二级聚类的统计信息 ===")

# 按主聚类分组
primary_groups = {}
for info in cluster_info:
    primary_val = info['primary_value']
    if primary_val not in primary_groups:
        primary_groups[primary_val] = []
    primary_groups[primary_val].append(info)

for i, (primary_val, secondary_infos) in enumerate(primary_groups.items()):
    print(f"\n主聚类 {i + 1}: {PRIMARY_CLUSTER} = {primary_val}")
    total_groups = sum([info['size'] for info in secondary_infos])
    print(f"  总群数量: {total_groups}")

    for j, info in enumerate(secondary_infos):
        print(f"  次聚类 {j + 1}: {SECONDARY_CLUSTER} = {info['secondary_value']}")
        print(f"    群数量: {info['size']}")
        print(f"    群ID: {info['ids']}")

# 显示排序方式
print(f"\n排序方式说明:")
print(f"  主聚类({PRIMARY_CLUSTER}): 从小到大排序")
print(f"  次聚类({SECONDARY_CLUSTER}): 从大到小排序（值越大表示阿贝尔化后群越大）")

# 重新排列同态计数表的行和列，按照聚类排序
hom_sorted = hom_df.loc[sorted_ids, sorted_ids]
print(f"\n重新排序后的同态计数表形状: {hom_sorted.shape}")

# 由于同态计数数值范围很大，使用对数刻度
hom_log = np.log10(hom_sorted.values.astype(float) + 1)  # +1避免log(0)

# 创建图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 25),
                               gridspec_kw={'height_ratios': [1, 0.05]})

# 绘制热力图
cax = ax1.imshow(hom_log, cmap='viridis', aspect='auto')

# 添加主聚类边界线（粗线）
for boundary in primary_boundaries[:-1]:  # 最后一个边界不需要画线
    ax1.axhline(y=boundary - 0.5, color='red', linewidth=3, linestyle='-', alpha=0.8)
    ax1.axvline(x=boundary - 0.5, color='red', linewidth=3, linestyle='-', alpha=0.8)

# 添加次聚类边界线（细线）
for boundary in secondary_boundaries[:-1]:
    if boundary not in primary_boundaries:  # 避免重复画线
        ax1.axhline(y=boundary - 0.5, color='white', linewidth=1, linestyle='--', alpha=0.6)
        ax1.axvline(x=boundary - 0.5, color='white', linewidth=1, linestyle='--', alpha=0.6)

# 设置坐标轴标签
sorted_id_str = [str(id) for id in sorted_ids]
ax1.set_xticks(np.arange(len(sorted_ids)))
ax1.set_yticks(np.arange(len(sorted_ids)))
ax1.set_xticklabels(sorted_id_str, fontsize=8, rotation=90)
ax1.set_yticklabels(sorted_id_str, fontsize=8)

# 设置标题和轴标签
ax1.set_title(
    f'32阶群同态计数热力图（按{PRIMARY_CLUSTER}→{SECONDARY_CLUSTER}二级聚类，对数刻度）\n红色粗线:主聚类边界, 白色虚线:次聚类边界',
    fontsize=16, pad=20)
ax1.set_xlabel('目标群 ID', fontsize=12)
ax1.set_ylabel('源群 ID', fontsize=12)

# 添加网格
ax1.grid(False)

# 创建颜色条
cbar = plt.colorbar(cax, cax=ax2, orientation='horizontal')
cbar.set_label('同态数量（对数刻度: log10(N+1))', fontsize=12)

# 添加主聚类信息文本（左侧和顶部）
for info in cluster_info:
    if info['secondary_index'] == 0:  # 只显示每个主聚类的第一个次聚类
        # 左侧标签
        cluster_center = (info['start'] + info['end']) / 2
        label = f"{PRIMARY_CLUSTER}={info['primary_value']}"
        ax1.text(-5, cluster_center, label,
                 ha='right', va='center', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # 顶部标签
        ax1.text(cluster_center, -5, label,
                 ha='center', va='bottom', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                 rotation=90)

# 调整布局
plt.tight_layout()

# 保存图像
output_filename = f'32阶群同态计数_{PRIMARY_CLUSTER}→{SECONDARY_CLUSTER}_二级聚类热力图.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n热力图已保存为 '{output_filename}'")

# 显示图形
plt.show()

# ============================== 创建聚类块间平均同态数量热力图（二级聚类） ==============================
print("\n正在创建聚类块间平均同态数量热力图（二级聚类）...")

# 计算聚类块间的平均同态数量
cluster_avg_matrix = np.zeros((len(cluster_info), len(cluster_info)))

for i, info_i in enumerate(cluster_info):
    ids_i = info_i['ids']
    for j, info_j in enumerate(cluster_info):
        ids_j = info_j['ids']
        # 计算聚类块i到聚类块j的平均同态数量
        sub_matrix = hom_sorted.loc[ids_i, ids_j]
        cluster_avg_matrix[i, j] = np.mean(sub_matrix.values)

# 创建聚类块间热力图
fig2, ax3 = plt.subplots(figsize=(14, 12))

# 使用对数刻度
matrix_log = np.log10(cluster_avg_matrix + 1)

# 绘制热力图
cax2 = ax3.imshow(matrix_log, cmap='plasma', aspect='auto')

# 为每个聚类块创建标签
cluster_labels = []
for info in cluster_info:
    label = f"{PRIMARY_CLUSTER}={info['primary_value']}\n{SECONDARY_CLUSTER}={info['secondary_value']}"
    cluster_labels.append(label)

# 简化标签显示
short_cluster_labels = []
for label in cluster_labels:
    short_label = label[:15] + '...' if len(label) > 15 else label
    short_cluster_labels.append(short_label.replace('\n', '/'))

# 添加聚类块边界线
# 主聚类边界（水平线和垂直线 - 粗线）
primary_cluster_boundaries = []
current_primary = None
for i, info in enumerate(cluster_info):
    if current_primary is None:
        current_primary = info['primary_value']
    elif current_primary != info['primary_value']:
        ax3.axhline(y=i - 0.5, color='white', linewidth=3, linestyle='-')
        ax3.axvline(x=i - 0.5, color='white', linewidth=3, linestyle='-')
        primary_cluster_boundaries.append(i)
        current_primary = info['primary_value']

# 设置坐标轴
ax3.set_xticks(np.arange(len(cluster_info)))
ax3.set_yticks(np.arange(len(cluster_info)))
ax3.set_xticklabels(short_cluster_labels, fontsize=8, rotation=90, ha='center')
ax3.set_yticklabels(short_cluster_labels, fontsize=8)

# 添加数值标签
for i in range(len(cluster_info)):
    for j in range(len(cluster_info)):
        text = ax3.text(j, i, f'{cluster_avg_matrix[i, j]:.0f}',
                        ha="center", va="center", color="white", fontsize=7)

ax3.set_title(f'按{PRIMARY_CLUSTER}→{SECONDARY_CLUSTER}聚类的聚类块间平均同态数量（对数刻度）\n白色粗线:主聚类边界',
              fontsize=14, pad=20)
ax3.set_xlabel('目标聚类块', fontsize=12)
ax3.set_ylabel('源聚类块', fontsize=12)

plt.colorbar(cax2, ax=ax3, label='平均同态数量（对数刻度: log10(avg+1))')
plt.tight_layout()

cluster_matrix_filename = f'按{PRIMARY_CLUSTER}→{SECONDARY_CLUSTER}聚类的聚类块间平均同态数量热力图.png'
plt.savefig(cluster_matrix_filename, dpi=300, bbox_inches='tight')
print(f"聚类块间热力图已保存为 '{cluster_matrix_filename}'")
plt.show()

# ============================== 保存详细信息到CSV文件 ==============================
print(f"\n正在保存按{PRIMARY_CLUSTER}→{SECONDARY_CLUSTER}聚类的详细信息到CSV文件...")
cluster_details = []
for i, info in enumerate(cluster_info):
    primary_val = info['primary_value']
    secondary_val = info['secondary_value']
    for group_id in info['ids']:
        group_info = summary_df[summary_df['Id'] == group_id].iloc[0]
        cluster_details.append({
            '主聚类值': primary_val,
            '次聚类值': secondary_val,
            '聚类块编号': f"{info['primary_index'] + 1}.{info['secondary_index'] + 1}",
            '群ID': group_id,
            '群名称': group_info['Name'],
            '指数': group_info['Exponent'],
            '中心阶': group_info['CenterOrder'],
            '导出子群阶': group_info['DerivedOrder'],
            '阿贝尔化阶': group_info['AbelianizationOrder'],
            '幂零类': group_info['NilpotencyClass'],
            'AbelRank': group_info['AbelRank'],
            '阿贝尔不变因子': group_info['AbelianInvariants']
        })

cluster_details_df = pd.DataFrame(cluster_details)
output_csv = f'32阶群_{PRIMARY_CLUSTER}→{SECONDARY_CLUSTER}_二级聚类详细信息.csv'
cluster_details_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"聚类详细信息已保存为 '{output_csv}'")

# 保存聚类块间平均同态数量
cluster_avg_df = pd.DataFrame(
    cluster_avg_matrix,
    index=[f"{PRIMARY_CLUSTER}={info['primary_value']},{SECONDARY_CLUSTER}={info['secondary_value']}"
           for info in cluster_info],
    columns=[f"{PRIMARY_CLUSTER}={info['primary_value']},{SECONDARY_CLUSTER}={info['secondary_value']}"
             for info in cluster_info]
)
cluster_avg_df.to_csv(f'按{PRIMARY_CLUSTER}→{SECONDARY_CLUSTER}聚类的聚类块间平均同态数量.csv', encoding='utf-8-sig')
print(f"聚类块间平均同态数量已保存为 '按{PRIMARY_CLUSTER}→{SECONDARY_CLUSTER}聚类的聚类块间平均同态数量.csv'")

# ============================== 显示聚类详情 ==============================
print(f"\n=== 按{PRIMARY_CLUSTER}→{SECONDARY_CLUSTER}聚类的群详情 ===")

# 按主聚类分组显示
for i, (primary_val, secondary_infos) in enumerate(primary_groups.items()):
    print(f"\n主聚类 {i + 1}: {PRIMARY_CLUSTER} = {primary_val}")
    total_groups = sum([info['size'] for info in secondary_infos])
    print(f"  总群数量: {total_groups}")

    for j, info in enumerate(secondary_infos):
        print(f"\n  次聚类 {j + 1}: {SECONDARY_CLUSTER} = {info['secondary_value']} (共{info['size']}个群)")
        for group_id in info['ids']:
            group_info = summary_df[summary_df['Id'] == group_id].iloc[0]
            print(f"    [32,{group_id}]: {group_info['Name']} ")

# ============================== 显示汇总信息 ==============================
print("\n" + "=" * 60)
print("汇总信息")
print("=" * 60)

print(f"\n聚类方式:")
print(f"  主聚类: {PRIMARY_CLUSTER} (按AbelianInvariants计算的秩)")
print(f"  次聚类: {SECONDARY_CLUSTER} (阿贝尔化阶)")

print(f"\n聚类统计:")
print(f"  主聚类数量: {len(primary_groups)}")
print(f"  聚类块总数: {len(cluster_info)}")

print(f"\nAbelRank分布:")
for rank, count in abel_rank_counts.items():
    print(f"  AbelRank={rank}: {count}个群")

print(f"\n排序规则:")
print(f"  主聚类({PRIMARY_CLUSTER}): 从小到大排序")
print(f"  次聚类({SECONDARY_CLUSTER}): 从大到小排序（值越大表示阿贝尔化后群越大）")

print("\n" + "=" * 60)
print("所有聚类依据可选值:")
print("  主聚类: 'Exponent', 'CenterOrder', 'DerivedOrder',")
print("          'AbelianizationOrder', 'NilpotencyClass', 'AbelRank'")
print("  次聚类: 'Exponent', 'CenterOrder', 'DerivedOrder',")
print("          'AbelianizationOrder', 'NilpotencyClass', 'AbelRank'")
print("注意：'AbelRank'是通过AbelianInvariants计算得到的秩")
print("=" * 60)