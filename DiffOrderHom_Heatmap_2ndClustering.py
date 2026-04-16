import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import matplotlib.patches as patches

# ============================== 配置选项 ==============================
# 第一次聚类依据（主聚类）：通常使用 Exponent
PRIMARY_CLUSTER_32 = 'AbelRank'  # 32阶群的第一次聚类依据
PRIMARY_CLUSTER_16 = 'AbelRank'  # 16阶群的第一次聚类依据

# 第二次聚类依据（次聚类）：在主聚类的基础上进一步细分
# 可选值: 'CenterOrder', 'DerivedOrder', 'AbelianizationOrder', 'NilpotencyClass'
# 对于32阶群还可以使用: 'AssignedClass'
SECONDARY_CLUSTER_32 = 'AbelianizationOrder'  # 32阶群的第二次聚类依据
SECONDARY_CLUSTER_16 = 'AbelianizationOrder'  # 16阶群的第二次聚类依据

# ============================== 主程序 ==============================
# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("正在读取数据...")

# 读取16阶群信息表
summary_16_df = pd.read_csv('hom16_group_summary.csv')
summary_16_df['Id'] = summary_16_df['Id'].astype(int)
print(f"16阶群信息表形状: {summary_16_df.shape}")

# 读取32阶群信息表
summary_32_df = pd.read_csv('hom32_group_summary.csv')
summary_32_df['Id'] = summary_32_df['Id'].astype(int)
print(f"32阶群信息表形状: {summary_32_df.shape}")

# 读取16阶到32阶的同态计数表
hom_16_to_32_df = pd.read_csv('hom_16_to_32.csv')
hom_16_to_32_df.rename(columns={'G16': 'Id'}, inplace=True)
hom_16_to_32_df['Id'] = hom_16_to_32_df['Id'].astype(int)
hom_16_to_32_df.columns = [int(col) if col != 'Id' else col for col in hom_16_to_32_df.columns]
print(f"16阶到32阶同态计数表形状: {hom_16_to_32_df.shape}")

# 读取32阶到16阶的同态计数表
hom_32_to_16_df = pd.read_csv('hom_32_to_16.csv')
hom_32_to_16_df.rename(columns={'G32': 'Id'}, inplace=True)
hom_32_to_16_df['Id'] = hom_32_to_16_df['Id'].astype(int)
hom_32_to_16_df.columns = [int(col) if col != 'Id' else col for col in hom_32_to_16_df.columns]
print(f"32阶到16阶同态计数表形状: {hom_32_to_16_df.shape}")


# ============================== 函数定义 ==============================
def get_hierarchical_cluster_info(summary_df, primary_column, secondary_column):
    """实现两次聚类：先按primary_column聚类，再按secondary_column聚类
       对于某些特定列（CenterOrder, AbelianizationOrder），次聚类按从大到小排序"""

    # 复制数据，避免修改原始数据
    df = summary_df.copy()

    # 检查次聚类依据是否为需要降序排列的列
    # 对于CenterOrder和AbelianizationOrder，值越大通常表示群越接近阿贝尔群，所以从大到小排列更合理
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


def create_cross_group_heatmap_with_hierarchical_clustering(
        hom_df, source_summary, target_summary,
        source_cluster_info, target_cluster_info,
        source_sorted_ids, target_sorted_ids,
        source_primary, source_secondary,
        target_primary, target_secondary,
        direction):
    """创建跨群同态计数热力图（支持两级聚类）"""

    # 重新排列同态计数表的行和列，按照聚类排序
    hom_sorted = hom_df.loc[source_sorted_ids, target_sorted_ids]

    # 由于同态计数数值范围可能很大，使用对数刻度
    hom_log = np.log10(hom_sorted.values.astype(float) + 1)  # +1避免log(0)

    # 获取源和目标聚类的边界信息
    # 我们只需要提取cluster_info中的边界信息
    source_primary_boundaries = sorted(
        list(set([info['start'] for info in source_cluster_info] + [len(source_sorted_ids)])))
    source_secondary_boundaries = sorted(
        list(set([info['start'] for info in source_cluster_info] + [len(source_sorted_ids)])))

    target_primary_boundaries = sorted(
        list(set([info['start'] for info in target_cluster_info] + [len(target_sorted_ids)])))
    target_secondary_boundaries = sorted(
        list(set([info['start'] for info in target_cluster_info] + [len(target_sorted_ids)])))

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 22),
                                   gridspec_kw={'height_ratios': [1, 0.05]})

    # 绘制热力图
    cax = ax1.imshow(hom_log, cmap='viridis', aspect='auto')

    # 添加目标群主聚类边界线（垂直线 - 粗线）
    for boundary in target_primary_boundaries[:-1]:
        ax1.axvline(x=boundary - 0.5, color='red', linewidth=3, linestyle='-', alpha=0.8)

    # 添加目标群次聚类边界线（垂直线 - 细线）
    for boundary in target_secondary_boundaries[:-1]:
        if boundary not in target_primary_boundaries:  # 避免重复画线
            ax1.axvline(x=boundary - 0.5, color='white', linewidth=1, linestyle='--', alpha=0.6)

    # 添加源群主聚类边界线（水平线 - 粗线）
    for boundary in source_primary_boundaries[:-1]:
        ax1.axhline(y=boundary - 0.5, color='red', linewidth=3, linestyle='-', alpha=0.8)

    # 添加源群次聚类边界线（水平线 - 细线）
    for boundary in source_secondary_boundaries[:-1]:
        if boundary not in source_primary_boundaries:  # 避免重复画线
            ax1.axhline(y=boundary - 0.5, color='white', linewidth=1, linestyle='--', alpha=0.6)

    # 设置坐标轴标签
    source_id_str = [str(id) for id in source_sorted_ids]
    target_id_str = [str(id) for id in target_sorted_ids]

    ax1.set_xticks(np.arange(len(target_sorted_ids)))
    ax1.set_yticks(np.arange(len(source_sorted_ids)))
    ax1.set_xticklabels(target_id_str, fontsize=8, rotation=90)
    ax1.set_yticklabels(source_id_str, fontsize=8)

    # 设置标题和轴标签
    if direction == '16_to_32':
        # 添加次聚类排序方式说明
        sort_note = ""
        if source_secondary in ['CenterOrder', 'AbelianizationOrder']:
            sort_note += f" (源次聚类降序)"
        if target_secondary in ['CenterOrder', 'AbelianizationOrder']:
            if sort_note:
                sort_note += ", "
            sort_note += f"目标次聚类降序"

        ax1.set_title(
            f'16阶群到32阶群同态计数热力图\n'
            f'(源:按{source_primary}→{source_secondary}聚类, '
            f'目标:按{target_primary}→{target_secondary}聚类, 对数刻度)\n'
            f'红色粗线:主聚类边界, 白色虚线:次聚类边界{sort_note}',
            fontsize=16, pad=25)
        ax1.set_xlabel('32阶群 ID', fontsize=12)
        ax1.set_ylabel('16阶群 ID', fontsize=12)
    else:
        # 添加次聚类排序方式说明
        sort_note = ""
        if source_secondary in ['CenterOrder', 'AbelianizationOrder']:
            sort_note += f" (源次聚类降序)"
        if target_secondary in ['CenterOrder', 'AbelianizationOrder']:
            if sort_note:
                sort_note += ", "
            sort_note += f"目标次聚类降序"

        ax1.set_title(
            f'32阶群到16阶群同态计数热力图\n'
            f'(源:按{source_primary}→{source_secondary}聚类, '
            f'目标:按{target_primary}→{target_secondary}聚类, 对数刻度)\n'
            f'红色粗线:主聚类边界, 白色虚线:次聚类边界{sort_note}',
            fontsize=16, pad=25)
        ax1.set_xlabel('16阶群 ID', fontsize=12)
        ax1.set_ylabel('32阶群 ID', fontsize=12)

    # 添加网格
    ax1.grid(False)

    # 创建颜色条
    cbar = plt.colorbar(cax, cax=ax2, orientation='horizontal')
    cbar.set_label('同态数量（对数刻度: log10(N+1))', fontsize=12)

    # 添加源群聚类信息文本（左侧）
    for info in source_cluster_info:
        if info['secondary_index'] == 0:  # 只显示每个主聚类的第一个次聚类
            cluster_center = (info['start'] + info['end']) / 2
            label = f"{source_primary}={info['primary_value']}"
            ax1.text(-1.5, cluster_center, label,
                     ha='right', va='center', fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

    # 添加目标群聚类信息文本（顶部）
    for info in target_cluster_info:
        if info['secondary_index'] == 0:  # 只显示每个主聚类的第一个次聚类
            cluster_center = (info['start'] + info['end']) / 2
            label = f"{target_primary}={info['primary_value']}"
            ax1.text(cluster_center, -1.5, label,
                     ha='center', va='bottom', fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='orange', alpha=0.7),
                     rotation=90)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    output_filename = f'{direction}_同态计数_{source_primary}→{source_secondary}_to_{target_primary}→{target_secondary}_二级聚类热力图.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"热力图已保存为 '{output_filename}'")

    plt.show()

    return hom_sorted


def print_hierarchical_cluster_statistics(cluster_info, primary_column, secondary_column, group_type):
    """打印两级聚类统计信息"""
    print(f"\n=== {group_type}阶群按{primary_column}→{secondary_column}二级聚类的统计信息 ===")

    # 按主聚类分组
    primary_groups = {}
    for info in cluster_info:
        primary_val = info['primary_value']
        if primary_val not in primary_groups:
            primary_groups[primary_val] = []
        primary_groups[primary_val].append(info)

    for i, (primary_val, secondary_infos) in enumerate(primary_groups.items()):
        print(f"\n主聚类 {i + 1}: {primary_column} = {primary_val}")
        total_groups = sum([info['size'] for info in secondary_infos])
        print(f"  总群数量: {total_groups}")

        for j, info in enumerate(secondary_infos):
            print(f"  次聚类 {j + 1}: {secondary_column} = {info['secondary_value']}")
            print(f"    群数量: {info['size']}")
            print(f"    群ID: {info['ids']}")


def create_cluster_blocks_heatmap(cluster_avg_matrix, source_cluster_info, target_cluster_info,
                                  source_primary, source_secondary, target_primary, target_secondary,
                                  direction):
    """创建聚类块间平均同态数量热力图"""

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 12))

    # 使用对数刻度
    matrix_log = np.log10(cluster_avg_matrix + 1)

    # 绘制热力图
    cax = ax.imshow(matrix_log, cmap='plasma', aspect='auto')

    # 设置坐标轴标签
    # 为每个聚类块创建标签
    source_labels = []
    for info in source_cluster_info:
        label = f"{source_primary}={info['primary_value']}\n{source_secondary}={info['secondary_value']}"
        source_labels.append(label)

    target_labels = []
    for info in target_cluster_info:
        label = f"{target_primary}={info['primary_value']}\n{target_secondary}={info['secondary_value']}"
        target_labels.append(label)

    # 简化标签显示
    short_source_labels = []
    for label in source_labels:
        short_label = label[:15] + '...' if len(label) > 15 else label
        short_source_labels.append(short_label.replace('\n', '/'))

    short_target_labels = []
    for label in target_labels:
        short_label = label[:15] + '...' if len(label) > 15 else label
        short_target_labels.append(short_label.replace('\n', '/'))

    # 添加聚类块边界线
    # 源聚类边界（水平线）
    source_primary_boundaries = []
    current_primary = None
    for i, info in enumerate(source_cluster_info):
        if current_primary is None:
            current_primary = info['primary_value']
        elif current_primary != info['primary_value']:
            ax.axhline(y=i - 0.5, color='white', linewidth=3, linestyle='-')
            source_primary_boundaries.append(i)
            current_primary = info['primary_value']

    # 目标聚类边界（垂直线）
    target_primary_boundaries = []
    current_primary = None
    for j, info in enumerate(target_cluster_info):
        if current_primary is None:
            current_primary = info['primary_value']
        elif current_primary != info['primary_value']:
            ax.axvline(x=j - 0.5, color='white', linewidth=3, linestyle='-')
            target_primary_boundaries.append(j)
            current_primary = info['primary_value']

    # 设置坐标轴刻度
    ax.set_xticks(np.arange(len(target_cluster_info)))
    ax.set_yticks(np.arange(len(source_cluster_info)))
    ax.set_xticklabels(short_target_labels, fontsize=8, rotation=90, ha='center')
    ax.set_yticklabels(short_source_labels, fontsize=8)

    # 添加数值标签
    for i in range(len(source_cluster_info)):
        for j in range(len(target_cluster_info)):
            text = ax.text(j, i, f'{cluster_avg_matrix[i, j]:.0f}',
                           ha="center", va="center", color="white", fontsize=7)

    # 设置标题
    if direction == '16_to_32':
        title = f'16阶群({source_primary}→{source_secondary})到32阶群({target_primary}→{target_secondary})聚类块间平均同态数量\n（对数刻度，白色粗线:主聚类边界）'
    else:
        title = f'32阶群({source_primary}→{source_secondary})到16阶群({target_primary}→{target_secondary})聚类块间平均同态数量\n（对数刻度，白色粗线:主聚类边界）'

    ax.set_title(title, fontsize=14, pad=20)

    # 设置轴标签
    if direction == '16_to_32':
        ax.set_xlabel(f'32阶群聚类块', fontsize=12)
        ax.set_ylabel(f'16阶群聚类块', fontsize=12)
    else:
        ax.set_xlabel(f'16阶群聚类块', fontsize=12)
        ax.set_ylabel(f'32阶群聚类块', fontsize=12)

    # 添加颜色条
    cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('平均同态数量（对数刻度: log10(avg+1))', fontsize=12)

    plt.tight_layout()

    # 保存图像
    output_filename = f'{direction}_聚类块间平均同态数量二级聚类热力图.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"聚类块热力图已保存为 '{output_filename}'")

    plt.show()


# ============================== 获取二级聚类信息 ==============================
print(f"\n正在获取16阶群二级聚类信息（主聚类: {PRIMARY_CLUSTER_16}, 次聚类: {SECONDARY_CLUSTER_16}）...")
summary_16_sorted, sorted_ids_16, cluster_info_16, primary_boundaries_16, secondary_boundaries_16 = get_hierarchical_cluster_info(
    summary_16_df, PRIMARY_CLUSTER_16, SECONDARY_CLUSTER_16
)

print(f"\n正在获取32阶群二级聚类信息（主聚类: {PRIMARY_CLUSTER_32}, 次聚类: {SECONDARY_CLUSTER_32}）...")
summary_32_sorted, sorted_ids_32, cluster_info_32, primary_boundaries_32, secondary_boundaries_32 = get_hierarchical_cluster_info(
    summary_32_df, PRIMARY_CLUSTER_32, SECONDARY_CLUSTER_32
)

# 打印聚类信息
print_hierarchical_cluster_statistics(cluster_info_16, PRIMARY_CLUSTER_16, SECONDARY_CLUSTER_16, "16")
print_hierarchical_cluster_statistics(cluster_info_32, PRIMARY_CLUSTER_32, SECONDARY_CLUSTER_32, "32")

# 显示排序方式
print(f"\n排序方式说明:")
print(f"  主聚类({PRIMARY_CLUSTER_16}, {PRIMARY_CLUSTER_32}): 总是从小到大排序")
print(f"  次聚类({SECONDARY_CLUSTER_16}, {SECONDARY_CLUSTER_32}):")

descending_secondary_columns = ['CenterOrder', 'AbelianizationOrder']
if SECONDARY_CLUSTER_16 in descending_secondary_columns:
    print(f"    16阶次聚类({SECONDARY_CLUSTER_16}): 从大到小排序（值越大越接近阿贝尔群）")
else:
    print(f"    16阶次聚类({SECONDARY_CLUSTER_16}): 从小到大排序")

if SECONDARY_CLUSTER_32 in descending_secondary_columns:
    print(f"    32阶次聚类({SECONDARY_CLUSTER_32}): 从大到小排序（值越大越接近阿贝尔群）")
else:
    print(f"    32阶次聚类({SECONDARY_CLUSTER_32}): 从小到大排序")

# ============================== 创建16阶到32阶的热力图（二级聚类） ==============================
print("\n" + "=" * 60)
print("正在创建16阶群到32阶群的同态计数热力图（二级聚类）...")
print("=" * 60)

hom_16_to_32_sorted = create_cross_group_heatmap_with_hierarchical_clustering(
    hom_16_to_32_df.set_index('Id'),
    summary_16_sorted,
    summary_32_sorted,
    cluster_info_16,
    cluster_info_32,
    sorted_ids_16,
    sorted_ids_32,
    PRIMARY_CLUSTER_16,
    SECONDARY_CLUSTER_16,
    PRIMARY_CLUSTER_32,
    SECONDARY_CLUSTER_32,
    '16_to_32'
)

# ============================== 创建32阶到16阶的热力图（二级聚类） ==============================
print("\n" + "=" * 60)
print("正在创建32阶群到16阶群的同态计数热力图（二级聚类）...")
print("=" * 60)

hom_32_to_16_sorted = create_cross_group_heatmap_with_hierarchical_clustering(
    hom_32_to_16_df.set_index('Id'),
    summary_32_sorted,
    summary_16_sorted,
    cluster_info_32,
    cluster_info_16,
    sorted_ids_32,
    sorted_ids_16,
    PRIMARY_CLUSTER_32,
    SECONDARY_CLUSTER_32,
    PRIMARY_CLUSTER_16,
    SECONDARY_CLUSTER_16,
    '32_to_16'
)

# ============================== 创建聚类块间平均同态数量热力图（二级聚类） ==============================
print("\n" + "=" * 60)
print("正在创建聚类块间平均同态数量热力图（二级聚类）...")
print("=" * 60)

# 计算16阶聚类块到32阶聚类块的平均同态数量
cluster_avg_16_to_32 = np.zeros((len(cluster_info_16), len(cluster_info_32)))

for i, info_i in enumerate(cluster_info_16):
    ids_i = info_i['ids']
    for j, info_j in enumerate(cluster_info_32):
        ids_j = info_j['ids']
        # 计算聚类块i到聚类块j的平均同态数量
        sub_matrix = hom_16_to_32_sorted.loc[ids_i, ids_j]
        cluster_avg_16_to_32[i, j] = np.mean(sub_matrix.values)

# 创建16到32聚类块间热力图
create_cluster_blocks_heatmap(
    cluster_avg_16_to_32,
    cluster_info_16,
    cluster_info_32,
    PRIMARY_CLUSTER_16,
    SECONDARY_CLUSTER_16,
    PRIMARY_CLUSTER_32,
    SECONDARY_CLUSTER_32,
    '16_to_32'
)

# 计算32阶聚类块到16阶聚类块的平均同态数量
cluster_avg_32_to_16 = np.zeros((len(cluster_info_32), len(cluster_info_16)))

for i, info_i in enumerate(cluster_info_32):
    ids_i = info_i['ids']
    for j, info_j in enumerate(cluster_info_16):
        ids_j = info_j['ids']
        # 计算聚类块i到聚类块j的平均同态数量
        sub_matrix = hom_32_to_16_sorted.loc[ids_i, ids_j]
        cluster_avg_32_to_16[i, j] = np.mean(sub_matrix.values)

# 创建32到16聚类块间热力图
create_cluster_blocks_heatmap(
    cluster_avg_32_to_16,
    cluster_info_32,
    cluster_info_16,
    PRIMARY_CLUSTER_32,
    SECONDARY_CLUSTER_32,
    PRIMARY_CLUSTER_16,
    SECONDARY_CLUSTER_16,
    '32_to_16'
)

# ============================== 保存详细信息到CSV文件 ==============================
print("\n" + "=" * 60)
print("正在保存详细的聚类信息到CSV文件...")
print("=" * 60)

# 保存16阶群聚类信息
cluster_details_16 = []
for i, info in enumerate(cluster_info_16):
    for group_id in info['ids']:
        group_info = summary_16_df[summary_16_df['Id'] == group_id].iloc[0]
        cluster_details_16.append({
            '主聚类值': info['primary_value'],
            '次聚类值': info['secondary_value'],
            '聚类块编号': f"{info['primary_index'] + 1}.{info['secondary_index'] + 1}",
            '群ID': group_id,
            '群名称': group_info['Name'],
            '指数': group_info['Exponent'],
            '中心阶': group_info['CenterOrder'],
            '导出子群阶': group_info['DerivedOrder'],
            '阿贝尔化阶': group_info['AbelianizationOrder'],
            '幂零类': group_info['NilpotencyClass']
        })

cluster_details_16_df = pd.DataFrame(cluster_details_16)
output_filename_16 = f'16阶群_{PRIMARY_CLUSTER_16}→{SECONDARY_CLUSTER_16}_二级聚类详细信息.csv'
cluster_details_16_df.to_csv(output_filename_16, index=False, encoding='utf-8-sig')
print(f"16阶群聚类详细信息已保存为 '{output_filename_16}'")

# 保存32阶群聚类信息
cluster_details_32 = []
for i, info in enumerate(cluster_info_32):
    for group_id in info['ids']:
        group_info = summary_32_df[summary_32_df['Id'] == group_id].iloc[0]
        cluster_details_32.append({
            '主聚类值': info['primary_value'],
            '次聚类值': info['secondary_value'],
            '聚类块编号': f"{info['primary_index'] + 1}.{info['secondary_index'] + 1}",
            '群ID': group_id,
            '群名称': group_info['Name'],
            'AssignedClass': group_info['AssignedClass'] if 'AssignedClass' in group_info else '',
            '指数': group_info['Exponent'],
            '中心阶': group_info['CenterOrder'],
            '导出子群阶': group_info['DerivedOrder'],
            '阿贝尔化阶': group_info['AbelianizationOrder'],
            '幂零类': group_info['NilpotencyClass']
        })

cluster_details_32_df = pd.DataFrame(cluster_details_32)
output_filename_32 = f'32阶群_{PRIMARY_CLUSTER_32}→{SECONDARY_CLUSTER_32}_二级聚类详细信息.csv'
cluster_details_32_df.to_csv(output_filename_32, index=False, encoding='utf-8-sig')
print(f"32阶群聚类详细信息已保存为 '{output_filename_32}'")

# 保存聚类块间平均同态数量
cluster_avg_16_to_32_df = pd.DataFrame(
    cluster_avg_16_to_32,
    index=[
        f"16阶聚类块{i + 1}:{PRIMARY_CLUSTER_16}={info['primary_value']},{SECONDARY_CLUSTER_16}={info['secondary_value']}"
        for i, info in enumerate(cluster_info_16)],
    columns=[
        f"32阶聚类块{j + 1}:{PRIMARY_CLUSTER_32}={info['primary_value']},{SECONDARY_CLUSTER_32}={info['secondary_value']}"
        for j, info in enumerate(cluster_info_32)]
)
cluster_avg_16_to_32_df.to_csv('16到32聚类块间平均同态数量_二级聚类.csv', encoding='utf-8-sig')
print("16到32聚类块间平均同态数量已保存为 '16到32聚类块间平均同态数量_二级聚类.csv'")

cluster_avg_32_to_16_df = pd.DataFrame(
    cluster_avg_32_to_16,
    index=[
        f"32阶聚类块{i + 1}:{PRIMARY_CLUSTER_32}={info['primary_value']},{SECONDARY_CLUSTER_32}={info['secondary_value']}"
        for i, info in enumerate(cluster_info_32)],
    columns=[
        f"16阶聚类块{j + 1}:{PRIMARY_CLUSTER_16}={info['primary_value']},{SECONDARY_CLUSTER_16}={info['secondary_value']}"
        for j, info in enumerate(cluster_info_16)]
)
cluster_avg_32_to_16_df.to_csv('32到16聚类块间平均同态数量_二级聚类.csv', encoding='utf-8-sig')
print("32到16聚类块间平均同态数量已保存为 '32到16聚类块间平均同态数量_二级聚类.csv'")

# ============================== 显示汇总信息 ==============================
print("\n" + "=" * 60)
print("汇总信息")
print("=" * 60)

print(f"\n16阶群二级聚类汇总:")
primary_groups_16 = {}
for info in cluster_info_16:
    primary_val = info['primary_value']
    if primary_val not in primary_groups_16:
        primary_groups_16[primary_val] = 0
    primary_groups_16[primary_val] += info['size']

for primary_val, count in primary_groups_16.items():
    print(f"  主聚类 {PRIMARY_CLUSTER_16} = {primary_val}: 共{count}个群")

print(f"\n32阶群二级聚类汇总:")
primary_groups_32 = {}
for info in cluster_info_32:
    primary_val = info['primary_value']
    if primary_val not in primary_groups_32:
        primary_groups_32[primary_val] = 0
    primary_groups_32[primary_val] += info['size']

for primary_val, count in primary_groups_32.items():
    print(f"  主聚类 {PRIMARY_CLUSTER_32} = {primary_val}: 共{count}个群")

print(f"\n详细聚类块信息:")
print(f"  16阶群: {len(cluster_info_16)} 个聚类块")
print(f"  32阶群: {len(cluster_info_32)} 个聚类块")

print("\n" + "=" * 60)
print("配置选项说明")
print("=" * 60)
print("当前使用二级聚类:")
print(f"  16阶群: 主聚类={PRIMARY_CLUSTER_16}, 次聚类={SECONDARY_CLUSTER_16}")
print(f"  32阶群: 主聚类={PRIMARY_CLUSTER_32}, 次聚类={SECONDARY_CLUSTER_32}")

print(f"\n排序规则:")
print("  1. 主聚类总是从小到大排序")
print("  2. 次聚类:")
print(f"     - 对于'CenterOrder'和'AbelianizationOrder': 从大到小排序")
print(f"     - 对于其他列: 从小到大排序")
print(f"  3. 每个聚类块内: 按群ID从小到大排序")

print(f"\n当前应用的排序:")
descending_secondary_columns = ['CenterOrder', 'AbelianizationOrder']
if SECONDARY_CLUSTER_16 in descending_secondary_columns:
    print(f"  16阶群次聚类({SECONDARY_CLUSTER_16}): 从大到小")
else:
    print(f"  16阶群次聚类({SECONDARY_CLUSTER_16}): 从小到大")

if SECONDARY_CLUSTER_32 in descending_secondary_columns:
    print(f"  32阶群次聚类({SECONDARY_CLUSTER_32}): 从大到小")
else:
    print(f"  32阶群次聚类({SECONDARY_CLUSTER_32}): 从小到大")

print("\n可以修改以下配置选项来调整聚类方式:")
print("  PRIMARY_CLUSTER_32: 32阶群的主聚类依据")
print("  SECONDARY_CLUSTER_32: 32阶群的次聚类依据")
print("  PRIMARY_CLUSTER_16: 16阶群的主聚类依据")
print("  SECONDARY_CLUSTER_16: 16阶群的次聚类依据")
print("\n可选值:")
print("  对于32阶群主聚类: 'AssignedClass', 'Exponent', 'CenterOrder', 'DerivedOrder',")
print("                     'AbelianizationOrder', 'NilpotencyClass'")
print("  对于16阶群主聚类: 'Exponent', 'CenterOrder', 'DerivedOrder',")
print("                     'AbelianizationOrder', 'NilpotencyClass'")
print("  对于次聚类: 同上（建议选择与主聚类不同的不变量）")
print("=" * 60)