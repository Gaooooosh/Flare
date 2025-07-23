import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (15, 10)

# 读取数据文件
file1 = "/raid_sdh/home/xyg/attention_scores-qwen2.5.csv"
file2 = "/raid_sdh/home/xyg/attention_scores-allrope.csv"
fill3 = "/raid_sdh/home/xyg/attention_scores-somerope.csv"
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(fill3)

df1['Model'] = 'Baseline'
df2['Model'] = 'AllRoPE'
df3['Model'] = 'SomeRoPE'

# 合并数据以便对比
combined_df = pd.concat([df1, df2, df3])

# 创建子图
fig, axes = plt.subplots(2, 2, sharex=True)
axes = axes.flatten()

# 定义要绘制的分数类型
score_types = ['Sink_Score', 'Magic_Score', 'Other_Score', 'Probe_Score']

# 为每种分数类型绘制对比图
for i, score in enumerate(score_types):
    sns.lineplot(data=combined_df, x='Layer', y=score, hue='Model', marker='o', ax=axes[i])
    axes[i].set_title(f'{score} Comparison')
    axes[i].set_xlabel('Layer')
    axes[i].set_ylabel(score)
    axes[i].legend(title='Model')

plt.tight_layout()

# 保存图表
output_path = "/raid_sdh/home/xyg/flare/attention_scores_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"可视化图表已保存至: {output_path}")

# 显示图表
plt.show()