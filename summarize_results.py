import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def summarize_mmlu_results(main_dir):
    # 初始化合并数据的DataFrame
    combined_df = pd.DataFrame()
    
    # 遍历主目录下的所有子文件夹
    for subdir in os.listdir(main_dir):
        subdir_path = os.path.join(main_dir, subdir)
        
        # 检查是否为目录
        if not os.path.isdir(subdir_path):
            continue
        
        # 查找summary.csv文件
        summary_path = os.path.join(subdir_path, 'summary.csv')
        if not os.path.exists(summary_path):
            print(f"警告: {subdir_path} 中未找到summary.csv文件，已跳过")
            continue
        
        # 读取CSV文件
        try:
            df = pd.read_csv(summary_path)
            # 转换Accuracy列为数值类型
            df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
            # 移除Accuracy为NaN的行
            df = df.dropna(subset=['Accuracy'])
            # 添加来源文件夹列
            df['source_folder'] = subdir
            # 合并数据
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            print(f"已成功读取: {summary_path}")
        except Exception as e:
            print(f"读取 {summary_path} 时出错: {str(e)}")
            continue
    
    if combined_df.empty:
        print("未找到任何有效的summary.csv文件")
        return None, None
    
    # 排除任何模型在该任务上准确率为0.0的测试项目
    tasks_to_exclude = combined_df[combined_df['Accuracy'] == 0.0]['Task'].unique()
    filtered_df = combined_df[~combined_df['Task'].isin(tasks_to_exclude)]
    
    if filtered_df.empty:
        print("过滤后没有剩余数据")
        return None, None
    
    # 选择需要的列（Task, source_folder, Accuracy）
    filtered_df = filtered_df[['Task', 'source_folder', 'Accuracy']]
    
    # 重塑数据为宽格式，使用pivot_table处理重复值
    pivot_df = filtered_df.pivot_table(index='Task', columns='source_folder', values='Accuracy', aggfunc='mean').reset_index()
    
    # 转换所有数值列为float类型，处理非数值数据
    numeric_cols = pivot_df.columns.drop('Task')
    pivot_df[numeric_cols] = pivot_df[numeric_cols].apply(pd.to_numeric, errors='coerce').astype(float)
    
    # 计算每行的平均值作为Average列
    pivot_df['Average'] = pivot_df[numeric_cols].mean(axis=1, skipna=True)
    
    # 保存宽格式CSV
    output_csv = os.path.join(main_dir, 'combined_summary.csv')
    pivot_df.to_csv(output_csv, index=False)
    print(f"过滤后的总表已保存至: {output_csv}")
    
    # 生成图表
    output_chart = os.path.join(main_dir, 'summary_accuracy_comparison.png')
    plt.figure(figsize=(15, 10))
    
    # 创建准确率对比柱状图
    ax = sns.barplot(data=filtered_df, x='Task', y='Accuracy', hue='source_folder')
    plt.title('Acc Compare')
    plt.xlabel('SubTask')
    plt.ylabel('Acc')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_chart, dpi=300)
    plt.close()
    print(f"准确率对比图表已保存至: {output_chart}")
    
    return combined_df, output_csv

if __name__ == "__main__":
    # MMLU结果主目录
    mmlu_main_dir = '/raid_sdh/home/xyg/flare/MMLU_result'
    
    # 执行汇总
    df, csv_path = summarize_mmlu_results(mmlu_main_dir)
    
    if df is not None:
        print("汇总完成！")
        print(f"总表路径: {csv_path}")
        print(f"图表路径: {os.path.join(mmlu_main_dir, 'summary_accuracy_comparison.png')}")
    else:
        print("汇总失败，请检查输入文件")