import pandas as pd
import ast

# 1. 读取表格
df = pd.read_csv('去重.csv')

def expand_dict_column(row, col):
    """将某一列中的字典字符串展开为多个字段"""
    try:
        d = ast.literal_eval(row[col]) if pd.notnull(row[col]) else {}
        if isinstance(d, dict):
            for k, v in d.items():
                row[f"{col}_{k}"] = v
    except Exception as e:
        print(f"解析失败: {row[col]}, 错误: {e}")
    return row

# 2. 展开嵌套字段
dict_cols = ['小区解读', '房源概况']  # 你实际表格中嵌套字典的字段名
for col in dict_cols:
    df = df.apply(lambda row: expand_dict_column(row, col), axis=1)

# 3. 删除原始嵌套列
df = df.drop(columns=dict_cols)


# 4. 保存为新CSV/JSONL
#df = pd.read_csv('processed_data.csv')
df.to_csv('processed_data.csv', index=False, encoding='utf-8-sig')
df.to_json('processed_data.jsonl', orient='records', lines=True, force_ascii=False)
