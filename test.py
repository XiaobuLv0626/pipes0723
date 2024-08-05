# 定义一个包含字符串和数字组合的元组的列表
tuple_list = [("apple", 1), ("banana", 2), ("cherry", 3)]

# 定义要检查的目标元组
target_tuple = ("banana", 2)

# 使用 in 运算符检查元组是否在列表中
if target_tuple in tuple_list:
    print(f"{target_tuple} 在列表中")
else:
    print(f"{target_tuple} 不在列表中")
