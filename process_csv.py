# 读取CSV文件并按照标题进行排序
def sort_csv_by_title(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 构建标题与数据的字典
    data_dict = {}
    current_title = None
    current_data = []

    for line in lines:
        line = line.strip()
        if line.startswith("error_bar"):
            if current_title:
                data_dict[current_title] = current_data
            current_title = line
            current_data = []
        else:
            current_data.append(line)

    if current_title:
        data_dict[current_title] = current_data

    # 根据标题排序
    sorted_titles = sorted(data_dict.keys())

    # 重建排序后的数据
    sorted_lines = []
    for title in sorted_titles:
        sorted_lines.append(title)
        sorted_lines.extend(data_dict[title])

    # 将排序后的数据写入文件
    with open("sort.csv", 'w') as file:
        file.write('\n'.join(sorted_lines))

# 测试
sort_csv_by_title('output_image.csv')
