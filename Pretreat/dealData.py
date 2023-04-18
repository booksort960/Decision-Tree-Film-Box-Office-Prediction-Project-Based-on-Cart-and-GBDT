"""
文件名：dealData.py
创建日期：2023-04-14 13:00
描述：该文件包含以下模块和作用：
        获取excel表格的行索引和列索引；
        将表格转换为矩阵后对其数据进行相关需求处理；
        最终返回处理后的矩阵；
        将原excel表格转换为矩阵后重新保存到一个新excel表格；
        将处理后的矩阵再保存到一个新的excel表格
"""
import numpy as np
import pandas as pd
# import datetime
from datetime import datetime


# 将列索引和行索引分别存放在字典中
def get_col_dict(df):
    cols = df.iloc[0].tolist()
    col_dict = {i: col for i, col in enumerate(cols)}
    col_dict1 = {value: key for key, value in col_dict.items()}
    return col_dict1


def get_row_dict(df):
    row_dict1 = {}
    for i, row_value in enumerate(df.iloc[:, 15]):
        if row_value == 'title':
            continue
        row_dict1[row_value] = i - 1
    return row_dict1


# 离散列字典
def get_rating_dict(df):
    # 转化为列表
    My_twelve_list = df.iloc[:, 11].tolist()
    # 将列表转化为为集合
    my_rating_dict12 = {}
    my_set12 = set(df.iloc[:, 11])
    for i, item in enumerate(my_set12):
        my_rating_dict12[item] = i
    return my_rating_dict12


# 自定义字典
def definition_dict():
    my_dict = {'audience_freshness': 0, 'rt_audience_score': 1, 'rt_freshness': 2, 'rt_score': 3, '2015_inflation': 4,
               'Crime': 5, 'Western': 6, 'Sport': 7, 'Family': 8, 'Romance': 9, 'Action': 10, 'Music': 11, 'Horror': 12,
               'Comedy': 13, 'Drama': 14, 'Musical': 15, 'Documentary': 16, 'Biography': 17, 'Mystery': 18, 'War': 19,
               'Animation': 20, 'Fantasy': 21, 'Adventure': 22, 'Thriller': 23, 'Sci-Fi': 24, 'History': 25,
               'Rastar': 26, 'American International Pictures': 27, 'Summit': 28, 'Miramax Films': 29,
               'Illumination': 30, 'Universal': 31, 'DreamWorks': 32, 'Carolco': 33, 'Lionsgate Films': 34,
               'United Artists': 35, 'Castle Rock Entertainment': 36, 'Sunn Classic Pictures': 37, 'Legendary': 38,
               '20th Century Fox Film Corporation': 39, 'Golden Harvest': 40, 'Lorimar': 41, 'New Line Cinema': 42,
               'Associated Film Distribution': 43, 'United Film Distribution Company': 44, 'Walt Disney Pictures': 45,
               'Hollywood Pictures': 46, 'Warner Bros': 47, 'PolyGram': 48, 'Walt Disney Productions': 49,
               'DreamWorks Pictures': 50, 'Lionsgate': 51, 'Amblin Entertainment': 52, 'Lucasfilm': 53,
               'Universal Studios': 54, 'Silver Pictures': 55, 'Newmarket': 56, 'Marvel Studios': 57,
               'National Air and Space Museum': 58, 'Village Roadshow': 59, 'Embassy Pictures': 60,
               'Lightstorm Entertainment': 61, 'Paramount Pictures': 62, 'Blue Sky': 63, 'Icon': 64,
               'Columbia Pictures': 65, 'Imagine': 66, 'Disney': 67, 'Touchstone Pictures': 68,
               'Paramount': 69, 'Marvel': 70, 'Carolco Pictures': 71,
               'Fox Searchlight Pictures': 72, 'Pixar': 73, 'New Line': 74, 'Touchstone': 75,
               'Orion Pictures': 76, 'Gramercy Pictures': 77, 'Warner Bros. Pictures': 78, 'Sony Pictures': 79,
               'MGM': 80, 'Ladd': 81, '20th Century Fox': 82, 'RKO Pictures': 83, 'Cinergi Pictures': 84,
               'Walden Media': 85, 'United Artists Pictures': 86, 'Universal Pictures': 87,
               'Metro-Goldwyn-Mayer': 88, 'The Ladd Company': 89, 'Cinema International Corporation': 90,
               'Columbia': 91, 'Summit Entertainment': 92, 'Nelson Entertainment': 93, 'Geffen Pictures': 94,
               'Imagine Entertainment': 95, 'IFC Films': 96, 'Fox': 97, 'Gracie Films': 98, 'ITC': 99,
               'TriStar Pictures': 100, 'Buena Vista Pictures Distribution': 101, 'imdb_rating': 102,
               'length': 103, 'rank_in_year': 104, 'rating': 105, 'release_date': 106, 'worldwide_gross': 107}
    return my_dict


# 处理年份
def get_year_dict(input_file_path, input_file_path_one):
    df = pd.read_excel(input_file_path, header=None)
    df1 = pd.read_excel(input_file_path_one)
    df1 = df1.astype(str)  # 将datafram中的数据转化为str类型
    matrix = df.drop(0).drop(columns=15).values
    # print(matrix)
    new_matrix = df1.values
    # print(new_matrix)
    # for i in range(397):
    #     print(new_matrix[i][106])
    #     print(type(new_matrix[i][106]))

    row_dict = get_row_dict(df)
    col_dict = get_col_dict(df)
    my_dict = definition_dict()
    keys1 = row_dict.keys()  # 行名
    keys2 = col_dict.keys()
    keys3 = my_dict.keys()  # 列名
    year_dict = {}
    for key2 in keys2:
        for key3 in keys3:
            if key2 == 'release_date':
                if key3 == key2:
                    for key1 in keys1:
                        i = row_dict[key1]
                        j = col_dict[key2]
                        if j == 16:
                            break
                        value = matrix[i, j]
                        k = my_dict[key3]
                        new_matrix[i][k] = '2023-4-12'
                        new_matrix[i][k] = value
                        value1 = new_matrix[i][k]
                        # print(value)
                        # print(type(value))
                        date_string = value1.strftime('%Y-%m-%d')
                        year, month, day = date_string.split('-')
                        year = year[-4:]
                        month = month.zfill(2)
                        day = day.zfill(2)
                        # print(date_string)
                        # print(date_string)
                        # print(type(date_string))
                        value2 = datetime.strptime(date_string, '%Y-%m-%d')
                        # print(type(value2))
                        # print(value2)
                        year_dict[value2] = year + month + day
                        # print(year_dict.values())

                        keys4 = year_dict.keys()
                        # print(keys4)
                        for key4 in keys4:
                            value4 = year_dict[key4]
                            if new_matrix[i][k] == key4:
                                new_matrix[i][k] = value4
                                # print(new_matrix[i][k])
                else:
                    continue
            else:
                continue
    new_matrix_one = new_matrix
    # print(new_matrix_one)
    return new_matrix_one


def deal_matrix(input_file_path):
    df = pd.read_excel(input_file_path, header=None)
    matrix = df.drop(0).drop(columns=15).values
    # 索引字典调用
    col_dict = get_col_dict(df)
    row_dict = get_row_dict(df)
    keys1 = col_dict.keys()  # 列名，列索引字典键
    keys2 = row_dict.keys()  # 行名，行索引字典键
    # print(row_dict)
    # 离散列字典调用
    my_rating_dict12 = get_rating_dict(df)
    keys3 = my_rating_dict12.keys()
    # 年份列字典调用
    # year_dict = get_year_dict(df)
    # keys4 = year_dict.keys()
    # print(keys4)
    for key1 in keys1:
        for key2 in keys2:
            value1 = col_dict[key1]
            value2 = row_dict[key2]
            if key1 == 'rating':
                # print(1)
                for key3 in keys3:
                    value3 = my_rating_dict12[key3]
                    if matrix[value2, value1] == key3:
                        matrix[value2, value1] = value3
    for row in keys2:
        value = row_dict[row]
    # 数据处理

    #自定义字典
    my_dict = definition_dict()
    keys5 = my_dict.keys()  #'release_date': 106,
    # print(My_dict.items())
    n = 397
    m = 108
    count = 0
    new_matrix = np.zeros((n, m))

    # 类型三列分类展开
    for key2 in keys2:
        for key1 in keys1:
            value1 = col_dict[key1]
            value2 = row_dict[key2]
            if key1 == 'Genre_1':
                list_tmp = [matrix[value2, value1], matrix[value2, value1 + 1], matrix[value2, value1 + 2]]
                # print(list_tmp)
                for i in range(3):
                    for key5 in my_dict:
                        # print(key5)
                        value3 = my_dict[key5]
                    # for key5, k in My_dict.items():
                        # print(value2,k)
                        # print(value3)
                        if list_tmp[i] == key5:
                            # print(list_tmp[i], key5)
                            new_matrix[value2, value3] = 1
                        # else:
                        #     new_matrix[value2, value3] = 2
            else:
                continue
    for key_2 in keys2:
        for key_year in keys5:
            if key_year == 'release_date':
                value_2 = row_dict[key_2]
                value_year = my_dict[key_year]
                new_matrix[value_2][value_year] = int(10)
    # for key in keys2:
    #     value4 = row_dict[key]
    #     print(new_matrix[value4][102])
    # 公司分类展开
    for key2 in keys2:
        for key1 in keys1:
            value1 = col_dict[key1]
            value2 = row_dict[key2]
            if key1 == 'studio1':
                list_tmp = [matrix[value2, value1], matrix[value2, value1 + 1]]
                # print(list_tmp)
                for i in range(2):
                    for key5 in my_dict:
                        # print(key5)
                        value3 = my_dict[key5]

                        if list_tmp[i] == key5:
                            # print(list_tmp[i], key5)
                            new_matrix[value2, value3] = 1
            else:
                continue

        # 剩余数据赋值
        for key1 in keys1:  # 保持matrix列不变，行先赋值
            for key3 in keys5:  # 遍历自定义字典，找到与matrix列对应列
                value1 = col_dict[key1]  # 获取matrix列字典值作为下标
                if value1 == 16:
                    value1 -= 1
                value3 = my_dict[key3]  # 获取自定义字典值作为new_matrix下标
                if key1 == key3 and key1 != 'release_date':
                    for key_2 in keys2:  # 按行进行赋值
                        value2 = row_dict[key_2]
                        new_matrix[value2, value3] = matrix[value2, value1]
                else:
                    continue
    return new_matrix


def set_fun(input_file_path):
    df = pd.read_excel(input_file_path, header=None)
    # 索引字典调用
    col_dict = get_col_dict(df)
    row_dict = get_row_dict(df)
    keys1 = col_dict.keys()  # 列名，列索引字典键
    keys2 = row_dict.keys()  # 行名，行索引字典键
    for key1 in keys1:
        for key2 in keys2:
            value1 = col_dict[key1]
            value2 = row_dict[key2]
            if key1 == 'Genre_1':
                n = col_dict[key1]
                Genre1_list = df.drop(0).iloc[:, n].tolist()
            if key1 == 'Genre_2':
                n = col_dict[key1]
                Genre2_list = df.drop(0).iloc[:, n].tolist()
            if key1 == 'Genre_3':
                n = col_dict[key1]
                Genre3_list = df.drop(0).iloc[:, n].tolist()
            if key1 == 'studio1':
                n = col_dict[key1]
                studio1 = df.drop(0).iloc[:, n].tolist()
            if key1 == 'studio2':
                n = col_dict[key1]
                studio2 = df.drop(0).iloc[:, n].tolist()
    total_list1 = Genre1_list + Genre2_list + Genre3_list
    set_result1 = set(total_list1)
    genre_dict = {}
    for i, item in enumerate(set_result1):
        genre_dict[i] = item
    genre_dict = {value: key for key, value in genre_dict.items()}
    # print(genre_dict)
    # set_result = set(total_list)
    set_result1.discard(0)
    total_list2 = studio1 + studio2
    set_result2 = set(total_list2)
    # print(len(genre_dict))
    coporation_dict = {}
    for i, item in enumerate(set_result2):
        coporation_dict[i] = item
    coporation_dict = {value: key for key, value in coporation_dict.items()}
    # print(len(coporation_dict))
    # print(coporation_dict)
    # set_result2.discard(0)
    # klen = len(set_result2)
    return genre_dict, coporation_dict
# 返回列，行，矩阵
def matrix_row_column_dict(input_file_path, input_file_path_one):
    df = pd.read_excel(input_file_path, header=None)
    row_dict = get_row_dict(df)  # 行索引
    my_dict = definition_dict()  # 列索引
    later_matrix = get_year_dict(input_file_path, input_file_path_one)  # 处理后矩阵
    return row_dict, my_dict, later_matrix

# def main()
def svd_convert(spa_mat):
    _, s, _ = np.linalg.svd(spa_mat)
    m, n = spa_mat.shape
    rearr = np.zeros((m, 1), float)
    for a in range(m):
        valsum = 0
        for b in range(n):
            if spa_mat[a, b] != 0:
                valsum += spa_mat[a, b] * s[b]
        rearr[a] = round(valsum, 3) * 10
    return rearr


def final_dict():
    my_dict2 = {'audience_freshness': 0, 'rt_audience_score': 1, 'rt_freshness': 2, 'rt_score': 3, '2015_inflation': 4,
                  'genre': 5, 'corporation': 6, 'imdb_rating': 7, 'length': 8, 'rank_in_year': 9, 'rating': 10,
                  'release_date': 11, 'worldwide_gross': 12}
    return my_dict2

#矩阵降维处理
def dimension_reduction(output_file_path):
    df = pd.read_excel(output_file_path)
    df1 = df.astype(int)  # 将datafram中的数据转化为int类型
    newMatrix = df1.values
    my_dict = definition_dict()
    keys3 = my_dict.keys()
    dict1, dict2 = set_fun(input_file_path)
    keys1 = dict1.keys()
    keys2 = dict2.keys()
    n = 397
    m = len(dict1)
    # print(m)
    genre_matrix = np.zeros((n, m), dtype=int)
    # print(genre_matrix)
    for key1 in keys1:  # 小字典里面的键
        for key3 in keys3:  # 大字典里面的键
            if key1 == key3:
                value1 = dict1[key1]
                value3 = my_dict[key3]
                for i in range(397):
                    genre_matrix[i][value1] = newMatrix[i][value3]
                    # print(genre_matrix[i][value1])
    # print(genre_matrix)
    l = len(dict2)
    coporation_matrix = np.zeros((n, l), dtype=int)
    # print(genre_matrix)
    for key2 in keys2:  # 小字典里面的键
        for key3 in keys3:  # 大字典里面的键
            if key2 == key3:
                # print(key2)
                value2 = dict2[key2]
                value3 = my_dict[key3]
                for i in range(397):
                    coporation_matrix[i][value2] = newMatrix[i][value3]
    # print(coporation_matrix)
    genre_matrix = svd_convert(genre_matrix)
    coporation_matrix = svd_convert(coporation_matrix)
    # print(genre_matrix)
    # print(coporation_matrix)
    result_matrix = np.concatenate((genre_matrix, coporation_matrix), axis=1)
    return result_matrix

#最终降维处理后合并的矩阵
def final_matrix():
    df = pd.read_excel(output_file_path)
    matrix = df.values
    my_dict = definition_dict()
    keys2 = my_dict.keys()
    my_dict2 = final_dict()
    keys1 = my_dict2.keys()
    keys = my_dict2.keys()
    # for key in keys:
    #     print(my_dict2[key])
    #将其类型和公司矩阵降维处理
    dimesion_matrix = dimension_reduction(output_file_path)
    # print(dimesion_matrix)
    n = 397
    m = len(my_dict2)
    # print(m)
    final_matrix = np.zeros((n, m))
    for key1 in keys1:  # 新矩阵需要的键值
        for key2 in keys2:  # 原大矩阵的键值
            value1 = my_dict2[key1]
            # print(value1)
            value2 = my_dict[key2]
            if key1 == key2:
                for i in range(397):
                    final_matrix[i][value1] = matrix[i][value2]
            if key1 == 'genre':
                for i in range(397):
                    final_matrix[i][value1] = dimesion_matrix[i][0]
            if key1 == 'corporation':
                for i in range(397):
                    final_matrix[i][value1] = dimesion_matrix[i][1]
    return final_matrix


"""
    函数综合调用实现模块
"""
def process_date_file(input_file_path, input_file_path_one, output_file_path, fin_file_path):
    # 先处理扩展列生成第一份文件output.xlsx
    new_matrix = deal_matrix(input_file_path)
    new_df = pd.DataFrame(new_matrix)
    new_df.to_excel(input_file_path_one, index=False)
    #再处理output.xlsx生成finaloutput.xlsx文件
    new_matrix_one = get_year_dict(input_file_path, input_file_path_one)
    new_matrix_one_df = pd.DataFrame(new_matrix_one)
    new_matrix_one_df.to_excel(output_file_path, index=False)
    #再对finaloutput.xlsx文件进行降维合并处理最终得到lastoutput.xlsx文件
    final_matr = final_matrix()
    fin_df = pd.DataFrame(final_matr)
    fin_df.to_excel(fin_file_path, index=False)


""" 对接函数
    输入文件路径即可返回文件矩阵和列索引字典
"""
def dictmatrix(fin_file_path):
    df = pd.read_excel(fin_file_path)
    _matrix = df.to_numpy()
    # print(_matrix)
    _mydict = final_dict()
    return _matrix, _mydict


if __name__ == '__main__':
    input_file_path = 'Date.xlsx'
    input_file_path_one = 'output.xlsx'
    output_file_path = 'finaloutput.xlsx'
    fin_file_path = 'lastoutputfile.xlsx'
    # print(newMatrix)
    process_date_file(input_file_path, input_file_path_one, output_file_path,fin_file_path)
    ma, b = dictmatrix(fin_file_path)










    # print(result_matrix)
    # x = result_matrix[396][1]
    # print(x)


    # a, b, c = matrix_row_column_dict(input_file_path, input_file_path_one)
    # print(a)
    # print(b)
    # print(c)
