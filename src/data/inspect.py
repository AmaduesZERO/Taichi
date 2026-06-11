import os
import pandas as pd


def show_data_info(file_path):
    if not os.path.exists(file_path):
        print(f"找不到文件，请检查路径对不对: {file_path}")
        return

    print("正在读取文件，请稍等...")
    try:
        # 根据文件后缀自动选择读取方式
        if file_path.endswith('.csv'):
            #utf-8-sig 编码，防乱码
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            print("目前只支持读取 csv 或 excel 文件。")
            return

        # 获取行数和列数
        row_count, col_count = df.shape

        # 获取所有列名并转成列表
        column_names = df.columns.tolist()

        print("\n" + "=" * 30)
        print(f"文件名: {os.path.basename(file_path)}")
        print("=" * 30)
        print(f"总行数 (不含表头): {row_count}")
        print(f"总列数: {col_count}") 
        print("\n包含的列名如下:")

        # 把列名打印出来
        for index, name in enumerate(column_names, start=1):
            print(f"{index}. {name}")

        print("=" * 30 + "\n")

    except Exception as e:
        print(f"读取的时候报错了: {e}")


if __name__ == '__main__':

    target_file = r"C:\Users\JT\PycharmProjects\TaiChi\dy_file\dy_video_csv\merged_dy_data_Allcomments_labeled_cleaned_onehot.csv"

    show_data_info(target_file)