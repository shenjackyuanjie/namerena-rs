from __future__ import annotations

import sys
import json
import sqlite3
import subprocess

# 线程池
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DB_FILE = Path("./data.db")

NEW_DATA_FILE = Path("./new_data.txt")
NEW_DATA_OUT_FILE = Path("./new_data.txt-out.txt")

TABLE_NAME = "namerena"

THREADS = 4

def get_db_conn(**args) -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, **args)
    # 创建表
    # name: 字符串, 主键
    # pp: int, 
    # pd: int, 
    # qp: int, 
    # qd: int, 
    # sum: int, pp + pd + qp + qd
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            name TEXT PRIMARY KEY,
            pp INTEGER,
            pd INTEGER,
            qp INTEGER,
            qd INTEGER,
            sum INTEGER
        );
        """
    )
    conn.commit()
    return conn


def check_name_exist(cursor: sqlite3.Cursor, name: str) -> bool:
    cursor.execute(f"SELECT name FROM {TABLE_NAME} WHERE name = ?", (name,))
    return bool(cursor.fetchone())


def check_names_not_exist(cursor: sqlite3.Cursor, names: list[str]) -> list[str]:
    placeholders = ','.join('?' for _ in names)
    cursor.execute(f"SELECT name FROM {TABLE_NAME} WHERE name NOT IN ({placeholders})", names)
    return list(set(row[0] for row in cursor.fetchall()))


def insert_data(cursor: sqlite3.Cursor, name: str, pp: int, pd: int, qp: int, qd: int) -> None:
    cursor.execute(
        f"INSERT INTO {TABLE_NAME} (name, pp, pd, qp, qd, sum) VALUES (?, ?, ?, ?, ?, ?)",
        (name, pp, pd, qp, qd, pp + pd + qp + qd),
    )


def main():
    if len(sys.argv) < 2:
        print("需要更多参数")
    target = sys.argv[1]
    print(f"正在读取 {target}")

    with open(target, "r", encoding="utf-8") as target_file:
        datas = target_file.readlines()

    datas = list(set(data.split(",")[0] for data in datas))

    stores = Path("./results.json")

    if stores.exists():
        print("result 存在, 进行stores")
        with open(stores, "r", encoding="utf-8") as stores_file:
            old_datas = json.load(stores_file)
        # 转存到 sqlite 里
        conn = get_db_conn(autocommit=False)
        cursor = conn.cursor()
        for name, data in old_datas.items():
            if not check_name_exist(cursor, name):
                insert_data(cursor, name, *data)
        conn.commit()
        conn.close()
    else:
        old_datas = {}

    conn = get_db_conn()
    cursor = conn.cursor()
    # 一一为睠樀㴁@shenjack114514,      245981296,5076.5655,5707.2106,一一为睠樀㴁,shenjack114514,331,85,88,91,98,78,93,81,724.3333,火球-7,命轮-8,狂暴-3,苏生-1,净化-46,蓄力-17,幻术-3,反击-31,
    # datas = check_names_not_exist(cursor, datas)
    datas = [data for data in datas if not check_name_exist(cursor, data)]

    datas = [data for data in datas if data not in old_datas]

    cursor.close()
    conn.close()

    print(f"找到 {len(datas)} 个新数据")
    
    if not datas:
        return

    # 写到文件里
    with open(NEW_DATA_FILE, "w", encoding="utf-8") as new_file:
        new_file.writelines(data + "\n" for data in datas)

    NEW_DATA_OUT_FILE.unlink(missing_ok=True)

    stores_datas = {}

    print("=====pp 1=====")
    subprocess.run(["bun", "run", "md5-api.ts", NEW_DATA_FILE, "pp"])
    with open(NEW_DATA_OUT_FILE, "r", encoding="utf-8") as result_file:
        result_datas = result_file.readlines()
        result_datas = [datas.split("|") for datas in result_datas if datas]
        for name, score in result_datas:
            stores_datas[name] = [float(score)]
            print(f"{name}: {score}")
    NEW_DATA_OUT_FILE.unlink(missing_ok=True)

    print("=====pd 2=====")
    subprocess.run(["bun", "run", "md5-api.ts", NEW_DATA_FILE, "pd"])
    with open(NEW_DATA_OUT_FILE, "r", encoding="utf-8") as result_file:
        result_datas = result_file.readlines()
        result_datas = [datas.split("|") for datas in result_datas if datas]
        for name, score in result_datas:
            stores_datas[name].append(float(score))
            print(f"{name}: {score}")
    NEW_DATA_OUT_FILE.unlink(missing_ok=True)

    print("=====qp 3=====")
    subprocess.run(["bun", "run", "md5-api.ts", NEW_DATA_FILE, "qp"])
    with open(NEW_DATA_OUT_FILE, "r", encoding="utf-8") as result_file:
        result_datas = result_file.readlines()
        result_datas = [datas.split("|") for datas in result_datas if datas]
        for name, score in result_datas:
            stores_datas[name].append(float(score))
            print(f"{name}: {score}")
    NEW_DATA_OUT_FILE.unlink(missing_ok=True)

    print("=====qd 4=====")
    subprocess.run(["bun", "run", "md5-api.ts", NEW_DATA_FILE, "qd"])
    with open(NEW_DATA_OUT_FILE, "r", encoding="utf-8") as result_file:
        result_datas = result_file.readlines()
        result_datas = [datas.split("|") for datas in result_datas if datas]
        for name, score in result_datas:
            stores_datas[name].append(float(score))
            print(f"{name}: {score}")
    NEW_DATA_OUT_FILE.unlink(missing_ok=True)

    # 排个序
    stores_datas = dict(sorted(stores_datas.items(), key=lambda x: sum(x[1]), reverse=True))

    # 写回文件
    conn = get_db_conn()
    cursor = conn.cursor()

    stores_datas = {k: v for k, v in stores_datas.items() if not check_name_exist(cursor, k)}

    for name, data in stores_datas.items():
        try:
            insert_data(cursor, name, *data)
            print(f"{name} 已写入")
        except sqlite3.IntegrityError:
            print(f"{name} 已存在")

    conn.commit()
    print("=====END=====")

    # with open(stores, "w", encoding="utf-8") as stores_file:
    #     json.dump(stores_datas, stores_file, ensure_ascii=False)


if __name__ == "__main__":
    main()