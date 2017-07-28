# coding=utf-8

import mysql.connector
from mysql.connector import errorcode
import string

# 'password':'password',
# 'host':'10.107.20.15','172.17.0.1''123123'
config = {
    'user': 'root',
    'password': 'password',
    'host': '10.107.20.15',
    'port': '13306',
    'database': 'scenes_new'
}

db = None


def _get_mysql_db():
    try:
        global db
        if db is None:
            db = mysql.connector.connect(**config)
        return db
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print('something is wrong with your user name or password')
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print('database does not exist')
        else:
            print(err)
    return None


# 根据训练号获取分类及其分类名称
def get_classification(_train_no):
    cur = _get_mysql_db().cursor()
    sql = 'SELECT t1.train_no,t2.id,t2.classification FROM train_class t1,classifications t2 WHERE t1.train_no="%s" AND t1.class_id=t2.id' % _train_no
    cur.execute(sql)
    return cur.fetchall()


# 根据分类号获取各个分类下图片存储路径
# 返回 （classification_id,url）
def get_classification_urls(_class_ids):
    cur = _get_mysql_db().cursor()
    # 将分类号转为字符串并按','拼接

    args = ','.join(map(str, _class_ids))
    sql = "SELECT t1.classification_id,t2.url FROM r_images_classifications t1 ,images t2 WHERE t1.classification_id IN (%s) AND t1.img_id = t2.id" % args
    cur.execute(sql)
    return cur.fetchall()


# 获取训练集数量
def get_train_number(_train_no):
    cur = _get_mysql_db().cursor()
    sql = 'SELECT num_train FROM trains WHERE train_no="%s"' % _train_no
    cur.execute(sql)
    data = cur.fetchone()
    return data[0]


# 设置训练集数量
def update_train_number(_train_no, length):
    cur = _get_mysql_db().cursor()
    # 设置训练集数量
    sql = 'UPDATE trains SET num_train= %d WHERE train_no="%s"' % (length, _train_no)
    cur.execute(sql)
    _get_mysql_db().commit()

# cnx=cur=None
# try:
#     cnx=mysql.connector.connect(**config)
# except mysql.connector.Error as err:
#     if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
#         print('something is wrong with your user name or password')
#     elif err.errno == errorcode.ER_BAD_DB_ERROR:
#         print('database does not exist')
#     else:
#         print(err)
# else:
#     cur = cnx.cursor()
#     cur.execute('select * from  {0}'.format("classifications"))
#     for row in cur.fetchall():
#         print(row[2].encode('utf8'))
# finally:
#     if cur:
#         cur.close()
#     if cnx:
#         cnx.close()