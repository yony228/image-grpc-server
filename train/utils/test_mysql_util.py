# coding=utf-8

import mysql_util


def test_get_classification():
    classifications = mysql_util.get_classification("T201707041104293093")
    for _, _id, _name in classifications:
        print(" %s %s " % (_id, _name))


def test_get_classification_urls():
    ids = [2, 3, 4]
    classification_urls = mysql_util.get_classification_urls(ids)
    print(len(classification_urls))

if __name__ == '__main__':
    test_get_classification()
    test_get_classification_urls()
