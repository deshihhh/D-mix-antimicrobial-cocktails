# coding:utf-8
import xlrd
import csv


def importxlsx(filename, sheetname):
    bk = xlrd.open_workbook(filename)
    shxrange = range(bk.nsheets)
    sh = bk.sheet_by_name(sheetname)
    # try:
    # sh = bk.sheet_by_name(sheetname)
    # except:
    #     print("There is no {}".format(sheetname))
    nrows = sh.nrows
    ncols = sh.ncols
    row_list = []
    for i in range(nrows):
        row_data = sh.row_values(i)
        row_list.append(row_data)
        # row_list.append(row_data[2:])     #exculde the first and second column data
    return row_list


def FtoS(list):
    temp = []
    for i in range(len(list)):
        temp.append([list[i]])
    return temp


def importcsv(filename):
    row_list = []
    with open(filename) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # if i!=0:
            row_list.append(row)
    f.close()
    return row_list
