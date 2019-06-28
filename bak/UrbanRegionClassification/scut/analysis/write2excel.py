# -*- encoding:utf-8 -*-
import datetime
import xlwt
from xlwt import *

# add workbook by set encoding='utf-8'
workbook = xlwt.Workbook(encoding='utf-8')
# set border of cell
borders = xlwt.Borders()
borders.left = xlwt.Borders.THIN
borders.right = xlwt.Borders.THIN
borders.top = xlwt.Borders.THIN
borders.bottom = xlwt.Borders.THIN

# normal style of cell --One
style = xlwt.XFStyle()
font = xlwt.Font()
# font.name = 'Times New Roman'
# font.name = 'Arial'
font.name = 'Microsoft YaHei'  # 微软雅黑字体
style.font = font
style.borders = borders

# date related style of cell --Two
dateStyle = xlwt.XFStyle()
dateStyle.num_format_str = 'YYYY-MM-DD hh:mm:ss'
dateStyle.font = font
dateStyle.borders = borders

# title related style of cell --Three
bgstyle = xlwt.XFStyle()
font1 = xlwt.Font()
font1.name = 'Microsoft YaHei'
font1.bold = True
bgstyle.font = font1
# Alignment
al = Alignment()
al.horz = Alignment.HORZ_CENTER
al.vert = Alignment.VERT_CENTER
bgstyle.alignment = al
# pattern
pattern = xlwt.Pattern()
pattern.pattern = xlwt.Pattern.SOLID_PATTERN
pattern.pattern_fore_colour = 22
bgstyle.pattern = pattern
bgstyle.borders = borders

# set color = red, add struck
zeroStyle = xlwt.XFStyle()
font2 = xlwt.Font()
font2.struck_out = True
font2.colour_index = 2
zeroStyle.font = font2
zeroStyle.borders = borders


def dowrite(titles, sheetn, datas, excelname):
    sheet = workbook.add_sheet(sheetn, cell_overwrite_ok=True)
    for i, v in enumerate(titles):
        sheet.write(0, i, v, bgstyle)
    for row, data in enumerate(datas):
        for i, v in enumerate(data):
            sheet.write(row+1, i, v, style)
    # save excel
    workbook.save(excelname)


def test():
    # add sheet by allowing overwrite cell
    sheet = workbook.add_sheet('sheet1', cell_overwrite_ok=True)
    titles = ['序号', '名称', '日期', '状态']
    datas = [
        [1, 'tianzhaixing', '2016-05-19', 1],
        [2, 'tianzx', '2016-05-20', 1],
        [3, '约翰', '2016-05-21', 0]
    ]

    # write title related content
    for i, v in enumerate(titles):
        sheet.write(0, i, v, bgstyle)

    # write data to excel
    for row, data in enumerate(datas):
        idx, name, date, status = data
        sheet.write(row + 1, 0, idx, style)
        sheet.write(row + 1, 1, name, style)
        sheet.write(row + 1, 2, date, dateStyle)
        sheet.write(row + 1, 3, status, style)
        if status == 0:
            sheet.write(row + 1, 3, status, zeroStyle)

    # save excel
    dayTime = datetime.datetime.now().strftime('%Y-%m-%d')
    workbookName = dayTime + '_write2excel.xls'
    workbook.save(workbookName)


if __name__ == '__main__':
    test()