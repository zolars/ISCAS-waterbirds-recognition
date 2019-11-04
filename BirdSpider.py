# -*- coding:utf-8 -*-

import requests
from tqdm import tqdm
import re
import json

"""
一个简单的python爬虫, 用于爬取 http://www.birder.cn/species 页面内容并整理

Authou: Xin Yifei
Version: 1.1
Date: 2019-06-11
Language: Python3.7.3
Modules:
    - requests
    - re
"""


class BirdSpider(object):
    """类说明
    用于抓取网站信息的爬虫
    """
    print('爬虫准备就绪,开始爬取数据...')

    def __init__(self):
        """函数说明
        类声明
        """
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36'
        self.headers = {'User-Agent': user_agent}

        # 建立匹配库
        self.match_data = {
            'list_pre': '',
            'list': '',
            'pic_pre':  r'<!-- 代码部分begin -->(.*?)</ul>',
            'pic': r'<img src="(.*?)"/></a></li>',
            'name_pre': r'<div class="title clearBox">.*?</div>',
            'name': r'\n +(.*?)<\/b>.*?拉丁名：<i>(.*?)</i> &nbsp;英文名：(.*?)\)',
            'describe_pre': r'<div class="pt17 meassage">(.*?)</div>',
            'describe': r'b>(.*?)：</b>(.*?)<',
            'error': r'参数错误',
        }

        # 初始化数据库
        self.data = {}

        print('类声明完毕...')

    def spider(self):

        # details with targets
        infile = open('./resource/target.txt', encoding='utf-8')
        outfile = open('./resource/bird.json', 'w+', encoding='utf-8')

        for i in tqdm(infile.readlines()):
            self.spider_target(i.split('\t')[
                1][:-1], i.split('\t')[0], model=False)

        json.dump(self.data, outfile, ensure_ascii=False)

        print('爬虫全部运行完毕...')

    def spider_target(self, target, ID, model=False):
        """函数说明
        爬虫入口, 控制页面信息
        """

        url = 'http://www.birder.cn/species/' + target

        self.data[ID] = {}

        self.get_page(url, target)

        if len(self.match('error', self.content)) != 0:
            self.data[ID] = {
                "pic":
                "https://www.interserver.net/tips/wp-content/uploads/2016/10/404error.jpeg",
                "中文名": "暂无信息",
                "拉丁文名": "暂无信息",
                "英文名": "暂无信息",
                "分类": "暂无信息",
                "IUCN 红色名录等级": "暂无信息",
                "描述": "暂无信息",
                "虹膜": "暂无信息",
                "嘴": "暂无信息",
                "脚": "暂无信息",
                "叫声": "暂无信息",
                "分布范围": "暂无信息",
                "分布状况": "暂无信息",
                "习性": "暂无信息",
                "俗名": "暂无信息"
            },
            # print('页面{}未找到...问题已记录'.format(target))
            return False

        # pic url
        pic_content = self.match('pic_pre', self.content)[0]
        pic_result = self.match('pic', pic_content)[0]
        pic_result = 'http://www.birder.cn' + pic_result
        self.data[ID]["pic"] = pic_result

        # foreign names
        name_content = self.match('name_pre', self.content)[0]
        name_result = self.match('name', name_content)
        self.data[ID]["中文名"] = name_result[0][0]
        self.data[ID]["拉丁文名"] = name_result[0][1]
        self.data[ID]["英文名"] = name_result[0][2]

        # describes
        describe_content = self.match('describe_pre', self.content)[0]
        describe_result = self.match('describe', describe_content)

        for i in describe_result:
            self.data[ID][i[0]] = i[1].replace(
                "&nbsp;&nbsp;", "  ").replace("。", "").replace("\"", "'")

        # print('抓取页面{}信息完毕...'.format(target))

    def get_page(self, cur_url, cur_page):
        """函数说明
        根据当前页码抓取网页HTML信息
        """
        temp_page = requests.get(
            url=cur_url,
            headers=self.headers
        )
        # 检测网页问题
        try:
            temp_page.raise_for_status()
        except:
            print(temp_page.status_code)

        self.content = str(temp_page.content, 'utf-8')

    def match(self, tool, content):
        """函数说明
        使用正则表达式匹配传入的网页HTML
        """
        result = re.findall(
            self.match_data[tool],
            content,
            re.S,
        )

        return(result)


def main():
    BirdSpider().spider()


if __name__ == "__main__":
    main()
