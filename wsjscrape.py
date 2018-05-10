# -*- coding:utf-8 -*-
import urllib
import re
import requests
from pyquery import PyQuery as pq
import csv

with open('datamonetarypolicy.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['headline', 'info', 'summary', 'content'])
    for page in range(1, 380):
        try:
            url = 'https://www.wsj.com/search/term.html?KEYWORDS=monetary%20policy&page=' + str(page)
            user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
            headers = {'User-Agent': user_agent}
            html = requests.get(url, headers=headers).text
            doc = pq(html)
            items = doc('.headline-container').items()
            print('#'+str(page))
            for item in items:
                try:
                    h3 = item.find('.headline').text()
                    info = item.find('.article-info').text()
                    summary = item.find('.summary-container').text()
                    link = item.find('.headline a')
                    url1 = link.attr('href')
                    if url1[0] == "/":
                        url1 = "https://www.wsj.com" + url1
                    html1 = requests.get(url1, headers=headers).text
                    doc = pq(html1)
                    item1 = doc('.wsj-snippet-body')
                    child = item1.children()
                    writer.writerow([h3, info, summary, child])
                    print(page)
                except:
                    continue
        except:
            continue