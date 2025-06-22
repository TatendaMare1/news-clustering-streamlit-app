# news_spider.py
import scrapy
from newsplease import NewsPlease
from scrapy.crawler import CrawlerProcess
import pandas as pd
import datetime


class NewsSpider(scrapy.Spider):
    name = 'news_spider'

    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'FEED_FORMAT': 'csv',
        'FEED_URI': 'news.csv'
    }

    newspapers = {
        'TheHerald': {
            'business': 'https://www.heraldonline.co.zw/tag/business/',
            'politics': 'https://www.heraldonline.co.zw/tag/politics/',
            'arts': 'https://www.heraldonline.co.zw/tag/arts/',
            'sports': 'https://www.heraldonline.co.zw/tag/sports/'
        },
        'cnn': {
            'business': 'https://edition.cnn.com/business',
            'politics': 'https://edition.cnn.com/politics',
            'arts': 'https://edition.cnn.com/entertainment',
            'sports': 'https://edition.cnn.com/sport'
        },
        'TheZimbabwean': {
            'business': 'https://www.thezimbabwean.co/category/business/',
            'politics': 'https://www.thezimbabwean.co/category/politics/',
            'arts': 'https://www.thezimbabwean.co/category/entertainment/',
            'sports': 'https://www.thezimbabwean.co/category/sport/'
        },
        'bbc': {
            'business': 'https://www.bbc.com/news/business',
            'politics': 'https://www.bbc.com/news/politics',
            'arts': 'https://www.bbc.com/culture',
            'sports': 'https://www.bbc.com/sport'
        }
    }

    def start_requests(self):
        for paper, sections in self.newspapers.items():
            for section, url in sections.items():
                yield scrapy.Request(url=url, callback=self.parse,
                                     meta={'newspaper': paper, 'section': section})

    def parse(self, response):
        links = response.css('a::attr(href)').getall()
        for link in set(links):
            if link and 'http' in link:
                yield response.follow(link, callback=self.parse_article,
                                      meta=response.meta)

    def parse_article(self, response):
        article = NewsPlease.from_url(response.url)
        if article:
            yield {
                'title': article.title,
                'text': article.maintext,
                'url': response.url,
                'date': article.date_publish,
                'newspaper': response.meta['newspaper'],
                'section': response.meta['section'],
                'authors': article.authors
            }


def run_spider():
    process = CrawlerProcess()
    process.crawl(NewsSpider)
    process.start()


if __name__ == "__main__":
    run_spider()