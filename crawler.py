import streamlit as st
import json
import os
from pathlib import Path
import hashlib
import scrapy
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from urllib.parse import urlparse
import multiprocessing
from functools import partial

class GeneralSpider(scrapy.Spider):
    name = "general_spider"

    def __init__(self, start_url, max_depth, min_content_length, *args, **kwargs):
        super(GeneralSpider, self).__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.allowed_domains = [urlparse(start_url).netloc]
        self.max_depth = max_depth
        self.min_content_length = min_content_length

    def parse(self, response):
        if self.is_valid_url(response.url):
            page_content = response.text
            clean_text = self.clean_html(page_content)

            if self.is_high_quality_content(clean_text):
                yield {'url': response.url, 'content': clean_text}

        for next_page in response.css('a::attr(href)').getall():
            next_page = response.urljoin(next_page)
            if self.is_valid_url(next_page) and self.is_within_depth(next_page):
                yield response.follow(next_page, self.parse)

    def clean_html(self, raw_html):
        soup = BeautifulSoup(raw_html, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()

        content = []
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'code']):
            text = element.get_text(separator=" ", strip=True)
            if len(text) > 30:
                content.append(text)

        return " ".join(content)

    def is_valid_url(self, url):
        exclude_patterns = ['contact', 'about',
                            'privacy', 'terms', 'login', 'signup']
        return not any(pattern in url for pattern in exclude_patterns) and urlparse(url).netloc in self.allowed_domains

    def is_within_depth(self, url):
        return url.count('/') <= self.max_depth + 2

    def is_high_quality_content(self, text):
        return len(text) > self.min_content_length


def scrape_url(url, max_depth, min_content_length):
    url_hash = hashlib.md5(url.encode()).hexdigest()
    process = CrawlerProcess(settings={
        'FEED_FORMAT': 'json',
        'FEED_URI': f'{url_hash}.json'
    })
    process.crawl(GeneralSpider, start_url=url, max_depth=max_depth,
                  min_content_length=min_content_length)
    process.start()
    return url


def scrape_urls_parallel(urls, max_depth, min_content_length):
    with multiprocessing.Pool() as pool:
        scrape_func = partial(scrape_url, max_depth=max_depth,
                              min_content_length=min_content_length)
        results = pool.map(scrape_func, urls)
    return results

if __name__ == '__main__':
    urls = ['https://pypi.org/project/mnelab/0.5.5/']
    max_depth = 1
    min_content_length = 100
    results = scrape_urls_parallel(urls, max_depth, min_content_length)
    print(results)