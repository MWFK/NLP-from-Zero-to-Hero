### Objectives
# Scrap business websites links and sub-links textual data

### Libs
import pandas as pd
import csv
import urllib.parse
import pickle
import re
import nltk
from bs4 import BeautifulSoup
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.crawler import CrawlerProcess
import os
Proxy = "***"

### Data
df = pd.read_excel (r'***')
print(df.shape)
print(df.iloc[0,12])
df.head() 

df.dropna(subset=["Web Adresse"], inplace=True)
print(df.shape)
url_list = df['Web Adresse']
print(url_list[0:5])
url_list = ['http://'+url for url in url_list]
print(url_list[0:5])


### Data processing
def clean_html(html):
    soup = BeautifulSoup(html)
    text = soup.get_text()
    text = re.sub("\s\s+", " ", text)
    return text

### Scrapper
class ClientSpider(CrawlSpider):
    name = "quotes" # if i comment this the scrapper won't work
    
    def __init__(self,  *a, **kw):
        super(ClientSpider, self).__init__(*a, **kw)
        
    def start_requests(self):
        sites ={}
        urls = url_list#[0:500]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse, meta={'proxy': Proxy,'sites': sites})
        file_to_write = open("data.pickle", "wb")
        pickle.dump(sites, file_to_write)
            
    def parse(self, response):
        sites = response.meta['sites']
        title = response.xpath('//title/text()').extract_first()
        body  = response.xpath('//body').extract()
        
        print('\n#######################################\n#######################################')
        
        results={}
        results['url']=response.url
        print(response.url)
        results['title']=title
        print('#####################', title, '#####################\n')
        results['text']=clean_html(body[0])
        #print('##################### Text\n', clean_html(body[0]))
        
        sites[response.url]=[]
        sites[response.url].append(results)
        
        file_to_write = open("data.pickle", "wb")
        pickle.dump(sites, file_to_write)
        
        
        # get the urls from the scrapped data
        for href in response.xpath('//body//a/@href').extract():
            #print(href)
            newurl = urllib.parse.urljoin(response.url, href)
            #print(newurl)
            if (newurl.startswith(response.url)) and (not newurl.endswith("pdf")) and (not newurl.endswith("jpg")) and (not newurl.endswith("png")):
                #print(newurl)
                #yield scrapy.Request(url=href, callback=self.parse1, meta={'proxy': 'xxxxxxxxxxxxx','sites': sites, 'urlpos':response.url})
                yield response.follow(href, self.parse1, meta={'proxy': Proxy,'sites': sites, 'urlpos':response.url})
            
    # get the data of the other urls filtered from the scraaped data
    def parse1(self, response):
        sites  = response.meta['sites']
        #print(response.meta['sites'])
        urlpos = response.meta['urlpos']
        #print(response.meta['urlpos'])
        title  = response.xpath('//title/text()').extract_first()
        #print(response.xpath('//title/text()').extract_first())
        body   = response.xpath('//body').extract()

        results={}
        results['url']   = response.url
        results['title'] = title
        results['text']  = clean_html(body[0])
        
        sites[urlpos].append(results)
        
        file_to_write = open("allwebsites.pickle", "wb")
        pickle.dump(sites, file_to_write)

### Execute the scrapper
%%time
c = CrawlerProcess({'USER_AGENT': '***', 'LOG_LEVEL' : 'ERROR'})
c.crawl(ClientSpider)
c.start()
