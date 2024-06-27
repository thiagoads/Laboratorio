from pathlib import Path

import scrapy

class QuotesSpiders(scrapy.Spider):
    name = "quotes"

    # use o atributo start_urls para criar requests
    start_urls = [
        "https://quotes.toscrape.com/page/1/",
        # "https://quotes.toscrape.com/page/2/" (uma ou uma lista de urls)
    ]

    # use o método start_requests para criar requests (alternativamente)
    # def start_requests(self):
    #     urls = [
    #         "https://quotes.toscrape.com/page/1/",
    #         "https://quotes.toscrape.com/page/2/"
    #     ]
    #     for url in urls:
    #         yield scrapy.Request(url = url, callback=self.parse)

    # use o método start_requests para receber atribuitos da linha de comando
    # scrapy crawl quotes -O data/quotes-humor.json -a tag=humor
    # def start_requests(self):
    #     url = "https://quotes.toscrape.com/"
    #     tag = getattr(self, "tag", None)
    #     if tag is not None:
    #         url = url + "tag/" + tag
    #     yield scrapy.Request(url, self.parse)

    # use o comando no terminal para auxiliar 
    # a identificação e leitura dos seletores 
    # scrapy shell 'https://quotes.toscrape.com'
    def parse(self, response):
        # acho que aqui pode ser feito algum cache p/ evitar chamadas já realizadas.
        # page = response.url.split("/")[-2]
        # filename = f"quotes-{page}.html"
        # Path(filename).write_bytes(response.body)
        # self.log(f"Saved file {filename}")
        for quote in response.css("div.quote"):
            yield {
                "text": quote.css("span.text::text").get(),
                "author": quote.css("small.author::text").get(),
                "tags": quote.css("div.tags a.tag::text").getall(),
            }

        # caso tenha uma página de next, existem várias maneiras de fazer
        next_page = response.css("li.next a::attr(href)").get()
        if next_page is not None:
            # next_page = response.urljoin(next_page)
            # yield scrapy.Request(next_page, callback=self.parse)
            yield response.follow(next_page, callback=self.parse)
            