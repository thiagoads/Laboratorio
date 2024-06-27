# Scraping c/ scrapy

## Ambiente

- debian12 (bookworm)
- miniconda
- vscode (IDE) + jupyter notebook (extensão)

## Configuração

1) criar ambiente conda
```
conda env create -f environment.yml
conda activate scraping
```

## Execução

1) criando novos projetos
   ```
   scrapy startproject foo
   ```

2) executando uma spider
   ```
   scrapy crawl bar
   ```

3) salvando resultdo de uma spider
   ```
   scrapy crawl bar -O bar.json  # overwrite
   scrapy crawl bar -o bar.jsonl # append...
   ```
   

Utilitário p/ processar arquivos json: https://jqlang.github.io/jq/

