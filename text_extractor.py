"""
Scrapes Wikipedia for a given list of topics.

TODO: Add function that checks if a source has been scraped already
TODO: Figure out if this should run in parallel using Pool
"""
import wikipediaapi
import pandas as pd
import concurrent.futures
from tqdm import tqdm


class Sources:
    def __init__(self, verbose=True, pages=[]):
        self.pages = pages
        self.sources = pd.DataFrame(columns=['page', 'text', 'link', 'categories', 'topic'])
        self.verbose = verbose

    # TODO: Add function here the checks if a source (or topic) has been scraped already
    # Something like "if current_topic in self.sources['']: return True"
    def alreadyScraped(self, page_name):
        if self.sources['page'].str.contains(page_name).any():
            print("Page '{}' has been scraped already!".format(page_name))
            return True
        else:
            return False

    def extract(self):
        def wikiLink(link):
            try:
                page = wiki_api.page(link)
                if page.exists():
                    return {'page': link, 'text': page.text, 'link': page.fullurl,
                            'categories': list(page.categories.keys())}
            except:
                return None

        for page_name in self.pages:
            if not self.alreadyScraped(page_name):
                wiki_api = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
                page_name = wiki_api.page(page_name)
                if not page_name.exists():
                    print('Page {} does not exist.'.format(page_name))
                    return
                
                page_links = list(page_name.links.keys())
                progress = tqdm(desc='Links Scraped', unit='', total=len(page_links)) if self.verbose else None
                current_source = [{'page': page_name, 'text': page_name.text, 'link': page_name.fullurl,
                            'categories': list(page_name.categories.keys())}]
                
                # Parallelize the scraping, to speed it up (?)
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_link = {executor.submit(wikiLink, link): link for link in page_links}
                    for future in concurrent.futures.as_completed(future_link):
                        data = future.result()
                        self.sources.append(data) if data else None
                        progress.update(1) if self.verbose else None     
                progress.close() if self.verbose else None
                
                namespaces = ('Wikipedia', 'Special', 'Talk', 'LyricWiki', 'File', 'MediaWiki', 'Template', 'Help', 'User', \
                    'Category talk', 'Portal talk')
                current_source = self.sources[(len(self.sources['text']) > 20) & ~(self.sources['page'].str.startswith(namespaces, na=True))]
                current_source['categories'] = self.sources.categories.apply(lambda x: [y[9:] for y in x])
                current_source['topic'] = page_name
                print('Wikipedia pages scraped so far:', len(self.sources))

            
        return self.sources
