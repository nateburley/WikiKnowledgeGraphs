"""
Class that holds a list of sources. Created by scraping Wikipedia
"""
import wikipediaapi
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from source import Source

class SourceList:
    def __init__(self, source_list=[]):
        self.source_list = source_list
    
    # Function that checks if a source has already been added
    def checkSourceAdded(self, new_source):
        for existing_source in self.source_list:
            if new_source.page_title == existing_source.page_title:
                print("The page '{}' has already been added!".format(existing_source.page_title))
                return True
            else:
                return False
    
    # Function that checks if a page (not a Source) has been added
    def checkPageAdded(self, new_title):
        for existing_source in self.source_list:
            if new_title == existing_source.page_title:
                print("The page [title] '{}' has already been added!".format(existing_source.page_title))
                return True
            else:
                return False

    # Function that adds a new source
    def addSource(self, new_source):
        if not (self.checkSourceAdded(new_source) or self.checkPageAdded(new_source.page_title)):
            self.source_list.append(new_source)
    
    # Function that scrapes Wikipedia to build the source list
    #TODO: Add logic from other file to build sources
    def buildSourceList(self, titles, verbose=True):
        def wikiLink(link):
            try:
                page = wiki_api.page(link)
                if page.exists():
                    return {'page': link, 'text': page.text, 'link': page.fullurl,
                            'categories': list(page.categories.keys())}
            except:
                return None

        for current_title in titles:
            if not self.checkPageAdded(current_title):
                wiki_api = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
                page_name = wiki_api.page(current_title)
                if not page_name.exists():
                    print('Page {} does not exist.'.format(page_name))
                    return
                
                page_links = list(page_name.links.keys())
                print_description = "Links Scraped for page '{}'".format(current_title)
                progress = tqdm(desc=print_description, unit='', total=len(page_links)) if verbose else None
                current_source = Source(page_name.title, page_name.text, page_name.fullurl, list(page_name.categories.keys()), page_name)
                
                # Parallelize the scraping, to speed it up (?)
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_link = {executor.submit(wikiLink, link): link for link in page_links}
                    for future in concurrent.futures.as_completed(future_link):
                        data = future.result()
                        current_source.append(data) if data else None
                        progress.update(1) if verbose else None     
                progress.close() if verbose else None
                
                namespaces = ('Wikipedia', 'Special', 'Talk', 'LyricWiki', 'File', 'MediaWiki', 'Template', 'Help', 'User', \
                    'Category talk', 'Portal talk')
                current_source = current_source[(len(current_source['text']) > 20) & ~(current_source['page'].str.startswith(namespaces, na=True))]
                current_source['categories'] = current_source.categories.apply(lambda x: [y[9:] for y in x])
                current_source['topic'] = page_name
                print('Wikipedia pages scraped so far:', len(current_source))