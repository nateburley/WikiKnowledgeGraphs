"""
Class that contains a single "source", i.e. one Wikipedia page
"""
import pandas as pd

class Source:
    # Function that initializes a source object
    def __init__(self, title, text, link, categories, topic):
        self.page_title = title
        self.text = text
        self.link = link
        self.categories = categories
        self.topic = topic


    ## GETTERS
    # Function that returns the title
    def getTitle(self):
        return self.page_title


    # Function that returns the text
    def getText(self):
        return self.text


    # Function that returns the link
    def getLink(self):
        return self.link 


    # Function that returns the categories
    def getCategories(self):
        return self.categories


    # Function that returns the topic
    def getTopic(self):
        return self.topic 


    ## SETTERS
    # Function that sets the title
    def setTitle(self, new_title):
        self.page_title = new_title


    # Function that sets the text
    def setText(self, new_text):
        self.text = new_text


    # Function that sets the link
    def setLink(self, new_link):
        self.link = new_link 


    # Function that sets the categories
    def setCategories(self, new_category):
        self.categories = new_category


    # Function that sets the topic
    def setTopic(self, new_topic):
        self.topic = new_topic


    ## CREATE AND RETURN DATA FRAME, IF NEEDED
    def getDF(self):
        return pd.DataFrame({'title': self.page_title, 'text': self.text, 'link': self.link,\
            'categories': self.categories, 'topic': self.topic})



## HELPER FUNCTION THAT TURNS A DATA FRAME INTO A LIST OF SOURCES