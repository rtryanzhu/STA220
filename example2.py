# Rules of game
# 1.Qualified links are from main paragraph (Tagged with <a> or <ul>), links from tables and navigation section are not considered.
# 2.Game starts with the first qualified link from a random article 'https://en.wikipedia.org/wiki/Special:Random'
# 3.Game stops when no qualified link is available or it reaches max iteration number

# Performance
# Runtime was about 14min

import pandas as pd
import requests as req
import time
import re
import requests

from tqdm import tqdm 
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm.notebook import tqdm_notebook
from collections import Counter
from itertools import chain

starter_url = 'https://en.wikipedia.org/wiki/Special:Random'

# Scrape the page
def get_html(keyword):
    general_url = 'https://en.wikipedia.org/wiki/' # general part of wiki pages
    full_url = urljoin(general_url, keyword)
    page = req.get(full_url)
    soup = BeautifulSoup(page.content, "html.parser")

    return soup

# Get the principle part of the html
def get_main_body(keyword): # The main content of a wiki page
    soup = get_html(keyword)

    main_body = soup.find("div",{'id':"mw-content-text"})

    res = main_body

    return res

# Remove unneeded content
def get_clean_body(keyword): # Remove italicized, parenthesized content/tags and any other irrelevant content
    res = get_main_body(keyword)
    
    for i_tag in res.find_all('i'): # remove <i> tag, italicized content
        i_tag.extract()
    for span_tag in res.find_all('span'): # remove <i> tag, italicized content
        span_tag.extract() # Remove IPA content  
    for table_tag in res.find_all('table'): # remove <table> tag, proposed for 'U.S._City'
        table_tag.extract() 
    for div in res.find_all("div", {'class':'navbox authority-control'}): # Remove links in navigation section
        div.decompose()

    old_str = "" # Remove all parenthesized content but keep url within the parenthesis, adopted from lab mateiral
    new_str = res

    while old_str != new_str:
        # print(new_str)
        old_str = new_str
        new_str = re.sub("(?<!_)\([^()]*\)", "", str(old_str))   

    text = new_str
    
    ul = re.findall(r"<li>(.*?)</li>", str(text), flags=re.DOTALL) # links in the list
    body = re.findall(r"<p>(.*?)</p>", str(text), flags=re.DOTALL) # links in the main paragraph
    cleaned = body+ul # Return a list of desired tags

    return cleaned

# Core of the function, get the first qualified keyword
def get_new_keyword(keyword): # Return a new keyword if available, otherwise, return None, default return is None
    res_clean = get_clean_body(keyword)
    list = re.findall(r'href=[\'"]/wiki/([^\'" >]+)', str(res_clean))
    new_keyword = None
    for i in range(len(list)):
        if keyword is None:
            break
        elif keyword.lower() != list[i].lower(): # If keyword doesn't match with the first new keyword
            new_keyword = list[i]
            break
#     print (new_keyword)
    return new_keyword

# Get a list of words to start
def get_starter_list(n):
    starter_list = []
    for i in tqdm(range(n)):
        random_url = 'https://en.wikipedia.org/wiki/Special:Random'
        excluded_url = 'https://en.wikipedia.org/wiki/Wikipedia:'
        page = req.get(starter_url)
        while excluded_url in page.url:
            page = req.get(starter_url)
        soup = BeautifulSoup(page.content, "html.parser")
        main_body = soup.find("div",{"id":"mw-content-text"})
        res = main_body

        for i_tag in res.find_all('i'): # remove <i> tag, italicized content
            i_tag.extract()
        for span_tag in res.find_all('span'): # remove <i> tag, italicized content
            span_tag.extract() # Remove IPA content  
        for table_tag in res.find_all('table'): # remove <i> tag, italicized content
            table_tag.extract() # Remove IPA content  
        for div in res.find_all("div", {'class':'navbox authority-control'}): # Remove links in navigation section
            div.decompose()

        old_str = ""
        new_str = res

        while old_str != new_str:
            # print(new_str)
            old_str = new_str
            new_str = re.sub("(?<!_)\([^()]*\)", "", str(old_str))   

        text = new_str
    #     res = re.sub(r'href=[\'"]/wiki/Help([^\'" >]+)', "", str(res)) # remove parenthesized content

        ul = re.findall(r"<li>(.*?)</li>", str(text), flags=re.DOTALL)
        body = re.findall(r"<p>(.*?)</p>", str(text), flags=re.DOTALL) # Return a list of desired tags
        cleaned = body+ul
        while("" in cleaned):
            cleaned.remove("")

        res_clean = cleaned
        list = re.findall(r'href=[\'"]/wiki/([^\'" >]+)', str(res_clean))
        starter_list.append(list[0])
    return (starter_list)



# Main function
def play(starter_list, stop_word, max_iteration): 
    # How many games to play (number of words), where to stop ('Philosophy'), max search per word (1000)
    gp = len(starter_list)
    eta = max_iteration+1
    output_df = pd.DataFrame(columns = ['Starter','Steps','History','infinity_indicator'])
    
    for i in tqdm_notebook(range(gp)):
        counter = 1
#         starter = get_starter() # Get a random keyword 
        starter = starter_list[i]
        keyword = starter # Duplicated, to be replaced through iterations
        history = [] # Record search history

        inf_indicator = 0
        while counter < eta:
            if keyword not in history:
                if keyword != None:
                    if keyword != stop_word:
                        history.append(keyword)
                        keyword = get_new_keyword(keyword)
                    else:
                        history.append(stop_word)
                        break
            else:
                inf_indicator = 1
                break
            counter += 1
        # The counter stops before reaching 'Philosophy'
        
        output_df.loc[len(output_df.index)]=[starter,counter,history,inf_indicator] # record the result
        
    return output_df

# Analysis part
# Get the result of 200 games
output = play(slist,'Philosophy',1000) 

# Get the frequency table of articles visited
result = Counter([elem for elem in chain.from_iterable(output['History'].values)])
df = pd.DataFrame.from_dict(result, orient='index').reset_index()
df.columns = ['Visit','Counter']
df.sort_values('Counter',ascending=False)
