import bs4
import requests
import string
import multiprocessing as mp
from pycorenlp import StanfordCoreNLP
import numpy as np
import time

main_url ='http://lyrics.wikia.com'
stopwds = ['i', 'you', 'the', 'a', 'if', 'or', 'you', 'they', 'are', 'is', 'of',
 'that', 'which', 'when', 'with', 'we', 'what', 'to']

################ BASIC UTILS ###################################

def clean_txt(txt, stopwords):
    chars = [x for x in list(string.punctuation) + ['«', '»', '©','■', '€','°']]
    txt = txt.replace('\n', ' ')
    txt = txt.replace('Lyrics', ' ')
    txt = txt.replace('  ', ' ')
    txt = txt.strip('       ')
    txt = txt.replace('Verse', ' ')
    txt = txt.lower()
    for char in chars:
        if char in txt:
            txt = txt.replace(char, '')
    words = txt.split()
    final_txt = ' '.join([w for w in words if w not in stopwords])
    return final_txt

def is_text_good(title):
    bool_ = (len(title) > 4) and any(letter in title for letter in ['a', 'e', 'i',\
     'o', 'u']) and \
    all(letter not in title for letter in ['à', 'è', 'í', 'ò', 'ù', 'á', 'é',\
                    'ï', 'ó', 'ú', 'ô', 'ö', 'ü', 'æ', 'ß', 'ę', 'ć', '君', '왜'])
    return bool_

############################### 1ST ROUND OF PARALLELIZATION: SCRAPING ########

def get_all_year_links(min_year, base_url = 'http://lyrics.wikia.com/wiki/Category:Albums_by_Release_Year'):

    year_list, main = [], 'http://lyrics.wikia.com/wiki/'
    y_pages, m = {}, 'http://lyrics.wikia.com'

    r = requests.get( base_url, allow_redirects=True, timeout=20)
    soup = bs4.BeautifulSoup(r.text, 'html.parser')
    cat_divs= soup.find_all('div', attrs={'class': "CategoryTreeSection"})
    for div in cat_divs:
        lyrics_link = div.find('a', attrs={'class': \
            'CategoryTreeLabel CategoryTreeLabelNs14 CategoryTreeLabelCategory'}, href=True)
        year = lyrics_link.text.split()[-1]
        http = lyrics_link['href'].split('/')[-1]
        link = http.split(':')[0]+':'+http.split(':')[1]
        if year.isdigit():
            if int(year) > min_year:
                year_list.append((year, main+link))
    return year_list

def get_lyrics(y_list, lyr_list, queue):
    y_pages, alb_pages, m = {}, {}, 'http://lyrics.wikia.com'
    for y in y_list:
        y_pages[y[0]] = []
        r = requests.get(y[1] , allow_redirects=True, timeout=20)
        soup = bs4.BeautifulSoup(r.text, 'html.parser')
        adivs = soup.find_all('div', attrs={'dir': "ltr"})
        for d in adivs:
            aes = d.find_all('a', href=True)
            for a in aes:
                if all(x in a['href'] for x in ['/wiki/', '(', ')', '_', ':']):
                    y_pages[y[0]].append(m+a['href'])
            
    for k, v in y_pages.items():
        alb_pages[k] = {}
        alb_pages[k]['song_links'] = []
        for link in v:
            r = requests.get(link , allow_redirects=True, timeout=20)
            soup = bs4.BeautifulSoup(r.text, 'html.parser')
            ols = soup.find_all('ol')
            for o in ols:
                if o:
                    try:
                        a = o.find('b').find('a', href=True)
                        alb_pages[k]['song_links'].append(a['href'])
                    except:
                        continue

    for k, v in alb_pages.items():
        for link in v['song_links']:
            r = requests.get(m + link , allow_redirects=True, timeout=20)
            soup = bs4.BeautifulSoup(r.text, 'html.parser')
            lyr = soup.find('div', attrs={'class': "lyricbox"})
            if lyr:
                if len(lyr.text) >= 50 and is_text_good(lyr.text):
                    tup = (k, clean_txt(lyr.text, stopwds))
                    lyr_list.append(tup)

def enqueue(lyr_list, queue):
    for tup in lyr_list:
        queue.put(tup)

############################### 2ND ROUND : CORE NLP ########

def connect_nlp(url):
    return StanfordCoreNLP(url)

def calculate_sentiment(sent_distrib):
    idx = 0
    for pond in [(i+1)*p for i, p in enumerate(sent_distrib)]:
        idx += pond 
    return idx

def process_text(nlp, queue, sent_dict_lst):
    sent_dict ={}
    while not queue.empty():
        song = queue.get()
        if len(song[1]) >= 400:
            lyrics = song[1][:400]
        else:
            lyrics = song[1]
        try:
            nlp_res = nlp.annotate(lyrics, properties={'annotators': 'sentiment',
                                    'outputFormat': 'json', 'timeout': 10000,})

            sent_distrib = nlp_res["sentences"][0]['sentimentDistribution']
            sentim = calculate_sentiment(sent_distrib)
            if song[0] not in sent_dict:
                sent_dict[song[0]] = {'all_sents': [sentim]}
            else:
                sent_dict[song[0]]['all_sents'].append(sentim)
        except Exception as e:
            pass
        
    for k, v in sent_dict.items():
        v['avg_sent'] = sum(v['all_sents'])/len(v['all_sents'])
    sent_dict_lst.append(sent_dict)

###############################################################################

def process_lyrics(min_year, lyr_list, queue, nlp, sent_dict_lst):
    '''
    Wrapper to get and process all lyrics in parallel
    '''
    y_links = get_all_year_links(min_year)

    ys1 = [yl for yl in y_links if int(yl[0])%2 == 0]
    ys2 = [yl for yl in y_links if (int(yl[0])%5 == 0 and int(yl[0])%2 != 0)]
    ys3 = [yl for yl in y_links if (int(yl[0])%2 != 0 and int(yl[0])%5 != 0)]

    prod1 = mp.Process(target=get_lyrics, args=(ys1, lyr_list, queue))
    prod2 = mp.Process(target=get_lyrics, args=(ys2, lyr_list, queue))
    prod3 = mp.Process(target=get_lyrics, args=(ys3, lyr_list, queue))

    prod1.start()
    prod2.start()
    prod3.start()
    prod1.join()
    prod2.join()
    prod3.join()

    enqueue(lyr_list, queue)

    cons = mp.Process(target=process_text, args=(nlp, queue, sent_dict_lst))
    cons.start()
    cons.join()

if __name__ == "__main__":

    ec2_url='http://ec2-18-188-144-39.us-east-2.compute.amazonaws.com:9000'
    nlp = connect_nlp(ec2_url)
    manager = mp.Manager()
    queue = mp.Queue()
    lyr_lst = manager.list()
    sent_dict_lst = manager.list()

    process_lyrics(min_year, lyr_list, queue, nlp, sent_dict_lst)
                    