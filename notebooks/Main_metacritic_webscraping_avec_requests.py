# Project GameCashPy- DataScientest
# Started 23rd February 2022
# List of websites that help producing this code:
# https://gist.github.com/tanpengshi/bb3b6e6fd83756a829b0aaf081660233
# https://medium.com/@raiyanquaium/how-to-web-scrape-using-beautiful-soup-in-python-without-running-into-http-error-403-554875e5abed
# https://stackoverflow.com/questions/13055208/httperror-http-error-403-forbidden
# https://www.kite.com/python/answers/how-to-get-the-status-code-of-a-website-using-urllib-in-python
# https://python-forum.io/thread-7221.html
# https://github.com/jhnwr/retry-requests/blob/master/yt-retry.py
# https://stackoverflow.com/questions/16511337/correct-way-to-try-except-using-python-requests-module
# https://findwork.dev/blog/advanced-usage-python-requests-timeouts-retries-hooks/#retry-on-failure

# Parameters for script user only.
# Choose pages for web scraping. Usually go by 20 pages.
# Also choose number of the first N movies in each page if this is wanted
No_page_start = 0
No_page_end = 19
N_start_movie_in_each_page = 0  # NB: 0 is the beginning
N_stop_movie_in_each_page = 100  # 100 is the number of movies per page in the main url, except for last page

# Importing packages
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Function and parameters for retrying requests to website
# see https://findwork.dev/blog/advanced-usage-python-requests-timeouts-retries-hooks/#retry-on-failure
retry_strategy = Retry(
    total=10,
    status_forcelist=[408, 429, 495, 496, 497, 500, 502, 503, 504],
    method_whitelist=["HEAD", "GET", "OPTIONS"],
    backoff_factor=2
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

# Creating the main url for primary pages ("mainpages")
url = 'https://www.metacritic.com/browse/games/score/userscore/all/all/filtered?sort=desc&view=condensed&page={}'

# Creating the list established from metacritic.com for the extracted information
Mylist_MC = []

# Time start
dateTimeObj = datetime.now()
timestampStart = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")

# Initiating the page number and total number of movies scraped
pageno = No_page_start
N_movies_scraped = 0
while True:
    if pageno == No_page_end + 1:
        break
    else:
        hdr = {
            'authority': 'ping.chartbeat.net',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="98", "Google Chrome";v="98"',
            'sec-ch-ua-mobile': '?0',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
            'sec-ch-ua-platform': '"macOS"',
            'accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'sec-fetch-site': 'cross-site',
            'sec-fetch-mode': 'no-cors',
            'sec-fetch-dest': 'image',
            'referer': 'https://www.google.com/',
            'accept-language': 'en-US,en;q=0.9,fr-FR;q=0.8,fr;q=0.7',
        }
        webpage = url.format(pageno)
        mainpage = http.get(webpage, headers=hdr, timeout=3).text
        mainsoup = BeautifulSoup(mainpage, 'html.parser')

        # Gathering other information from MetaCritic (MC)from subpages
        #tags = mainsoup.findAll('td', attrs={'class': 'details'})[
        #        N_start_movie_in_each_page:N_stop_movie_in_each_page]  # iterate through each movie on primary pages
        tags = mainsoup.findAll('td', attrs={'class': 'details'})
        No_movie_in_page = 1
        for tag in tags:
            try:
                platform_MC = tag.contents[3].contents[3].text.strip('\n ')
            except IndexError:
                platform_MC = None

            webpage2 = 'https://www.metacritic.com' + str(tag.find('a')['href'])  # link for secondary pages ("subpages")
            subpage = http.get(webpage2, headers=hdr, timeout=3).text
            soup = BeautifulSoup(subpage, 'html.parser')  # turn secondary page into soup
            try:
                game_title_MC = soup.findAll('a', class_='hover_none')[0].contents[1].text
            except IndexError:
                game_title_MC = None

            try:
                note_userscore_MC = soup.findAll('div', class_='metascore_w user large game positive')[0].contents[
                    0].text
            except IndexError:
                try:
                    note_userscore_MC = soup.findAll('div', class_='metascore_w user large game mixed')[0].contents[
                        0].text
                except IndexError:
                    try:
                        note_userscore_MC = soup.findAll('div', class_='metascore_w user large game negative')[0].contents[
                            0].text
                    except IndexError:
                        note_userscore_MC = None

            try:
                num_ratings_userscore_MC = soup.select(".feature_userscore a")[1].contents[0].strip(' Ratings')
            except IndexError:
                num_ratings_userscore_MC = None
            try:
                note_metascore_MC = soup.findAll('span', itemprop="ratingValue")[0].text
            except IndexError:
                note_metascore_MC = None
            try:
                num_ratings_metascore_MC = soup.select(".highlight_metascore .count a")[0].contents[1].text.strip('\n ')
            except IndexError:
                num_ratings_metascore_MC = None
            try:
                num_players = soup.findAll('li', class_='summary_detail product_players')[0].contents[3].text
            except IndexError:
                num_players = None
            try:
                rating_age = soup.findAll('li', class_='summary_detail product_rating')[0].contents[3].text
            except IndexError:
                rating_age = None
            try:
                releasedate_MC = soup.findAll('li', class_='summary_detail release_data')[0].contents[3].text
            except IndexError:
                releasedate_MC = None
            try:
                publisher_MC = \
                    soup.findAll('li', class_='summary_detail publisher')[0].contents[3].contents[1].contents[
                        0].text.strip('\n ')
            except IndexError:
                publisher_MC = None

            Dict_MC = {'game_title_MC': game_title_MC,
                       'platform_MC': platform_MC,
                       'releasedate_MC': releasedate_MC,
                       'publisher_MC': publisher_MC,
                       'note_userscore_MC': note_userscore_MC,
                       'num_ratings_userscore_MC': num_ratings_userscore_MC,
                       'note_metascore_MC': note_metascore_MC,
                       'num_ratings_metascore_MC': num_ratings_metascore_MC,
                       'num_players': num_players,
                       'rating_age': rating_age}

            Mylist_MC.append(Dict_MC)
            time.sleep(random.randint(2, 4))
            N_movies_scraped += 1
            print(
                f"No_page({pageno})--No_movie_in_the_page({No_movie_in_page}) // N_movies_scraped ({N_movies_scraped})")
            No_movie_in_page += 1
        print(f"No_page({pageno})")
        pageno += 1

# Time end
dateTimeObj = datetime.now()
timestampEnd = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")

#print(
#    f" Code Start:{timestampStart} // Code End:{timestampEnd} // No_page_start:{No_page_start} "
#    f"// No_page_end:{No_page_end} // N_start_movie_in_each_page:{N_start_movie_in_each_page}"
#    f"//N_stop_movie_in_each_page:{N_stop_movie_in_each_page}"
#    f"// Nmoviesscraped{N_movies_scraped}")

print(
    f" Code Start:{timestampStart} // Code End:{timestampEnd} // No_page_start:{No_page_start} "
    f"// No_page_end:{No_page_end} // Nmoviesscraped{N_movies_scraped}")

import pandas as pd

df = pd.DataFrame(Mylist_MC)
print(df)
df.to_csv(
    f'metacritic__Npagestart{No_page_start}__Npageend{No_page_end}__Nmovies{N_movies_scraped}.csv', index=False)
