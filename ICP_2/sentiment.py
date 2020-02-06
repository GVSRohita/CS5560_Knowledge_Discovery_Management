import json

from stanfordcorenlp import StanfordCoreNLP

# Preset
host = 'http://localhost'
port = 9000
nlp = StanfordCoreNLP(host, port=port, timeout=30000)

# Importing requests library to send HTTP/1.1 requests
import requests

# Variable assigned to url of website from which data is to be scrapped
wikipage = requests.get('https://en.wikipedia.org/wiki/Text_mining').text

# Parsing data using BeautifulSoup function
from bs4 import BeautifulSoup
soup = BeautifulSoup(wikipage, 'html')

# Identifying text data accessed from web page
data = soup.get_text()

# Sentiment analysis
res = json.loads(nlp.annotate(data,
                              properties={
                                  'annotators': 'sentiment',
                                  'outputFormat': 'json'
                              }))

for s in res["sentences"]:
    print(s['sentimentDistribution'])
    print(s["sentiment"])

# Close Stanford Parser
nlp.close()