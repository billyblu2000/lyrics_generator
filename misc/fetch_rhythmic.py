from bs4 import BeautifulSoup
import requests

res = requests.get('http://www.gushicimingju.com/gushi/cipaiming/page7', headers={
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,'
              'application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Cookie': 'Hm_lvt_98e7ed61a239ebe5fe1ce355caf04c03=1664898117; '
              '__gads=ID=d584d156368cb780-220407f9efd5002c:T=1664898117:RT=1664898117:S=ALNI_MZF5H18c3EH9lSUbZM3vpj'
              '-LtETYg; __gpi=UID=00000afec93528ae:T=1664898117:RT=1664898117:S=ALNI_MYcT-LXqwdg0Jr3D-ij5BzvT5ag9w; '
              'Hm_lpvt_98e7ed61a239ebe5fe1ce355caf04c03=1664899323',
    'Host': 'www.gushicimingju.com',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 '
                  'Safari/537.36 Edg/105.0.1343.53',
})
soup = BeautifulSoup(res.text, 'html.parser')
ele = soup.find(attrs={'class': 'main-data simple-people'})
for e in ele.find_all('li'):
    print(e.text)