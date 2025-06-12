"""
naver news api
"""

import urllib.request
import urllib.parse
import json
from dotenv import load_dotenv
import os

class NaverNewsAPI:
    def __init__(self):
        load_dotenv()  # .env 파일 자동 로딩
        client_id = os.getenv("NAVER_API_CLIENT_ID")
        client_secret = os.getenv("NAVER_API_CLIENT_SECRET")
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://openapi.naver.com/v1/search/news.json"

    def search(self, keyword: str) -> dict:
        display = 100
        enc_text = urllib.parse.quote(keyword)
        url = f"{self.base_url}?query={enc_text}&display={display}"

        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", self.client_id)
        request.add_header("X-Naver-Client-Secret", self.client_secret)

        try:
            with urllib.request.urlopen(request) as response:
                if response.getcode() == 200:
                    response_body = response.read().decode('utf-8')
                    print("✅ request Success")
                    return json.loads(response_body)
                else:
                    print(f"❌ Error Code: {response.getcode()}")
                    return {}
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return {}
