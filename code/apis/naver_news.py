"""
naver news
"""
import os
import urllib.request
import urllib.parse
import json
import time

class NaverNewsAPI:
    """네이버 뉴스 검색 API 클라이언트"""
    
    def __init__(self):
        self.client_id = "EPCR5d1whNbimUA9ICpK"
        self.client_secret = "oAjgY5t6Pi"
        self.base_url = "https://openapi.naver.com/v1/search/news.json"
        self.request_headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }

    def search(self, keyword: str, display: int = 50, sort: str = 'date') -> dict:
        """
        네이버 뉴스 검색 실행
        Args:
            keyword (str): 검색 키워드
            display (int): 결과 개수 (기본 50, 최대 100)
            sort (str): 정렬 기준 (date/sim)
        Returns:
            dict: JSON 형식의 검색 결과
        """
        enc_text = urllib.parse.quote(keyword)
        url = f"{self.base_url}?query={enc_text}&display={display}&sort={sort}"
        
        request = urllib.request.Request(url, headers=self.request_headers)
        
        try:
            with urllib.request.urlopen(request) as response:
                if response.getcode() == 200:
                    print("\n✅ 네이버 API 요청 성공")
                    return json.loads(response.read().decode('utf-8'))
                print(f"❌ HTTP 에러 코드: {response.getcode()}")
                return {}
        except Exception as e:
            print(f"❌ 요청 실패: {e}")
            return {}

class NewsProcessor:
    """뉴스 데이터 처리 유틸리티 클래스"""
    
    @staticmethod
    def parse_news_items(result: dict) -> list:
        """API 응답에서 뉴스 아이템 추출"""
        return result.get("items", [])
    
    @staticmethod
    def validate_news_item(item: dict) -> bool:
        """뉴스 아이템 유효성 검증"""
        required_fields = ['title', 'description', 'link', 'pubDate']
        return all(field in item for field in required_fields)

class NewsErrorHandler:
    """에러 처리 전용 클래스"""
    
    @staticmethod
    def handle_api_error(error: Exception) -> dict:
        """API 통신 에러 처리"""
        return {
            'error': True,
            'message': f"API 요청 실패: {str(error)}",
            'retryable': True
        }
    
    @staticmethod
    def handle_parsing_error(item: dict) -> dict:
        """데이터 파싱 에러 처리"""
        return {
            'error': True,
            'message': f"잘못된 형식의 뉴스 데이터: {item}",
            'retryable': False
        }

# API 인스턴스 생성 (싱글턴 패턴 적용)
news_api = NaverNewsAPI()
