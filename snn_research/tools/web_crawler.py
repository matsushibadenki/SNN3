# snn_research/tools/web_crawler.py
# Title: Web Crawler Tool
# Description: 指定されたURLからWebページのコンテンツを取得し、HTMLからテキストデータを抽出するツール。

import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, urlparse
from typing import Set, List, Optional
import time
import os
import json

class WebCrawler:
    """
    Webを巡回し、テキストコンテンツを収集するシンプルなクローラー。
    """
    def __init__(self, output_dir: str = "workspace/web_data"):
        self.visited_urls: Set[str] = set()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _is_valid_url(self, url: str, base_domain: str) -> bool:
        """巡回対象として適切なURLか判断する。"""
        parsed_url = urlparse(url)
        return (
            parsed_url.scheme in ["http", "https"] and
            parsed_url.netloc == base_domain and
            url not in self.visited_urls
        )

    def _extract_text_from_html(self, html_content: str) -> str:
        """BeautifulSoupを使ってHTMLから主要なテキストを抽出する。"""
        soup = BeautifulSoup(html_content, 'html.parser')
        # ヘッダー、フッター、ナビゲーションなどの不要な部分を削除
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            element.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        return text

    def crawl(self, start_url: str, max_pages: int = 5) -> str:
        """
        指定されたURLからクロールを開始し、収集したテキストデータをファイルに保存する。

        Returns:
            str: 保存されたjsonlファイルのパス。
        """
        urls_to_visit: List[str] = [start_url]
        base_domain = urlparse(start_url).netloc
        
        output_filename = f"crawled_data_{int(time.time())}.jsonl"
        output_filepath = os.path.join(self.output_dir, output_filename)

        page_count = 0
        with open(output_filepath, 'w', encoding='utf-8') as f:
            while urls_to_visit and page_count < max_pages:
                current_url = urls_to_visit.pop(0)
                if not self._is_valid_url(current_url, base_domain):
                    continue

                try:
                    print(f"📄 クロール中: {current_url}")
                    response = requests.get(current_url, timeout=10)
                    response.raise_for_status()
                    self.visited_urls.add(current_url)
                    page_count += 1

                    text_content = self._extract_text_from_html(response.text)
                    
                    if text_content:
                        # データをjsonl形式で保存
                        record = {"text": text_content, "source_url": current_url}
                        f.write(json.dumps(record) + "\n")

                    # ページ内の新しいリンクを探す
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
                        # mypyがlink['href']でエラーを出すため、より安全な方法で属性を取得
                        if isinstance(link, Tag):
                            href = link.get('href')
                            if isinstance(href, str):
                                absolute_link = urljoin(current_url, href)
                                if self._is_valid_url(absolute_link, base_domain):
                                    urls_to_visit.append(absolute_link)
                        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
                    
                    time.sleep(1)  # サーバーへの負荷を軽減

                except requests.RequestException as e:
                    print(f"❌ クロールエラー: {current_url} ({e})")

        print(f"✅ クロール完了。{page_count}ページのデータを '{output_filepath}' に保存しました。")
        return output_filepath

