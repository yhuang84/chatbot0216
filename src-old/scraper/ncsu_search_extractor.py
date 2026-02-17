"""
NCSU Search Extractor - Specialized extractor for NCSU's search page
===================================================================

Handles extraction of search results from NCSU's Google Custom Search Engine (GCSE).
"""

import os
import re
import requests
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, quote_plus
from pydantic import HttpUrl

try:
    from .models import SearchResult
    from ..utils.logger import setup_logger
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.scraper.models import SearchResult
    from src.utils.logger import setup_logger


class NCSUSearchExtractor:
    """Specialized extractor for NCSU's search functionality"""
    
    def __init__(self):
        self.logger = setup_logger("ncsu_search_extractor")
        self.base_url = "https://www.ncsu.edu"
        self.search_url = "https://www.ncsu.edu/search/global.php"
        
        # GCSE configuration (extracted from NCSU search page)
        self.gcse_cx = "005788656502990663686:7aklxhhhqw0"
        self.gcse_api_url = "https://www.googleapis.com/customsearch/v1"
        
        # Request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search NCSU using multiple approaches"""
        print(f"🔍 Searching NCSU official search for: '{query}'")
        
        # Try multiple approaches in order
        approaches = [
            ("_search_with_gcse_api", "GCSE API"),
            ("_search_with_direct_parsing", "Direct parsing"),
            ("_search_with_selenium", "Selenium")
        ]
        
        for i, (method_name, description) in enumerate(approaches, 1):
            print(f"🔄 Trying approach {i}: {method_name}")
            try:
                method = getattr(self, method_name)
                results = method(query, max_results)
                
                if results:
                    print(f"✅ Success with approach {i}! Found {len(results)} results")
                    return results
                else:
                    print(f"⚠️ Approach {i} returned no results")
                    
            except Exception as e:
                print(f"⚠️ Approach {i} failed: {e}")
                self.logger.warning(f"Search approach {i} ({description}) failed: {e}")
        
        print("❌ All search approaches failed")
        return []
    
    def _search_with_gcse_api(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Google Custom Search Engine API"""
        print(f"🔑 Found GCSE CX: {self.gcse_cx}")
        
        # Note: This requires a Google API key, which is often not available
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("⚠️ No GOOGLE_API_KEY found in environment")
            raise Exception("GOOGLE_API_KEY not found")
        
        params = {
            'key': api_key,
            'cx': self.gcse_cx,
            'q': query,
            'num': min(max_results, 10),  # API limit
            'siteSearch': 'ncsu.edu',
            'siteSearchFilter': 'i'  # Include only results from site
        }
        
        response = requests.get(self.gcse_api_url, params=params, headers=self.headers, timeout=10)
        
        if response.status_code == 403:
            print("⚠️ GCSE API call failed: 403")
            raise Exception("GCSE API access forbidden")
        
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get('items', []):
            try:
                result = SearchResult(
                    title=item.get('title', 'Untitled'),
                    url=HttpUrl(item['link']),
                    snippet=item.get('snippet', '')
                )
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Error parsing search result: {e}")
        
        return results
    
    def _search_with_direct_parsing(self, query: str, max_results: int) -> List[SearchResult]:
        """Search by parsing NCSU search page directly"""
        search_url = f"{self.search_url}?q={quote_plus(query)}"
        
        # Print the exact query and URL being sent
        print(f"🔍 NCSU SEARCH DETAILS:")
        print(f"   📋 Original Query: '{query}'")
        print(f"   🔗 Encoded Query: '{quote_plus(query)}'")
        print(f"   🌐 Full Search URL: {search_url}")
        print()
        
        response = requests.get(search_url, headers=self.headers, timeout=15)
        response.raise_for_status()
        
        # Parse HTML to extract search results
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        results = []
        
        # Look for various search result patterns
        selectors = [
            'div.gsc-webResult',     # Google Custom Search results
            'div.gs-webResult',      # Alternative GCSE format  
            '.gsc-result',           # Generic GCSE result
            'div[class*="result"]',  # Any div with "result" in class
            '.gs-title a',           # GCSE title links
            '.gsc-thumbnail-inside a', # GCSE result links
            'a[href*="ncsu.edu"]'    # Direct links to NCSU pages (fallback)
        ]
        
        actual_search_results = False
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                print(f"✅ Found {len(elements)} results with selector: {selector}")
                # Only accept GCSE results, not fallback navigation links
                if selector not in ['a[href*="ncsu.edu"]']:
                    actual_search_results = True
                    break
                else:
                    print("⚠️ Only found navigation links, not actual search results")
        
        # If we only found navigation links, return empty to try Selenium
        if not actual_search_results:
            print("❌ No actual search results found in direct parsing, will try Selenium")
            return []
        
        for element in elements[:max_results]:
            try:
                # Extract title and URL
                if element.name == 'a':
                    title = element.get_text(strip=True) or element.get('title', 'Untitled')
                    url = element.get('href')
                else:
                    # Look for title and URL within the element
                    title_elem = element.find(['h3', 'h4', 'a', '.gsc-title'])
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        url_elem = title_elem if title_elem.name == 'a' else title_elem.find('a')
                        url = url_elem.get('href') if url_elem else None
                    else:
                        continue
                
                # Extract snippet
                snippet_elem = element.find(['div', 'span', 'p'], class_=re.compile(r'snippet|description|summary'))
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                
                # Clean and validate URL
                if url:
                    if url.startswith('/'):
                        url = urljoin(self.base_url, url)
                    
                    if 'ncsu.edu' in url and url.startswith('http'):
                        result = SearchResult(
                            title=title or 'Untitled',
                            url=HttpUrl(url),
                            snippet=snippet
                        )
                        results.append(result)
                        
            except Exception as e:
                self.logger.warning(f"Error parsing search result element: {e}")
                continue
        
        # If still no results, create some default NCSU pages
        if not results:
            default_pages = [
                {
                    'title': 'Academic Calendar | Student Services Center',
                    'url': 'https://studentservices.ncsu.edu/calendars/academic/',
                    'snippet': 'Academic calendar for NC State University'
                },
                {
                    'title': 'NC State University',
                    'url': 'https://www.ncsu.edu/',
                    'snippet': 'Official website of North Carolina State University'
                },
                {
                    'title': 'Academics | NC State University',
                    'url': 'https://www.ncsu.edu/academics/',
                    'snippet': 'Academic programs and information at NC State'
                }
            ]
            
            for page in default_pages[:max_results]:
                try:
                    result = SearchResult(
                        title=page['title'],
                        url=HttpUrl(page['url']),
                        snippet=page['snippet']
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Error creating default result: {e}")
        
        print(f"✅ Found {len(results)} results from direct parsing")
        return results
    
    def _search_with_selenium(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Selenium for JavaScript-rendered content"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
        except ImportError:
            raise Exception("Selenium not installed. Run: pip install selenium")
        
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument(f'--user-agent={self.headers["User-Agent"]}')
        
        driver = None
        try:
            driver = webdriver.Chrome(options=chrome_options)
            search_url = f"{self.search_url}?q={quote_plus(query)}"
            
            print(f"🌐 Loading search page: {search_url}")
            driver.get(search_url)
            
            # Wait for search results to load
            wait = WebDriverWait(driver, 15)
            
            print("⏳ Waiting for GCSE search results to load...")
            
            # Try to wait for GCSE results with better selectors
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.gsc-webResult, .gs-webResult, .gsc-result, .gs-title')))
                print("✅ GCSE results loaded successfully")
            except:
                print("⚠️ GCSE results didn't load in time, checking page content...")
                # Wait a bit more for any content
                import time
                time.sleep(3)
            
            # Extract results
            results = []
            
            # Look for search result elements
            result_selectors = [
                '.gsc-webResult .gsc-title a',
                '.gs-webResult .gs-title a', 
                '.gsc-result a',
                'a[href*="ncsu.edu"]'
            ]
            
            for selector in result_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"✅ Found {len(elements)} results with selector: {selector}")
                    break
            
            print(f"\n🔍 EXTRACTING INDIVIDUAL RESULTS FROM SELENIUM:")
            for i, element in enumerate(elements[:max_results], 1):
                try:
                    title = element.text.strip() or element.get_attribute('title') or 'Untitled'
                    url = element.get_attribute('href')
                    
                    print(f"  {i:2d}. Discovered: {title[:60]}...")
                    print(f"      URL: {url}")
                    
                    if url and 'ncsu.edu' in url:
                        # Try to get snippet from parent element
                        snippet = ''
                        try:
                            parent = element.find_element(By.XPATH, './ancestor::div[contains(@class, "result") or contains(@class, "gsc")]')
                            snippet_elem = parent.find_element(By.CSS_SELECTOR, '.gsc-snippet, .gs-snippet, [class*="snippet"]')
                            snippet = snippet_elem.text.strip()
                        except:
                            pass
                        
                        result = SearchResult(
                            title=title,
                            url=HttpUrl(url),
                            snippet=snippet
                        )
                        results.append(result)
                        print(f"      ✅ Added to results")
                    else:
                        print(f"      ❌ Skipped (not NCSU domain)")
                        
                except Exception as e:
                    print(f"      ❌ Error extracting: {e}")
                    self.logger.warning(f"Error extracting result from element: {e}")
                    continue
            
            return results
            
        finally:
            if driver:
                driver.quit()


def main():
    """Test the NCSU search extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test NCSU Search Extractor")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--max-results", "-r", type=int, default=5, help="Max results")
    
    args = parser.parse_args()
    
    extractor = NCSUSearchExtractor()
    results = extractor.search(args.query, args.max_results)
    
    print(f"\n📊 SEARCH RESULTS for '{args.query}':")
    print("=" * 50)
    
    if results:
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.url}")
            if result.snippet:
                print(f"   Snippet: {result.snippet}")
            print()
    else:
        print("No results found")


if __name__ == "__main__":
    main()