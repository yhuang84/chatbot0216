"""
NCSU sitemap crawler for discovering pages.
"""

import time
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

from .models import SearchResult, ScrapingConfig


class NCSUSitemapCrawler:
    """Crawls NCSU sitemaps to find relevant pages."""
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        self.config = config or ScrapingConfig()
        self.session = self._create_session()
        self.discovered_urls: Set[str] = set()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with headers."""
        session = requests.Session()
        
        session.headers.update({
            'User-Agent': self.config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        return session
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """
        Search for relevant pages by crawling NCSU sitemaps and filtering by query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
        """
        max_results = max_results or self.config.max_pages_per_query
        
        print(f"🗺️ Searching NCSU sitemaps for: '{query}'")
        
        # Discover URLs from sitemaps
        urls = self._discover_urls_from_sitemaps()
        
        if not urls:
            print("⚠️ No URLs found in sitemaps")
            return []
        
        print(f"📄 Found {len(urls)} URLs in sitemaps")
        
        # Filter URLs by relevance to query
        relevant_results = self._filter_urls_by_query(urls, query, max_results)
        
        print(f"🎯 Found {len(relevant_results)} relevant results")
        return relevant_results
    
    def _discover_urls_from_sitemaps(self) -> Set[str]:
        """Discover URLs from NCSU sitemaps."""
        urls = set()
        
        # Common sitemap locations
        sitemap_urls = [
            "https://www.ncsu.edu/sitemap.xml",
            "https://www.ncsu.edu/sitemap_index.xml",
            "https://www.ncsu.edu/robots.txt",  # Check robots.txt for sitemap references
        ]
        
        for sitemap_url in sitemap_urls:
            try:
                print(f"🔍 Checking sitemap: {sitemap_url}")
                
                if sitemap_url.endswith('robots.txt'):
                    # Parse robots.txt for sitemap references
                    sitemap_refs = self._parse_robots_txt(sitemap_url)
                    for ref in sitemap_refs:
                        urls.update(self._parse_sitemap(ref))
                else:
                    # Parse XML sitemap directly
                    urls.update(self._parse_sitemap(sitemap_url))
                    
            except Exception as e:
                print(f"⚠️ Error processing {sitemap_url}: {str(e)}")
                continue
        
        return urls
    
    def _parse_robots_txt(self, robots_url: str) -> List[str]:
        """Parse robots.txt file for sitemap references."""
        sitemaps = []
        
        try:
            response = self.session.get(robots_url, timeout=self.config.timeout)
            response.raise_for_status()
            
            for line in response.text.split('\n'):
                line = line.strip()
                if line.lower().startswith('sitemap:'):
                    sitemap_url = line.split(':', 1)[1].strip()
                    sitemaps.append(sitemap_url)
                    
        except Exception as e:
            print(f"Error parsing robots.txt: {str(e)}")
        
        return sitemaps
    
    def _parse_sitemap(self, sitemap_url: str) -> Set[str]:
        """Parse XML sitemap and extract URLs."""
        urls = set()
        
        try:
            response = self.session.get(sitemap_url, timeout=self.config.timeout)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            # Handle sitemap index (contains references to other sitemaps)
            if 'sitemapindex' in root.tag:
                for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                    loc_elem = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc_elem is not None:
                        # Recursively parse referenced sitemaps
                        urls.update(self._parse_sitemap(loc_elem.text))
            
            # Handle regular sitemap (contains URLs)
            else:
                for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                    loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc_elem is not None:
                        urls.add(loc_elem.text)
                        
        except ET.ParseError as e:
            print(f"XML parsing error for {sitemap_url}: {str(e)}")
        except Exception as e:
            print(f"Error parsing sitemap {sitemap_url}: {str(e)}")
        
        return urls
    
    def _filter_urls_by_query(self, urls: Set[str], query: str, max_results: int) -> List[SearchResult]:
        """Filter URLs by relevance to the search query."""
        query_terms = query.lower().split()
        results = []
        
        for url in urls:
            try:
                # Calculate relevance based on URL and fetch page title
                relevance_score = self._calculate_url_relevance(url, query_terms)
                
                if relevance_score > 0:  # Only include somewhat relevant URLs
                    # Try to get page title
                    title = self._get_page_title(url)
                    
                    # Adjust relevance based on title
                    if title:
                        title_relevance = self._calculate_text_relevance(title, query_terms)
                        relevance_score = max(relevance_score, title_relevance)
                    
                    result = SearchResult(
                        title=title or self._extract_title_from_url(url),
                        url=url,
                        snippet="",  # No snippet available from sitemap
                        relevance_score=relevance_score
                    )
                    
                    results.append(result)
                    
            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")
                continue
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:max_results]
    
    def _calculate_url_relevance(self, url: str, query_terms: List[str]) -> float:
        """Calculate relevance score based on URL path."""
        url_lower = url.lower()
        score = 0.0
        
        for term in query_terms:
            if term in url_lower:
                score += 1.0
        
        # Bonus for certain URL patterns
        if any(pattern in url_lower for pattern in ['/academics/', '/programs/', '/admissions/', '/research/']):
            score += 0.5
        
        return score
    
    def _calculate_text_relevance(self, text: str, query_terms: List[str]) -> float:
        """Calculate relevance score based on text content."""
        text_lower = text.lower()
        score = 0.0
        
        for term in query_terms:
            if term in text_lower:
                score += 2.0  # Higher weight for title matches
        
        return score
    
    def _get_page_title(self, url: str) -> Optional[str]:
        """Get page title by making a HEAD request and optionally a GET request."""
        try:
            # First try HEAD request to be efficient
            response = self.session.head(url, timeout=5)
            
            # If HEAD doesn't work or we need the title, make a GET request
            if response.status_code != 200:
                response = self.session.get(url, timeout=10)
            
            if response.status_code == 200 and 'text/html' in response.headers.get('content-type', ''):
                # Get just the beginning of the page to extract title
                content = response.content[:2048]  # First 2KB should contain title
                soup = BeautifulSoup(content, 'html.parser')
                title_elem = soup.find('title')
                if title_elem:
                    return title_elem.get_text(strip=True)
            
        except Exception:
            pass  # Ignore errors, we'll use URL-based title
        
        return None
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract a readable title from URL path."""
        try:
            parsed = urlparse(url)
            path = parsed.path.strip('/')
            
            if not path:
                return "NCSU Home"
            
            # Take the last part of the path and clean it up
            parts = path.split('/')
            title_part = parts[-1] if parts[-1] else parts[-2] if len(parts) > 1 else path
            
            # Clean up the title
            title = title_part.replace('-', ' ').replace('_', ' ')
            title = ' '.join(word.capitalize() for word in title.split())
            
            return title
            
        except Exception:
            return "NCSU Page"
