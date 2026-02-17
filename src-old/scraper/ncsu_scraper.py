"""
NCSU Scraper - Main scraper class for NCSU website
=================================================

Handles searching and scraping content from NCSU website with multiple strategies.
"""

import os
import sys
from typing import List, Optional
from urllib.parse import urljoin, urlparse

try:
    from .models import SearchResult, ScrapedPage, ScrapingConfig
    from .ncsu_search_extractor import NCSUSearchExtractor
    from .markitdown_extractor import MarkItDownExtractor
    from .sitemap_crawler import NCSUSitemapCrawler
    from ..utils.logger import setup_logger
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.scraper.models import SearchResult, ScrapedPage, ScrapingConfig
    from src.scraper.ncsu_search_extractor import NCSUSearchExtractor
    from src.scraper.markitdown_extractor import MarkItDownExtractor
    from src.scraper.sitemap_crawler import NCSUSitemapCrawler
    from src.utils.logger import setup_logger


class NCSUScraper:
    """Main NCSU scraper with multiple search and extraction strategies"""
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        self.config = config or ScrapingConfig()
        self.logger = setup_logger("ncsu_scraper")
        
        # Initialize components
        self.search_extractor = NCSUSearchExtractor()
        self.content_extractor = MarkItDownExtractor()
        self.sitemap_crawler = NCSUSitemapCrawler()
        
        # Configuration
        self.target_domain = "ncsu.edu"
        self.search_url_template = "https://www.ncsu.edu/search/global.php?q={query}"
        
        self.logger.info("🚀 NCSU Scraper initialized")
        self.logger.info(f"📋 Config: selenium={self.config.selenium_enabled}, markitdown=True")
        self.logger.info(f"🎯 Target domain: {self.target_domain}")
        self.logger.info(f"🔍 Search URL: {self.search_url_template}")
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search NCSU website for the given query"""
        self.logger.info(f"🔍 Starting search for query: '{query}' (max_results={max_results})")
        
        results = []
        
        # Strategy 1: Official NCSU search
        self.logger.info("🎯 Using official NCSU search (ncsu.edu only)...")
        try:
            ncsu_results = self.search_extractor.search(query, max_results=max_results)
            if ncsu_results:
                results.extend(ncsu_results)
                self.logger.info(f"✅ NCSU search SUCCESS! Found {len(ncsu_results)} results from ncsu.edu")
            else:
                self.logger.warning("⚠️ NCSU search returned no results")
        except Exception as e:
            self.logger.error(f"❌ NCSU search failed: {e}")
        
        # Strategy 2: Sitemap crawling (if needed)
        if len(results) < max_results:
            self.logger.info("🗺️ Trying sitemap crawling as fallback...")
            try:
                sitemap_results = self.sitemap_crawler.search_sitemap(query, max_results=max_results-len(results))
                if sitemap_results:
                    results.extend(sitemap_results)
                    self.logger.info(f"✅ Sitemap crawling found {len(sitemap_results)} additional results")
            except Exception as e:
                self.logger.error(f"❌ Sitemap crawling failed: {e}")
        
        # Filter and deduplicate results
        print(f"\n🔍 DEDUPLICATION PROCESS:")
        print(f"   📊 Raw results found: {len(results)}")
        
        unique_results = []
        seen_urls = set()
        duplicates_removed = 0
        
        for i, result in enumerate(results, 1):
            url_str = str(result.url)
            if url_str not in seen_urls and self.target_domain in url_str:
                unique_results.append(result)
                seen_urls.add(url_str)
                print(f"   ✅ {len(unique_results):2d}. {result.title[:50]}...")
                print(f"       {url_str}")
            else:
                duplicates_removed += 1
                if url_str in seen_urls:
                    print(f"   ❌ {i:2d}. DUPLICATE: {result.title[:50]}...")
                else:
                    print(f"   ❌ {i:2d}. WRONG DOMAIN: {result.title[:50]}...")
        
        print(f"   📊 Duplicates removed: {duplicates_removed}")
        print(f"   📊 Unique results: {len(unique_results)}")
        
        final_results = unique_results[:max_results]
        print(f"   📊 Final results (limited by top_k={max_results}): {len(final_results)}")
        self.logger.info(f"🎯 Final results: {len(final_results)} unique NCSU pages")
        
        return final_results
    
    def scrape_pages(self, search_results: List[SearchResult]) -> List[ScrapedPage]:
        """Scrape content from search result pages"""
        self.logger.info(f"📄 Scraping {len(search_results)} pages...")
        
        scraped_pages = []
        
        for i, result in enumerate(search_results, 1):
            self.logger.info(f"🔄 Scraping page {i}/{len(search_results)}: {result.title}")
            
            try:
                # Extract content using enhanced extractor
                scraped_page = self.content_extractor.extract_page(str(result.url))
                
                if scraped_page and scraped_page.content and scraped_page.content.strip():
                    # Update with search result info
                    scraped_page.title = result.title or scraped_page.title or "Untitled"
                    scraped_page.extraction_success = True
                    scraped_pages.append(scraped_page)
                    word_count = len(scraped_page.content.split())
                    self.logger.info(f"✅ Successfully scraped: {scraped_page.title} ({word_count} words)")
                else:
                    self.logger.warning(f"⚠️ No content extracted from: {result.url}")
                    scraped_page = ScrapedPage(
                        url=result.url,
                        title=result.title or "Untitled",
                        content="",
                        extraction_success=False,
                        metadata={"error": "No content extracted"}
                    )
                    scraped_pages.append(scraped_page)
                    
            except Exception as e:
                self.logger.error(f"❌ Error scraping {result.url}: {e}")
                scraped_page = ScrapedPage(
                    url=result.url,
                    title=result.title or "Untitled",
                    content="",
                    extraction_success=False,
                    metadata={"error": str(e)}
                )
                scraped_pages.append(scraped_page)
        
        successful_pages = [p for p in scraped_pages if p.extraction_success]
        self.logger.info(f"📊 Scraping complete: {len(successful_pages)}/{len(scraped_pages)} pages successful")
        
        return scraped_pages
    
    def search_and_scrape(self, query: str, max_results: int = 10, max_pages: int = 5) -> List[ScrapedPage]:
        """Combined search and scrape operation"""
        self.logger.info(f"🎯 Starting combined search and scrape for: '{query}'")
        
        # Search for results
        search_results = self.search(query, max_results=max_results)
        
        if not search_results:
            self.logger.warning("❌ No search results found")
            return []
        
        # Limit pages to scrape
        pages_to_scrape = search_results[:max_pages]
        self.logger.info(f"📄 Will scrape top {len(pages_to_scrape)} pages")
        
        # Scrape pages
        scraped_pages = self.scrape_pages(pages_to_scrape)
        
        return scraped_pages


def main():
    """Test the NCSU scraper"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test NCSU Scraper")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--max-results", "-r", type=int, default=5, help="Max search results")
    parser.add_argument("--max-pages", "-p", type=int, default=3, help="Max pages to scrape")
    parser.add_argument("--selenium", action="store_true", help="Enable Selenium")
    
    args = parser.parse_args()
    
    # Create config
    config = ScrapingConfig(
        selenium_enabled=args.selenium,
        enhanced_extraction=True,
        timeout=30
    )
    
    # Initialize scraper
    scraper = NCSUScraper(config)
    
    # Test search and scrape
    print(f"🔍 Testing NCSU scraper with query: '{args.query}'")
    scraped_pages = scraper.search_and_scrape(
        query=args.query,
        max_results=args.max_results,
        max_pages=args.max_pages
    )
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"Found {len(scraped_pages)} pages")
    
    for i, page in enumerate(scraped_pages, 1):
        status = "✅" if page.extraction_success else "❌"
        word_count = len(page.content.split()) if page.content else 0
        print(f"{i}. {status} {page.title} ({word_count} words)")
        print(f"   URL: {page.url}")
        if not page.extraction_success and page.metadata.get('error'):
            print(f"   Error: {page.metadata['error']}")
        print()


if __name__ == "__main__":
    main()