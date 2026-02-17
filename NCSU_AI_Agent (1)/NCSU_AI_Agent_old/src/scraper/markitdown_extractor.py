"""
MarkItDown Content Extractor - Clean content extraction using markitdown
========================================================================

Uses the markitdown package for reliable content extraction from web pages.
"""

import os
import sys
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any
from urllib.parse import urlparse

try:
    from markitdown import MarkItDown
except ImportError:
    print("❌ markitdown package not installed. Run: pip install markitdown")
    sys.exit(1)

# No longer needed - MarkItDown handles HTTP requests internally
from pydantic import HttpUrl

try:
    from .models import ScrapedPage
    from ..utils.logger import setup_logger
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.scraper.models import ScrapedPage
    from src.utils.logger import setup_logger


class MarkItDownExtractor:
    """Content extractor using MarkItDown for clean text extraction"""
    
    def __init__(self, user_agent: str = "NCSU Research Assistant Bot 1.0"):
        self.user_agent = user_agent
        self.logger = setup_logger("markitdown_extractor")
        self.markitdown = MarkItDown()
        
        self.logger.info("🚀 MarkItDown extractor initialized (direct URL conversion)")
    

    
    def extract_page(self, url: str, timeout: int = 30) -> Optional[ScrapedPage]:
        """Extract content from a URL using MarkItDown"""
        try:
            self.logger.info(f"🔄 Extracting content from: {url}")
            
            # Use MarkItDown to directly convert the URL
            result = self.markitdown.convert(url)
            
            if not result or not result.text_content:
                self.logger.warning(f"⚠️ No content extracted from: {url}")
                return None
            
            # Extract title from the content (first line or from metadata)
            content_lines = result.text_content.strip().split('\n')
            title = None
            
            # Try to get title from the first line if it looks like a title
            if content_lines:
                first_line = content_lines[0].strip()
                if first_line and len(first_line) < 200 and not first_line.startswith('http'):
                    title = first_line
                    # Remove title from content to avoid duplication
                    content = '\n'.join(content_lines[1:]).strip()
                else:
                    content = result.text_content.strip()
            else:
                content = result.text_content.strip()
            
            # If no title found, try to extract from URL
            if not title:
                parsed_url = urlparse(url)
                title = parsed_url.path.split('/')[-1] or parsed_url.netloc
                title = title.replace('-', ' ').replace('_', ' ').title()
            
            # Create metadata
            metadata = {
                'source': 'markitdown',
                'url': url,
                'content_length': len(content),
                'extraction_time': datetime.now().isoformat(),
                'extraction_method': 'direct_url_conversion'
            }
            
            # Calculate content hash
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # Create ScrapedPage object
            scraped_page = ScrapedPage(
                url=HttpUrl(url),
                title=title,
                content=content,
                extraction_success=True,
                metadata=metadata,
                content_hash=content_hash,
                word_count=len(content.split())
            )
            
            word_count = len(content.split())
            self.logger.info(f"✅ Successfully extracted: {title} ({word_count} words)")
            
            return scraped_page
            
        except Exception as e:
            self.logger.error(f"❌ MarkItDown extraction error for {url}: {e}")
            return self._create_failed_page(url, f"MarkItDown error: {e}")
    
    def _create_failed_page(self, url: str, error_msg: str) -> ScrapedPage:
        """Create a ScrapedPage object for failed extractions"""
        return ScrapedPage(
            url=HttpUrl(url),
            title="Failed to extract",
            content="",
            extraction_success=False,
            metadata={'error': error_msg, 'source': 'markitdown'},
            word_count=0
        )
    
    def extract_multiple_pages(self, urls: list, timeout: int = 30) -> list[ScrapedPage]:
        """Extract content from multiple URLs"""
        results = []
        
        for i, url in enumerate(urls, 1):
            self.logger.info(f"🔄 Processing {i}/{len(urls)}: {url}")
            
            scraped_page = self.extract_page(url, timeout)
            if scraped_page:
                results.append(scraped_page)
        
        successful = len([r for r in results if r.extraction_success])
        self.logger.info(f"📊 Extraction complete: {successful}/{len(results)} pages successful")
        
        return results


def main():
    """Test the MarkItDown extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MarkItDown Content Extractor")
    parser.add_argument("--url", "-u", required=True, help="URL to extract content from")
    parser.add_argument("--output", "-o", help="Output file to save content")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = MarkItDownExtractor()
    
    # Extract content
    print(f"🔍 Extracting content from: {args.url}")
    scraped_page = extractor.extract_page(args.url)
    
    if scraped_page and scraped_page.extraction_success:
        print(f"\n✅ SUCCESS!")
        print(f"📋 Title: {scraped_page.title}")
        print(f"📊 Word Count: {scraped_page.word_count}")
        print(f"🔗 URL: {scraped_page.url}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"Title: {scraped_page.title}\n")
                f.write(f"URL: {scraped_page.url}\n")
                f.write(f"Word Count: {scraped_page.word_count}\n")
                f.write(f"Extracted: {scraped_page.scraped_at}\n\n")
                f.write("CONTENT:\n")
                f.write("=" * 50 + "\n")
                f.write(scraped_page.content)
            print(f"💾 Content saved to: {args.output}")
        else:
            print(f"\n📄 CONTENT PREVIEW:")
            print("=" * 50)
            print(scraped_page.content[:500] + "..." if len(scraped_page.content) > 500 else scraped_page.content)
    else:
        print(f"❌ FAILED to extract content")
        if scraped_page and scraped_page.metadata.get('error'):
            print(f"Error: {scraped_page.metadata['error']}")


if __name__ == "__main__":
    main()
