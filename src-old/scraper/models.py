"""
Data models for the scraper module.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, HttpUrl, Field


class ScrapedPage(BaseModel):
    """Model for a scraped web page."""
    
    url: HttpUrl
    title: Optional[str] = None
    content: str
    extraction_success: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    scraped_at: datetime = Field(default_factory=datetime.now)
    content_hash: Optional[str] = None
    word_count: int = 0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            HttpUrl: str
        }


class SearchResult(BaseModel):
    """Model for search results from NCSU search."""
    
    title: str
    url: HttpUrl
    snippet: Optional[str] = None
    relevance_score: Optional[float] = None
    
    class Config:
        json_encoders = {
            HttpUrl: str
        }


class ScrapingConfig(BaseModel):
    """Configuration for scraping operations."""
    
    target_domain: str = "ncsu.edu"
    search_url_template: str = "https://www.ncsu.edu/search/global.php?q={query}"
    user_agent: str = "NCSU Research Assistant Bot 1.0"
    delay: float = 1.0
    concurrent_requests: int = 8
    respect_robots_txt: bool = True
    max_pages_per_query: int = 50
    timeout: int = 30
    
    # Extraction options
    selenium_enabled: bool = False
    enhanced_extraction: bool = True
    
    # Content filtering
    min_content_length: int = 100
    max_content_length: int = 50000
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 2.0


class ScrapingStats(BaseModel):
    """Statistics for scraping operations."""
    
    total_pages_found: int = 0
    pages_scraped: int = 0
    pages_failed: int = 0
    pages_filtered: int = 0
    total_time: float = 0.0
    average_time_per_page: float = 0.0
    
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_pages_found == 0:
            return 0.0
        return (self.pages_scraped / self.total_pages_found) * 100
