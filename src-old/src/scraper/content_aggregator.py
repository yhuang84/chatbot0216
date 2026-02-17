"""
Content aggregator for preparing comprehensive content for LLM processing.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import ScrapedPage, SearchResult


class ContentAggregator:
    """Aggregates and prepares scraped content for LLM processing."""
    
    def __init__(self):
        self.aggregation_timestamp = datetime.now()
    
    def aggregate_for_llm(self, scraped_pages: List[ScrapedPage], query: str, max_total_tokens: int = 100000) -> Dict[str, Any]:
        """
        Aggregate scraped content into a comprehensive package for LLM processing.
        
        Args:
            scraped_pages: List of scraped pages with full content
            query: Original search query
            max_total_tokens: Maximum total tokens to include (rough estimate)
            
        Returns:
            Comprehensive content package for LLM
        """
        
        print(f"📦 Aggregating {len(scraped_pages)} pages for LLM processing...")
        
        # Sort pages by relevance and content quality
        sorted_pages = self._rank_pages_for_llm(scraped_pages, query)
        
        # Prepare comprehensive content package
        content_package = {
            'query': query,
            'aggregation_timestamp': self.aggregation_timestamp.isoformat(),
            'total_pages': len(sorted_pages),
            'pages': [],
            'summary_stats': self._calculate_summary_stats(sorted_pages),
            'content_index': self._create_content_index(sorted_pages),
        }
        
        # Add full content from each page
        current_tokens = 0
        for i, page in enumerate(sorted_pages):
            
            # Estimate tokens (rough: 1 token ≈ 4 characters)
            page_tokens = len(page.content) // 4
            
            if current_tokens + page_tokens > max_total_tokens and i > 0:
                print(f"⚠️ Token limit reached. Including {i} of {len(sorted_pages)} pages.")
                break
            
            page_data = self._prepare_page_for_llm(page, i + 1)
            content_package['pages'].append(page_data)
            current_tokens += page_tokens
            
            print(f"📄 Added page {i+1}: {page.title} ({page.word_count} words)")
        
        content_package['included_pages'] = len(content_package['pages'])
        content_package['estimated_tokens'] = current_tokens
        
        print(f"✅ Content package ready: {content_package['included_pages']} pages, ~{current_tokens:,} tokens")
        
        return content_package
    
    def _rank_pages_for_llm(self, pages: List[ScrapedPage], query: str) -> List[ScrapedPage]:
        """Rank pages by relevance and quality for LLM processing."""
        
        query_terms = query.lower().split()
        
        def calculate_page_score(page: ScrapedPage) -> float:
            score = 0.0
            
            # Title relevance (high weight)
            if page.title:
                title_lower = page.title.lower()
                title_matches = sum(1 for term in query_terms if term in title_lower)
                score += title_matches * 3.0
            
            # Content relevance
            content_lower = page.content.lower()
            content_matches = sum(1 for term in query_terms if term in content_lower)
            score += content_matches * 1.0
            
            # Content quality indicators
            if page.word_count > 500:
                score += 2.0
            elif page.word_count > 200:
                score += 1.0
            
            # URL quality (academic pages often have better content)
            url_lower = str(page.url).lower()
            quality_indicators = [
                'academics', 'programs', 'admissions', 'requirements',
                'research', 'faculty', 'graduate', 'undergraduate'
            ]
            url_quality = sum(1 for indicator in quality_indicators if indicator in url_lower)
            score += url_quality * 0.5
            
            # Metadata richness
            if page.metadata and isinstance(page.metadata, dict):
                if page.metadata.get('headings'):
                    score += 1.0
                if page.metadata.get('links'):
                    score += 0.5
            
            return score
        
        # Sort by score (descending)
        ranked_pages = sorted(pages, key=calculate_page_score, reverse=True)
        
        print(f"📊 Page ranking completed:")
        for i, page in enumerate(ranked_pages[:5]):  # Show top 5
            score = calculate_page_score(page)
            print(f"   {i+1}. {page.title} (score: {score:.1f}, words: {page.word_count})")
        
        return ranked_pages
    
    def _prepare_page_for_llm(self, page: ScrapedPage, page_number: int) -> Dict[str, Any]:
        """Prepare a single page for LLM processing."""
        
        page_data = {
            'page_number': page_number,
            'title': page.title,
            'url': str(page.url),
            'word_count': page.word_count,
            'content': page.content,
            'scraped_at': page.scraped_at.isoformat() if page.scraped_at else None,
            'content_hash': page.content_hash,
        }
        
        # Add structured metadata if available
        if page.metadata and isinstance(page.metadata, dict):
            
            # Add headings structure
            if 'headings' in page.metadata:
                page_data['headings'] = page.metadata['headings']
            
            # Add internal links
            if 'links' in page.metadata:
                internal_links = [
                    link for link in page.metadata['links'] 
                    if isinstance(link, dict) and link.get('is_internal', False)
                ]
                page_data['internal_links'] = internal_links[:10]  # Limit to prevent bloat
            
            # Add meta tags
            if 'meta_tags' in page.metadata:
                relevant_meta = {}
                for key, value in page.metadata['meta_tags'].items():
                    if key in ['description', 'keywords', 'author', 'og:description']:
                        relevant_meta[key] = value
                if relevant_meta:
                    page_data['meta_tags'] = relevant_meta
        
        return page_data
    
    def _calculate_summary_stats(self, pages: List[ScrapedPage]) -> Dict[str, Any]:
        """Calculate summary statistics for the content package."""
        
        if not pages:
            return {}
        
        total_words = sum(page.word_count for page in pages)
        total_chars = sum(len(page.content) for page in pages)
        
        stats = {
            'total_pages': len(pages),
            'total_words': total_words,
            'total_characters': total_chars,
            'average_words_per_page': total_words / len(pages),
            'word_count_distribution': {
                'min': min(page.word_count for page in pages),
                'max': max(page.word_count for page in pages),
                'median': sorted([page.word_count for page in pages])[len(pages)//2]
            }
        }
        
        return stats
    
    def _create_content_index(self, pages: List[ScrapedPage]) -> Dict[str, Any]:
        """Create an index of the content for quick reference."""
        
        index = {
            'pages_by_title': {},
            'pages_by_domain': {},
            'common_topics': [],
        }
        
        # Index by title
        for i, page in enumerate(pages):
            if page.title:
                index['pages_by_title'][page.title] = i + 1
        
        # Index by domain/subdomain
        for i, page in enumerate(pages):
            try:
                from urllib.parse import urlparse
                parsed = urlparse(str(page.url))
                domain = parsed.netloc
                if domain not in index['pages_by_domain']:
                    index['pages_by_domain'][domain] = []
                index['pages_by_domain'][domain].append(i + 1)
            except:
                continue
        
        # Find common topics (simple keyword extraction)
        all_content = ' '.join(page.content.lower() for page in pages)
        common_words = [
            'admission', 'requirement', 'program', 'degree', 'course',
            'research', 'faculty', 'student', 'graduate', 'undergraduate',
            'application', 'academic', 'engineering', 'science', 'computer'
        ]
        
        for word in common_words:
            count = all_content.count(word)
            if count > 5:  # Appears in multiple contexts
                index['common_topics'].append({'topic': word, 'frequency': count})
        
        # Sort topics by frequency
        index['common_topics'].sort(key=lambda x: x['frequency'], reverse=True)
        index['common_topics'] = index['common_topics'][:10]  # Top 10
        
        return index
    
    def format_for_prompt(self, content_package: Dict[str, Any]) -> str:
        """
        Format the content package as a structured prompt for the LLM.
        
        Args:
            content_package: Aggregated content package
            
        Returns:
            Formatted string ready for LLM processing
        """
        
        prompt_parts = []
        
        # Header
        prompt_parts.append(f"# NCSU Research Query: {content_package['query']}")
        prompt_parts.append(f"Content aggregated from {content_package['included_pages']} pages")
        prompt_parts.append(f"Total content: ~{content_package['estimated_tokens']:,} tokens")
        prompt_parts.append("")
        
        # Summary stats
        stats = content_package['summary_stats']
        prompt_parts.append("## Content Summary")
        prompt_parts.append(f"- Total pages analyzed: {stats.get('total_pages', 0)}")
        prompt_parts.append(f"- Total words: {stats.get('total_words', 0):,}")
        prompt_parts.append(f"- Average words per page: {stats.get('average_words_per_page', 0):.0f}")
        prompt_parts.append("")
        
        # Content index
        if content_package.get('content_index', {}).get('common_topics'):
            prompt_parts.append("## Key Topics Found")
            for topic in content_package['content_index']['common_topics'][:5]:
                prompt_parts.append(f"- {topic['topic'].title()}: {topic['frequency']} mentions")
            prompt_parts.append("")
        
        # Individual pages
        prompt_parts.append("## Page Contents")
        prompt_parts.append("")
        
        for page_data in content_package['pages']:
            prompt_parts.append(f"### Page {page_data['page_number']}: {page_data['title']}")
            prompt_parts.append(f"**URL:** {page_data['url']}")
            prompt_parts.append(f"**Word Count:** {page_data['word_count']}")
            
            # Add headings if available
            if page_data.get('headings'):
                prompt_parts.append("**Page Structure:**")
                for heading in page_data['headings'][:5]:  # Top 5 headings
                    indent = "  " * (heading['level'] - 1)
                    prompt_parts.append(f"{indent}- {heading['text']}")
            
            prompt_parts.append("")
            prompt_parts.append("**Full Content:**")
            prompt_parts.append(page_data['content'])
            prompt_parts.append("")
            prompt_parts.append("---")
            prompt_parts.append("")
        
        return '\n'.join(prompt_parts)
    
    def save_content_package(self, content_package: Dict[str, Any], filename: str) -> None:
        """Save content package to file for debugging/analysis."""
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(content_package, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Content package saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving content package: {str(e)}")
