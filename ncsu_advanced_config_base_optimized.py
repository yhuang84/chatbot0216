#!/usr/bin/env python3
"""
NCSU Advanced Research Assistant - OPTIMIZED VERSION
====================================================

Performance improvements over base version:
- Parallel content extraction using ThreadPoolExecutor
- Parallel LLM grading for faster processing
- Content truncation for grading (configurable)
- Separate models for grading vs answer generation
- Early stopping when enough high-quality content found
- Caching support for previously scraped pages
- Configurable concurrency levels

Features:
- Configurable top-k search results
- 100% content extraction from web pages using MarkItDown
- LLM-based content grading (relevance score 0-1)
- Configurable relevance threshold filtering
- Multiple LLM providers (OpenAI, Anthropic, Mock)
- Embedded configuration (edit the config dict in main())
- Comprehensive logging and result saving

Usage:
    1. Edit the 'config' dictionary in the main() function below
    2. Run: python ncsu_advanced_config_base_optimized.py
"""

import json
import os
import sys
import yaml
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scraper.ncsu_scraper import NCSUScraper
from scraper.content_aggregator import ContentAggregator
from scraper.models import ScrapingConfig
from utils.logger import setup_logger


class ContentCache:
    """Simple in-memory cache for scraped content and grades"""
    
    def __init__(self):
        self.page_cache = {}  # URL -> content
        self.grade_cache = {}  # (content_hash, query) -> score
        self.lock = threading.Lock()
    
    def get_page(self, url: str) -> Optional[str]:
        with self.lock:
            return self.page_cache.get(url)
    
    def set_page(self, url: str, content: str):
        with self.lock:
            self.page_cache[url] = content
    
    def get_grade(self, content: str, query: str) -> Optional[float]:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        key = (content_hash, query)
        with self.lock:
            return self.grade_cache.get(key)
    
    def set_grade(self, content: str, query: str, score: float):
        content_hash = hashlib.md5(content.encode()).hexdigest()
        key = (content_hash, query)
        with self.lock:
            self.grade_cache[key] = score


class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, provider_name: str, model: str = None, temperature: float = 0.7, max_tokens: int = 1000):
        self.provider_name = provider_name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from LLM"""
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""
    
    def __init__(self):
        super().__init__("mock", "mock-model", 0.7, 1000)
    
    def generate_response(self, prompt: str) -> str:
        if "grade" in prompt.lower() or "relevance" in prompt.lower():
            # Return a mock relevance score
            return "0.333"
        else:
            # Return a mock answer
            return f"""Based on the NCSU website content I analyzed, here's what I found regarding your question: "{prompt.split('Question:')[-1].split('Content:')[0].strip() if 'Question:' in prompt else 'your query'}"

**Content Analysis:**
- Analyzed {len(prompt.split())} words of content from NCSU website
- Applied relevance filtering and content grading
- Selected only the most relevant content for this answer

**Key Information from NCSU:**
{prompt.split('Content:')[-1][:200] if 'Content:' in prompt else 'Content analysis completed'}...

**Summary:**
The NCSU website provides comprehensive information about your query. The content above was selected based on relevance scoring and contains the most pertinent details.

*Note: This is a mock response. For AI-generated answers, configure a real LLM provider (OpenAI, Anthropic, or Ollama).*"""


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 8000):
        super().__init__("openai", model, temperature, max_tokens)
        try:
            import openai
            
            # Get API key from environment
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    "❌ OPENAI_API_KEY not found in environment variables!\n"
                    "Please set it using: export OPENAI_API_KEY='your-key-here'\n"
                    "Or add it to the config: 'openai_api_key': 'your-key-here'"
                )
            
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider"""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", temperature: float = 0.7, max_tokens: int = 1000):
        super().__init__("anthropic", model, temperature, max_tokens)
        try:
            import anthropic
            
            # Get API key from environment
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError(
                    "❌ ANTHROPIC_API_KEY not found in environment variables!\n"
                    "Please set it using: export ANTHROPIC_API_KEY='your-key-here'\n"
                    "Or add it to the config: 'anthropic_api_key': 'your-key-here'"
                )
            
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
    
    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"


class NCSUAdvancedResearcher:
    """Advanced NCSU research assistant with performance optimizations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("ncsu_advanced_researcher")
        
        # Initialize cache
        self.cache = ContentCache() if config.get('enable_caching', True) else None
        
        # Initialize LLM providers (separate for grading and answer generation)
        self.grading_provider = self._setup_grading_provider()
        self.answer_provider = self._setup_answer_provider()
        
        # Initialize scraper
        scraper_config = ScrapingConfig(
            selenium_enabled=config.get('selenium_enabled', False),
            enhanced_extraction=config.get('enhanced_extraction', True),
            timeout=config.get('timeout', 30)
        )
        self.scraper = NCSUScraper(config=scraper_config)
        
        # Initialize content aggregator
        self.aggregator = ContentAggregator()
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', 'results'))
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🎯 NCSU Advanced Researcher (OPTIMIZED) initialized")
        print(f"🤖 Grading Provider: {self.grading_provider.provider_name} ({self.grading_provider.model})")
        print(f"🤖 Answer Provider: {self.answer_provider.provider_name} ({self.answer_provider.model})")
        print(f"🔍 Top-K Results: {config.get('top_k', 10)}")
        print(f"📄 Max Pages to Extract: {config.get('max_pages', 5)}")
        print(f"📊 Relevance Threshold: {config.get('relevance_threshold', 0.6)}")
        print(f"🎯 Content Grading: {'Enabled' if config.get('enable_grading', True) else 'Disabled'}")
        print(f"⚡ Parallel Extraction: {config.get('parallel_extraction', True)} (workers: {config.get('extraction_workers', 5)})")
        print(f"⚡ Parallel Grading: {config.get('parallel_grading', True)} (workers: {config.get('grading_workers', 5)})")
        print(f"💾 Caching: {'Enabled' if config.get('enable_caching', True) else 'Disabled'}")
        print(f"🛑 Early Stopping: {'Enabled' if config.get('enable_early_stopping', True) else 'Disabled'}")
        print(f"📁 Output Directory: {self.output_dir}")
    
    def _setup_grading_provider(self) -> LLMProvider:
        """Setup LLM provider for content grading (optimized for speed)"""
        provider_name = self.config.get('grading_provider', self.config.get('llm_provider', 'mock')).lower()
        
        if provider_name == 'openai':
            return OpenAIProvider(
                model=self.config.get('grading_model', 'gpt-4o-mini'),  # Use faster model for grading
                temperature=self.config.get('grading_temperature', 0.3),
                max_tokens=self.config.get('grading_max_tokens', 10)  # Only need a number
            )
        elif provider_name == 'anthropic':
            return AnthropicProvider(
                model=self.config.get('grading_model', 'claude-3-haiku-20240307'),  # Use faster model
                temperature=self.config.get('grading_temperature', 0.3),
                max_tokens=self.config.get('grading_max_tokens', 10)
            )
        else:
            return MockLLMProvider()
    
    def _setup_answer_provider(self) -> LLMProvider:
        """Setup LLM provider for answer generation (optimized for quality)"""
        provider_name = self.config.get('llm_provider', 'mock').lower()
        
        if provider_name == 'openai':
            return OpenAIProvider(
                model=self.config.get('llm_model', 'gpt-4o'),
                temperature=self.config.get('llm_temperature', 0.7),
                max_tokens=self.config.get('llm_max_tokens', 4000)
            )
        elif provider_name == 'anthropic':
            return AnthropicProvider(
                model=self.config.get('llm_model', 'claude-3-sonnet-20240229'),
                temperature=self.config.get('llm_temperature', 0.7),
                max_tokens=self.config.get('llm_max_tokens', 4000)
            )
        else:
            return MockLLMProvider()
    
    def grade_content_relevance(self, content: str, query: str) -> float:
        """Grade content relevance using LLM with caching and truncation"""
        
        # Check cache first
        if self.cache:
            cached_score = self.cache.get_grade(content, query)
            if cached_score is not None:
                return cached_score
        
        # Truncate content for grading (configurable)
        max_grading_chars = self.config.get('max_grading_content_length', 2000)
        content_to_grade = content[:max_grading_chars] if len(content) > max_grading_chars else content
        
        prompt = f"""You are an expert content grader. Grade how relevant this content is to answering the user's query.

USER QUERY: {query}

CONTENT TO GRADE:
{content_to_grade}

GRADING INSTRUCTIONS:
- Analyze the content thoroughly
- Consider how well the content answers or relates to the query
- Ignore navigation menus, headers, and boilerplate text
- Focus on the substantive information that addresses the query
- Consider information quality, accuracy, and completeness

SCORING SCALE:
- 1.0 = Perfect match - content directly and comprehensively answers the query
- 0.8-0.9 = Highly relevant - content strongly relates and provides good information
- 0.6-0.7 = Moderately relevant - content relates but may be incomplete or tangential
- 0.4-0.5 = Somewhat relevant - content has some connection but limited usefulness
- 0.2-0.3 = Minimally relevant - content barely relates to the query
- 0.0-0.1 = Irrelevant - content does not relate to the query

Return ONLY a decimal number between 0.0 and 1.0 (e.g., 0.85):"""
        
        try:
            response = self.grading_provider.generate_response(prompt)
            # Extract number from response
            import re
            match = re.search(r'(\d+\.?\d*)', response)
            if match:
                score = float(match.group(1))
                score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
                
                # Cache the result
                if self.cache:
                    self.cache.set_grade(content, query, score)
                
                return score
            return 0.5  # Default if parsing fails
        except Exception as e:
            self.logger.warning(f"Error grading content: {e}")
            return 0.5
    
    def _extract_single_page(self, result) -> Optional[Dict]:
        """Extract content from a single page (for parallel processing)"""
        try:
            # Check cache first
            if self.cache:
                cached_content = self.cache.get_page(str(result.url))
                if cached_content:
                    return {
                        'title': result.title,
                        'url': str(result.url),
                        'content': cached_content,
                        'word_count': len(cached_content.split()),
                        'extraction_success': True,
                        'cached': True
                    }
            
            # Extract content
            pages = self.scraper.scrape_pages([result])
            if pages and pages[0].extraction_success:
                page = pages[0]
                
                # Cache the result
                if self.cache:
                    self.cache.set_page(str(page.url), page.content)
                
                return {
                    'title': page.title,
                    'url': str(page.url),
                    'content': page.content,
                    'word_count': len(page.content.split()),
                    'extraction_success': True,
                    'cached': False
                }
            return None
        except Exception as e:
            self.logger.warning(f"Error extracting page {result.url}: {e}")
            return None
    
    def _grade_single_page(self, page: Dict, query: str) -> Dict:
        """Grade a single page (for parallel processing)"""
        try:
            relevance_score = self.grade_content_relevance(page['content'], query)
            return {
                'title': page['title'],
                'url': page['url'],
                'content': page['content'],
                'word_count': page['word_count'],
                'relevance_score': relevance_score
            }
        except Exception as e:
            self.logger.warning(f"Error grading page {page['url']}: {e}")
            return {
                'title': page['title'],
                'url': page['url'],
                'content': page['content'],
                'word_count': page['word_count'],
                'relevance_score': 0.5
            }
    
    def generate_answer(self, content: str, query: str, sources: List[Dict]) -> str:
        """Generate final answer using LLM"""
        
        def extract_main_content(content: str) -> str:
            """Extract main content - use full content without truncation"""
            return content
        
        sources_text = "\n".join([
            f"=== SOURCE {i+1}: {source['title']} (Relevance: {source.get('relevance_score', 'N/A')}) ==="
            f"\nURL: {source['url']}"
            f"\nContent: {extract_main_content(source['content'])}\n"
            for i, source in enumerate(sources)
        ])
        
        prompt = f"""You are an expert research assistant. Based on the NCSU website content provided below, answer the user's question comprehensively and accurately.

USER QUESTION: {query}

NCSU WEBSITE CONTENT:
{sources_text}

INSTRUCTIONS:
- Analyze all the provided content thoroughly
- Extract and synthesize relevant information to answer the question
- Provide a comprehensive, well-structured response
- Use specific details and facts from the content
- If the content contains the answer, provide it in full detail
- If the content is incomplete, mention what information is available
- Be accurate and factual - only use information from the provided content
- Organize your response logically with clear paragraphs
- Include specific details, names, dates, and facts when available

COMPREHENSIVE ANSWER:"""
        
        return self.answer_provider.generate_response(prompt)
    
    def research(self, query: str) -> Dict[str, Any]:
        """Conduct advanced research with performance optimizations"""
        print(f"\n🔍 ADVANCED NCSU RESEARCH (OPTIMIZED)")
        print("=" * 70)
        print(f"📋 Query: '{query}'")
        print(f"🤖 Grading Provider: {self.grading_provider.provider_name} ({self.grading_provider.model})")
        print(f"🤖 Answer Provider: {self.answer_provider.provider_name} ({self.answer_provider.model})")
        print(f"🔍 Top-K Results: {self.config.get('top_k', 10)}")
        print(f"📊 Relevance Threshold: {self.config.get('relevance_threshold', 0.6)}")
        
        results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'search_results': [],
            'extracted_pages': [],
            'graded_pages': [],
            'filtered_pages': [],
            'final_answer': '',
            'sources': [],
            'performance_stats': {
                'cached_pages': 0,
                'cached_grades': 0,
                'early_stopped': False
            }
        }
        
        # Step 1: Search NCSU website
        print(f"\n📋 STEP 1: Searching NCSU website for top-k results...")
        print("-" * 50)
        search_results = self.scraper.search(query, max_results=self.config.get('top_k', 10))
        results['search_results'] = [
            {'title': r.title, 'url': str(r.url), 'snippet': r.snippet}
            for r in search_results
        ]
        print(f"✅ Found {len(search_results)} search results")
        
        # Print all search result URLs
        print(f"\n🔗 SEARCH RESULT URLs:")
        for i, result in enumerate(search_results, 1):
            print(f"  {i:2d}. {result.title}")
            print(f"      🌐 {result.url}")
            if result.snippet:
                snippet_preview = result.snippet[:100] + "..." if len(result.snippet) > 100 else result.snippet
                print(f"      📝 {snippet_preview}")
            print()
        
        if not search_results:
            print("❌ No search results found")
            return results
        
        # Step 2: Extract content from top pages (PARALLEL)
        max_pages_config = self.config.get('max_pages', 5)
        pages_to_extract = search_results[:max_pages_config]
        
        print(f"\n📋 STEP 2: Extracting content from top {max_pages_config} pages (PARALLEL)...")
        print("-" * 50)
        print(f"📄 Available search results: {len(search_results)}")
        print(f"📄 Will extract content from {len(pages_to_extract)} pages")
        
        if self.config.get('parallel_extraction', True):
            # Parallel extraction
            extraction_workers = self.config.get('extraction_workers', 5)
            print(f"⚡ Using {extraction_workers} parallel workers for extraction")
            
            extracted_pages = []
            with ThreadPoolExecutor(max_workers=extraction_workers) as executor:
                future_to_result = {executor.submit(self._extract_single_page, result): result for result in pages_to_extract}
                
                for future in as_completed(future_to_result):
                    page_data = future.result()
                    if page_data:
                        extracted_pages.append(page_data)
                        cached_str = " (cached)" if page_data.get('cached') else ""
                        print(f"  ✅ Extracted: {page_data['title']}{cached_str} ({page_data['word_count']:,} words)")
                        if page_data.get('cached'):
                            results['performance_stats']['cached_pages'] += 1
        else:
            # Sequential extraction (fallback)
            print(f"📄 Using sequential extraction")
            extracted_pages = []
            for result in pages_to_extract:
                page_data = self._extract_single_page(result)
                if page_data:
                    extracted_pages.append(page_data)
        
        results['extracted_pages'] = extracted_pages
        total_words = sum(p['word_count'] for p in extracted_pages)
        
        print(f"✅ Extracted content from {len(extracted_pages)} pages")
        print(f"📊 Total content: {total_words:,} words")
        print(f"💾 Cached pages: {results['performance_stats']['cached_pages']}")
        
        if not extracted_pages:
            print("❌ No content extracted")
            return results
        
        # Step 3: Grade content relevance (PARALLEL)
        if self.config.get('enable_grading', True):
            print(f"\n📋 STEP 3: Grading content relevance using LLM (PARALLEL)...")
            print("-" * 50)
            
            if self.config.get('parallel_grading', True):
                # Parallel grading
                grading_workers = self.config.get('grading_workers', 5)
                print(f"⚡ Using {grading_workers} parallel workers for grading")
                
                graded_pages = []
                with ThreadPoolExecutor(max_workers=grading_workers) as executor:
                    future_to_page = {executor.submit(self._grade_single_page, page, query): page for page in extracted_pages}
                    
                    for i, future in enumerate(as_completed(future_to_page), 1):
                        graded_page = future.result()
                        graded_pages.append(graded_page)
                        print(f"  🔍 [{i}/{len(extracted_pages)}] {graded_page['title']}: {graded_page['relevance_score']:.3f}")
            else:
                # Sequential grading (fallback)
                print(f"📄 Using sequential grading")
                graded_pages = []
                for i, page in enumerate(extracted_pages, 1):
                    print(f"🔍 Grading page {i}/{len(extracted_pages)}: {page['title']}")
                    graded_page = self._grade_single_page(page, query)
                    graded_pages.append(graded_page)
                    print(f"   📊 Relevance Score: {graded_page['relevance_score']:.3f}")
            
            results['graded_pages'] = graded_pages
            print(f"✅ Graded {len(graded_pages)} pages using LLM")
        else:
            graded_pages = [
                {
                    'title': page['title'],
                    'url': page['url'],
                    'content': page['content'],
                    'word_count': page['word_count'],
                    'relevance_score': 1.0
                }
                for page in extracted_pages
            ]
            results['graded_pages'] = graded_pages
        
        # Step 4: Filter by relevance threshold with early stopping
        print(f"\n📋 STEP 4: Filtering by relevance threshold ({self.config.get('relevance_threshold', 0.6)})...")
        print("-" * 50)
        
        threshold = self.config.get('relevance_threshold', 0.6)
        filtered_pages = [p for p in graded_pages if p['relevance_score'] >= threshold]
        
        # Sort by relevance score
        filtered_pages.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Early stopping: if we have enough high-quality content, stop
        if self.config.get('enable_early_stopping', True):
            early_stop_threshold = self.config.get('early_stop_threshold', 0.85)
            early_stop_min_pages = self.config.get('early_stop_min_pages', 3)
            
            high_quality_pages = [p for p in filtered_pages if p['relevance_score'] >= early_stop_threshold]
            if len(high_quality_pages) >= early_stop_min_pages:
                print(f"🛑 Early stopping: Found {len(high_quality_pages)} pages with relevance ≥ {early_stop_threshold}")
                filtered_pages = high_quality_pages[:early_stop_min_pages]
                results['performance_stats']['early_stopped'] = True
        
        if not filtered_pages:
            print(f"⚠️ No pages meet threshold {threshold}, using top page")
            filtered_pages = [max(graded_pages, key=lambda x: x['relevance_score'])]
        
        results['filtered_pages'] = filtered_pages
        filtered_words = sum(p['word_count'] for p in filtered_pages)
        
        print(f"✅ {len(filtered_pages)} pages meet relevance threshold")
        print(f"📊 Filtered content: {filtered_words:,} words")
        
        print(f"\n📊 Filtered Pages (relevance ≥ {threshold}):")
        for i, page in enumerate(filtered_pages, 1):
            print(f"  {i}. {page['title']} (score: {page['relevance_score']:.3f})")
        
        # Step 5: Generate final answer
        print(f"\n📋 STEP 5: Generating LLM answer from filtered content...")
        print("-" * 50)
        
        # Prepare content for LLM
        combined_content = "\n\n".join([
            f"Title: {page['title']}\nURL: {page['url']}\nContent: {page['content']}"
            for page in filtered_pages
        ])
        
        print(f"📝 Generating answer from {len(combined_content):,} characters of filtered content...")
        final_answer = self.generate_answer(combined_content, query, filtered_pages)
        results['final_answer'] = final_answer
        
        print(f"✅ Generated LLM answer ({len(final_answer):,} characters)")
        
        # Prepare sources
        results['sources'] = [
            {
                'title': page['title'],
                'url': page['url'],
                'relevance_score': page['relevance_score'],
                'word_count': page['word_count']
            }
            for page in filtered_pages
        ]
        
        return results
    
    def save_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Save research results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(c for c in results['query'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        query_short = query_safe[:50].replace(' ', '_')
        
        files = {}
        
        # Save answer
        answer_file = self.output_dir / f"answer_{query_short}_{timestamp}.txt"
        with open(answer_file, 'w', encoding='utf-8') as f:
            f.write(f"Query: {results['query']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Grading Provider: {self.grading_provider.provider_name} ({self.grading_provider.model})\n")
            f.write(f"Answer Provider: {self.answer_provider.provider_name} ({self.answer_provider.model})\n\n")
            f.write("ANSWER:\n")
            f.write("=" * 50 + "\n")
            f.write(results['final_answer'])
            f.write("\n\n" + "=" * 50 + "\n")
            f.write("SOURCES:\n")
            for i, source in enumerate(results['sources'], 1):
                f.write(f"[{i}] {source['title']} (Relevance: {source['relevance_score']:.3f})\n")
                f.write(f"    {source['url']}\n")
                f.write(f"    ({source['word_count']:,} words)\n\n")
            
            # Add performance stats
            f.write("\n" + "=" * 50 + "\n")
            f.write("PERFORMANCE STATS:\n")
            f.write(f"Cached pages: {results['performance_stats']['cached_pages']}\n")
            f.write(f"Early stopped: {results['performance_stats']['early_stopped']}\n")
        files['answer'] = str(answer_file)
        
        # Save data
        data_file = self.output_dir / f"data_{query_short}_{timestamp}.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        files['data'] = str(data_file)
        
        # Save config
        config_file = self.output_dir / f"config_{query_short}_{timestamp}.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(results['config'], f, default_flow_style=False)
        files['config'] = str(config_file)
        
        return files
    
    def display_results(self, results: Dict[str, Any]):
        """Display research results"""
        print(f"\n📋 STEP 6: Results")
        print("-" * 50)
        print(f"\n🔍 QUERY: {results['query']}")
        print(f"\n🤖 LLM ANSWER:")
        print(results['final_answer'])
        
        print(f"\n📚 SOURCES (Filtered by Relevance ≥ {self.config.get('relevance_threshold', 0.6)}):")
        for i, source in enumerate(results['sources'], 1):
            print(f"[{i}] {source['title']} (Relevance: {source['relevance_score']:.3f})")
            print(f"    {source['url']}")
            print(f"    ({source['word_count']:,} words)")
            print()
        
        print(f"\n⚡ PERFORMANCE STATS:")
        print(f"💾 Cached pages: {results['performance_stats']['cached_pages']}")
        print(f"🛑 Early stopped: {results['performance_stats']['early_stopped']}")


def main():
    """Main function with optimized embedded configuration"""
    
    # ========================================
    # 🔑 LOAD ENVIRONMENT VARIABLES FROM .env
    # ========================================
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded environment variables from .env file")
    except ImportError:
        print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    except Exception as e:
        print(f"⚠️ Could not load .env file: {e}")
    
    # ========================================
    # 🔧 OPTIMIZED CONFIGURATION - EDIT HERE
    # ========================================
    
    config = {
        # 📋 Query Configuration
        'query': 'What are the requirements for the Computer Science major at NC State University?',
        
        # 🤖 LLM Configuration for Answer Generation
        'llm_provider': 'openai',  # Options: 'mock', 'openai', 'anthropic'
        'llm_model': 'gpt-4o',     # High-quality model for final answer
        'llm_temperature': 0.3,
        'llm_max_tokens': 4000,
        
        # 🎯 LLM Configuration for Content Grading (OPTIMIZED)
        'grading_provider': 'openai',  # Can be different from llm_provider
        'grading_model': 'gpt-4o-mini',  # Fast, cheap model for grading
        'grading_temperature': 0.3,
        'grading_max_tokens': 10,  # Only need a number
        'max_grading_content_length': 2000,  # Truncate content for faster grading
        
        # 🔍 Search Configuration (OPTIMIZED)
        'top_k': 10,              # Reduced from 20 for faster processing
        'max_pages': 8,           # Reduced from 20 for faster processing
        'search_timeout': 30,
        'extraction_timeout': 60,
        
        # 📊 Content Processing Configuration
        'relevance_threshold': 0.6,    # Filter threshold
        'enable_grading': True,
        'min_content_length': 100,
        'max_content_length': 50000,
        
        # ⚡ PERFORMANCE OPTIMIZATIONS (NEW)
        'parallel_extraction': True,    # Enable parallel page extraction
        'extraction_workers': 5,        # Number of parallel workers for extraction
        'parallel_grading': True,       # Enable parallel content grading
        'grading_workers': 5,           # Number of parallel workers for grading
        'enable_caching': True,         # Cache scraped pages and grades
        'enable_early_stopping': True,  # Stop when enough high-quality content found
        'early_stop_threshold': 0.85,   # Relevance threshold for early stopping
        'early_stop_min_pages': 3,      # Minimum pages needed for early stopping
        
        # 🚀 Extraction Configuration
        'selenium_enabled': True,
        'enhanced_extraction': True,
        'user_agent': 'NCSU Research Assistant Bot 1.0',
        'delay': 1.0,
        'max_retries': 3,
        
        # 💾 Output Configuration
        'output_dir': 'results',
        'save_config': True,
        'save_data': True,
        'save_answer': True,
        'log_level': 'INFO',
        
        # ⚙️ Advanced Configuration
        'concurrent_requests': 8,
        'respect_robots_txt': True,
        'verify_ssl': True,
        
        # 🔑 API Keys (optional - can also use environment variables)
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        
        # Legacy/compatibility
        'timeout': 30
    }
    
    # ========================================
    # 🔑 API Key Setup
    # ========================================
    
    if config.get('openai_api_key') and isinstance(config.get('openai_api_key'), str):
        os.environ['OPENAI_API_KEY'] = config['openai_api_key']
    
    if config.get('anthropic_api_key') and isinstance(config.get('anthropic_api_key'), str):
        os.environ['ANTHROPIC_API_KEY'] = config['anthropic_api_key']
    
    # ========================================
    # 🚀 EXECUTION SECTION
    # ========================================
    
    print("🎯 NCSU Advanced Research Assistant (OPTIMIZED)")
    print("=" * 50)
    print(f"📋 Query: {config['query']}")
    print(f"🤖 Grading Provider: {config['grading_provider']} ({config['grading_model']})")
    print(f"🤖 Answer Provider: {config['llm_provider']} ({config['llm_model']})")
    print(f"🔍 Top-K Results: {config['top_k']}")
    print(f"📄 Max Pages: {config['max_pages']}")
    print(f"📊 Relevance Threshold: {config['relevance_threshold']}")
    print(f"⚡ Parallel Processing: Extraction={config['parallel_extraction']}, Grading={config['parallel_grading']}")
    print(f"💾 Caching: {config['enable_caching']}")
    print(f"🛑 Early Stopping: {config['enable_early_stopping']}")
    print("=" * 50)
    
    try:
        # Initialize researcher
        researcher = NCSUAdvancedResearcher(config)
        
        # Conduct research
        results = researcher.research(config['query'])
        
        # Display results
        researcher.display_results(results)
        
        # Save results
        print(f"\n📋 STEP 7: Saving results...")
        print("-" * 50)
        saved_files = researcher.save_results(results)
        
        for file_type, file_path in saved_files.items():
            print(f"💾 {file_type.title()} saved to: {file_path}")
        
        # Final summary
        print(f"\n🎉 OPTIMIZED RESEARCH COMPLETE!")
        print("=" * 70)
        print(f"✅ Query: '{config['query']}'")
        print(f"✅ Found {len(results['search_results'])} search results")
        print(f"✅ Extracted content from {len(results['extracted_pages'])} pages ({sum(p['word_count'] for p in results['extracted_pages']):,} words)")
        if config.get('enable_grading', True):
            print(f"✅ Graded {len(results['graded_pages'])} pages using LLM")
        print(f"✅ Filtered to {len(results['filtered_pages'])} relevant pages ({sum(p['word_count'] for p in results['filtered_pages']):,} words)")
        print(f"✅ Generated LLM answer ({len(results['final_answer']):,} characters)")
        print(f"🤖 Grading Provider: {researcher.grading_provider.provider_name} ({researcher.grading_provider.model})")
        print(f"🤖 Answer Provider: {researcher.answer_provider.provider_name} ({researcher.answer_provider.model})")
        print(f"📊 Relevance Threshold: {config.get('relevance_threshold', 0.6)}")
        print(f"💾 Cached pages: {results['performance_stats']['cached_pages']}")
        print(f"🛑 Early stopped: {results['performance_stats']['early_stopped']}")
        
        print(f"\n🎯 SUCCESS! Optimized research completed for: '{config['query']}'")
        print(f"📄 Answer: {saved_files['answer']}")
        print(f"📊 Data: {saved_files['data']}")
        print(f"⚙️ Config: {saved_files['config']}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Research interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during research: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
