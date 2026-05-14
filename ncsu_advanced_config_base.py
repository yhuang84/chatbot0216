#!/usr/bin/env python3
"""
NCSU Advanced Research Assistant - HYBRID VERSION (v2)
======================================================

Changes vs prior version:
- Prompt: concise, Perplexity-style with inline [n] citations
- Grading: shorter prompt + smaller content window (faster + cheaper)
- URL deduplication in build_prompt
- Config tuned for shorter, more precise answers
"""

import json
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scraper.ncsu_scraper import NCSUScraper
from scraper.content_aggregator import ContentAggregator
from scraper.models import ScrapingConfig
from utils.logger import setup_logger


# ========================================
# 🔧 LLM PROVIDER CLASSES
# ========================================

class LLMProvider:
    """Base class for LLM providers"""
    def __init__(self, provider_name: str, model: str = None, temperature: float = 0.7, max_tokens: int = 1000):
        self.provider_name = provider_name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_response(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    def __init__(self):
        super().__init__("mock", "mock-model", 0.7, 1000)

    def generate_response(self, prompt: str) -> str:
        if "score" in prompt.lower() or "rate" in prompt.lower():
            return "0.85"
        return "Mock response for testing."


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 1500):
        super().__init__("openai", model, temperature, max_tokens)
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                try:
                    import streamlit as st
                    api_key = st.secrets["openai"]["api_key"]
                except Exception:
                    pass

            if not api_key:
                print("⚠️ Warning: OPENAI_API_KEY not found.")
            else:
                self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed.")

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider"""
    def __init__(self, model: str = "claude-3-sonnet-20240229", temperature: float = 0.7, max_tokens: int = 1500):
        super().__init__("anthropic", model, temperature, max_tokens)
        try:
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                print("⚠️ Warning: ANTHROPIC_API_KEY not found.")
            else:
                self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed.")

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"


# ========================================
# 🧠 MAIN RESEARCHER CLASS
# ========================================

class NCSUAdvancedResearcher:
    """
    Advanced NCSU research assistant.
    Logic: Batch scrape, serial grade, filter, then build a Perplexity-style prompt.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("ncsu_advanced_researcher")

        # Initialize LLM providers
        self.grading_provider = self._setup_provider('grading')
        self.answer_provider = self._setup_provider('llm')

        # Initialize scraper
        scraper_config = ScrapingConfig(
            selenium_enabled=config.get('selenium_enabled', False),
            enhanced_extraction=config.get('enhanced_extraction', True),
            timeout=config.get('timeout', 30),
        )
        self.scraper = NCSUScraper(config=scraper_config)
        self.aggregator = ContentAggregator()

        # Output setup
        self.output_dir = Path(config.get('output_dir', 'results'))
        self.output_dir.mkdir(exist_ok=True)

        print(f"🎯 NCSU Advanced Researcher Initialized")
        print(f"🤖 Grading Model: {self.grading_provider.model}")
        print(f"🤖 Answer Model:  {self.answer_provider.model}")

    def _setup_provider(self, prefix: str) -> LLMProvider:
        """Setup providers based on config keys (e.g., 'llm_provider' or 'grading_provider')"""
        name = self.config.get(f'{prefix}_provider', 'mock').lower()
        model = self.config.get(f'{prefix}_model')
        temp = self.config.get(f'{prefix}_temperature', 0.7)
        # Different defaults for grading vs answer
        default_tokens = 10 if prefix == 'grading' else 1500
        tokens = self.config.get(f'{prefix}_max_tokens', default_tokens)

        if name == 'openai':
            return OpenAIProvider(model=model, temperature=temp, max_tokens=tokens)
        elif name == 'anthropic':
            return AnthropicProvider(model=model, temperature=temp, max_tokens=tokens)
        return MockLLMProvider()

    # ----------------------------------------------------------------
    # CHANGED: shorter grading prompt + smaller content window
    # ----------------------------------------------------------------
    def grade_content_relevance(self, content: str, query: str) -> float:
        """Rate content relevance to the query. Compact prompt for speed/cost."""
        prompt = f"""Rate how well this content answers the query.

QUERY: {query}

CONTENT:
{content[:5000]}

Return ONLY a number 0.0–1.0:
- 1.0 = directly answers the query
- 0.5 = related but doesn't directly answer
- 0.0 = irrelevant

Score:"""

        try:
            response = self.grading_provider.generate_response(prompt)
            import re
            match = re.search(r'(\d+\.?\d*)', response)
            if match:
                return max(0.0, min(1.0, float(match.group(1))))
            return 0.5
        except Exception as e:
            self.logger.warning(f"Error grading content: {e}")
            return 0.5

    # ----------------------------------------------------------------
    # CHANGED: Perplexity-style prompt — concise + inline [n] citations
    # ----------------------------------------------------------------
    def build_prompt(self, query: str, sources: List[Dict]) -> str:
        """
        Build a concise, Perplexity-style prompt:
        - Inline [n] citations matching numbered sources
        - 3–6 sentence answer, no preamble
        - Explicit "say so" fallback if info is missing
        """
        # Deduplicate URLs, keep first occurrence (already sorted by relevance)
        seen_urls = set()
        unique_sources = []
        for s in sources:
            url = s.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(s)

        # Per-source content cap: enough context, not enough to bloat
        per_source_chars = 8000
        sources_text = "\n\n".join([
            f"[{i+1}] {s['title']}\nURL: {s['url']}\n{s['content'][:per_source_chars]}"
            for i, s in enumerate(unique_sources)
        ])

        prompt = f"""You are a research assistant answering questions using NCSU website content. Write comprehensive, well-sourced answers in clear prose.

QUESTION: {query}

SOURCES:
{sources_text}

ANSWER FORMAT:
- Write 2–4 short paragraphs. Each paragraph covers one distinct aspect of the answer.
- Synthesize information across at least 3 different sources whenever the sources support it.
- Cover what, who, how, and any specifics (names, contacts, dates, processes, links) the sources provide.
- Use a bulleted list ONLY when enumerating discrete items (e.g., listing multiple labs, scholarships, contacts, programs). Otherwise write prose.

CITATIONS:
- Cite every factual claim inline using [n] matching the source number, e.g. "Dr. Smith leads the yarn lab [2]."
- Combine sources when they add complementary information.
- If multiple sources support the same claim, cite together: [1][3].
- Only cite sources you actually used.

CONTENT:
- Use ONLY information from the sources. If they don't contain the answer, say so clearly.
- No preamble. Start the answer with the substance, not "Based on the sources..." or restating the question.
- Do not invent URLs, names, dates, or numbers.

ANSWER:
"""
        return prompt

    def generate_answer(self, content: str, query: str, sources: List[Dict]) -> str:
        """Non-streaming answer generation (for terminal use)"""
        prompt = self.build_prompt(query, sources)
        print(f"📝 Generating answer from {len(sources)} sources...")
        return self.answer_provider.generate_response(prompt)

    def research(self, query: str) -> Dict[str, Any]:
        """
        Conduct research: Search → Batch Scrape → Serial Grade → Filter.
        Does NOT generate the answer (UI handles streaming).
        """
        print(f"\n🔍 RESEARCH STARTED: '{query}'")

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
        }

        # 1. Search
        print(f"\n📋 STEP 1: Searching NCSU (Top-K={self.config.get('top_k')})...")
        if self.config.get('message_callback'):
            self.config['message_callback']("📋 STEP 1: Searching NCSU...")

        search_results = self.scraper.search(query, max_results=self.config.get('top_k', 30))
        results['search_results'] = [{'title': r.title, 'url': str(r.url)} for r in search_results]

        if not search_results:
            print("❌ No search results found.")
            if self.config.get('message_callback'):
                self.config['message_callback']("❌ No search results found")
            return results

        if self.config.get('message_callback'):
            self.config['message_callback']("✅ Search completed successfully")

        # 2. Extract
        max_pages = self.config.get('max_pages', 20)
        pages_to_extract = search_results[:max_pages]
        print(f"\n📋 STEP 2: Batch Extracting {len(pages_to_extract)} pages...")
        if self.config.get('message_callback'):
            self.config['message_callback']("📋 STEP 2: Extracting content from pages...")

        scraped_pages = self.scraper.scrape_pages(pages_to_extract)
        results['extracted_pages'] = [
            {
                'title': p.title,
                'url': str(p.url),
                'content': p.content,
                'word_count': len(p.content.split()),
                'extraction_success': p.extraction_success,
            }
            for p in scraped_pages
        ]
        successful_pages = [p for p in results['extracted_pages'] if p['extraction_success']]
        print(f"✅ Extracted {len(successful_pages)} pages.")
        if self.config.get('message_callback'):
            self.config['message_callback']("✅ Successfully extracted content from pages")

        # 3. Grade
        if self.config.get('enable_grading', True):
            print(f"\n📋 STEP 3: Grading content (Serial)...")
            if self.config.get('message_callback'):
                self.config['message_callback']("📋 STEP 3: Analyzing content relevance...")

            graded_pages = []
            for i, page in enumerate(successful_pages, 1):
                score = self.grade_content_relevance(page['content'], query)
                print(f"  [{i}] {page['title'][:30]}... Score: {score:.2f}")
                if self.config.get('message_callback'):
                    self.config['message_callback'](
                        f"  🔍 Analyzing: {page['title'][:50]}... Score: {score:.2f}"
                    )
                graded_pages.append({**page, 'relevance_score': score})
            results['graded_pages'] = graded_pages

            if self.config.get('message_callback'):
                self.config['message_callback']("✅ Content analysis complete")
        else:
            results['graded_pages'] = [{**p, 'relevance_score': 1.0} for p in successful_pages]

        # 4. Filter
        print(f"\n📋 STEP 4: Filtering...")
        if self.config.get('message_callback'):
            self.config['message_callback']("📋 STEP 4: Filtering by relevance...")

        threshold = self.config.get('relevance_threshold', 0.25)
        min_sources = self.config.get('min_sources', 5)  # Always keep at least N sources

        # Sort all graded pages by relevance first
        all_sorted = sorted(results['graded_pages'], key=lambda x: x['relevance_score'], reverse=True)
        filtered_pages = [p for p in all_sorted if p['relevance_score'] >= threshold]

        # Top-N fallback: if threshold filter is too aggressive, ensure min_sources
        if len(filtered_pages) < min_sources and all_sorted:
            print(f"⚠️ Only {len(filtered_pages)} passed threshold; topping up to {min_sources}.")
            if self.config.get('message_callback'):
                self.config['message_callback'](
                    f"⚠️ Topping up to {min_sources} sources (threshold too strict)"
                )
            filtered_pages = all_sorted[:min_sources]

        # Fallback: keep top result if literally nothing
        if not filtered_pages and results['graded_pages']:
            print("⚠️ No graded content; using top result.")
            if self.config.get('message_callback'):
                self.config['message_callback']("⚠️ Threshold too high, keeping top result")
            filtered_pages = [max(results['graded_pages'], key=lambda x: x['relevance_score'])]

        # Sort by relevance
        filtered_pages.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Deduplicate by URL (preserve relevance order)
        seen = set()
        deduped = []
        for p in filtered_pages:
            if p['url'] not in seen:
                seen.add(p['url'])
                deduped.append(p)
        filtered_pages = deduped

        results['filtered_pages'] = filtered_pages
        results['sources'] = [
            {
                'title': p['title'],
                'url': p['url'],
                'relevance_score': p['relevance_score'],
                'word_count': p['word_count'],
                'content': p['content'],
            }
            for p in filtered_pages
        ]

        print(f"✅ Ready with {len(filtered_pages)} relevant pages.")
        if self.config.get('message_callback'):
            self.config['message_callback']("✅ Research complete - ready to generate answer")

        return results

    def save_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}

        # Answer file
        ans_file = self.output_dir / f"answer_{timestamp}.txt"
        with open(ans_file, 'w', encoding='utf-8') as f:
            f.write(f"Query: {results['query']}\n\nANSWER:\n{results['final_answer']}\n\nSOURCES:\n")
            for i, s in enumerate(results['sources'], 1):
                f.write(f"[{i}] {s['title']} — {s['url']}\n")
        files['answer'] = str(ans_file)

        # JSON data
        json_file = self.output_dir / f"data_{timestamp}.json"
        clean_config = {k: v for k, v in results['config'].items() if not callable(v)}
        results['config'] = clean_config
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        files['data'] = str(json_file)

        return files

    def display_results(self, results: Dict[str, Any]):
        print(f"\n🤖 ANSWER:\n{results['final_answer']}")
        print(f"\n📚 SOURCES:")
        for i, s in enumerate(results['sources'], 1):
            print(f"[{i}] {s['title']} ({s['relevance_score']:.2f}) — {s['url']}")


def main():
    """Main function with embedded config (terminal mode)"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    # ==========================================
    # ⚙️ CONFIGURATION (tuned for concise, cited answers)
    # ==========================================
    config = {
        'query': 'Who is doing research on yarn?',

        # 1. Answer Model
        'llm_provider': 'openai',
        'llm_model': 'gpt-4o',
        'llm_temperature': 0.1,
        'llm_max_tokens': 1500,           # ← was 8000; hard cap output length

        # 2. Grading Model
        'grading_provider': 'openai',
        'grading_model': 'gpt-4o-mini',
        'grading_temperature': 0.0,
        'grading_max_tokens': 10,         # ← grading only needs a number

        # 3. Search Settings
        'top_k': 30,
        'max_pages': 20,
        'relevance_threshold': 0.25,      # ← lowered: more sources reach the LLM
        'min_sources': 5,                 # ← new: guarantee at least N sources

        # 4. Features
        'enable_grading': True,
        'selenium_enabled': True,
        'enhanced_extraction': True,
        'output_dir': 'results',

        # API Keys
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
    }

    if config.get('openai_api_key'):
        os.environ['OPENAI_API_KEY'] = config['openai_api_key']

    print("🚀 Running NCSU Research Assistant (Hybrid Serial Mode)")
    print("=" * 50)

    try:
        researcher = NCSUAdvancedResearcher(config)
        results = researcher.research(config['query'])

        print(f"\n📋 STEP 5: Generating Answer (Terminal Mode)...")
        final_answer = researcher.generate_answer('', config['query'], results['filtered_pages'])
        results['final_answer'] = final_answer

        researcher.display_results(results)
        researcher.save_results(results)
        print("\n✅ Done!")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
