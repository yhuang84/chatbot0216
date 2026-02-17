#!/usr/bin/env python3
"""
NCSU Advanced Research Assistant - HYBRID VERSION
=================================================

Logic: "Old Code" (Serial, Batch, Full Content, High Quality)
Feature: Streaming Ready (Separated research from answer generation)

Usage:
    1. Edit the config dictionary in main()
    2. Run: python ncsu_advanced_config_base.py
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
# 🔧 LLM PROVIDER CLASSES (Old Code Style)
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
        if "grade" in prompt.lower(): return "0.85"
        return "Mock response for testing."

class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 8000):
        super().__init__("openai", model, temperature, max_tokens)
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                # Try to find it in streamlit secrets if running via streamlit
                try:
                    import streamlit as st
                    api_key = st.secrets["openai"]["api_key"]
                except:
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
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key: print("⚠️ Warning: ANTHROPIC_API_KEY not found.")
            else: self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed.")
    
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


# ========================================
# 🧠 MAIN RESEARCHER CLASS
# ========================================

class NCSUAdvancedResearcher:
    """
    Advanced NCSU research assistant.
    Logic: Strictly follows 'Old Code' logic (Batch scrape, Serial grade, No truncation).
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
            timeout=config.get('timeout', 30)
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
        """Helper to setup providers based on config keys (e.g., 'llm_provider' or 'grading_provider')"""
        name = self.config.get(f'{prefix}_provider', 'mock').lower()
        model = self.config.get(f'{prefix}_model')
        temp = self.config.get(f'{prefix}_temperature', 0.7)
        tokens = self.config.get(f'{prefix}_max_tokens', 4000)
        
        if name == 'openai':
            return OpenAIProvider(model=model, temperature=temp, max_tokens=tokens)
        elif name == 'anthropic':
            return AnthropicProvider(model=model, temperature=temp, max_tokens=tokens)
        return MockLLMProvider()

    def grade_content_relevance(self, content: str, query: str) -> float:
        """
        Grade content relevance.
        LOGIC: Old Code Style - No complex truncation, just raw content grading.
        """
        prompt = f"""You are an expert content grader. Grade how relevant this content is to answering the user's query.

USER QUERY: {query}

CONTENT TO GRADE:
{content[:25000]} 

GRADING INSTRUCTIONS:
- Analyze the content thoroughly.
- 1.0 = Perfect match, 0.0 = Irrelevant.
- Return ONLY a decimal number between 0.0 and 1.0.

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

    def build_prompt(self, query: str, sources: List[Dict]) -> str:
        """
        Builds the final prompt for the Answer LLM.
        This is separated so the UI can use it for streaming.
        LOGIC: Old Code Style - "Generous" inclusion, no fancy token counting/cutting.
        """
        sources_text = "\n".join([
            f"=== SOURCE {i+1}: {source['title']} (Relevance: {source.get('relevance_score', 'N/A')}) ==="
            f"\nURL: {source['url']}"
            f"\nContent: {source['content']}\n"
            for i, source in enumerate(sources)
        ])
        
        prompt = f"""You are an expert research assistant. Based on the NCSU website content provided below, answer the user's question comprehensively and accurately.

USER QUESTION: {query}

NCSU WEBSITE CONTENT:
{sources_text}

INSTRUCTIONS:
- Analyze all the provided content thoroughly.
- Extract and synthesize relevant information.
- Provide a comprehensive, well-structured response.
- Use specific details, names, dates, and facts from the content.
- Cite the sources implicitly by context.

COMPREHENSIVE ANSWER:"""
        return prompt

    def generate_answer(self, content: str, query: str, sources: List[Dict]) -> str:
        """Non-streaming answer generation (for Terminal use)"""
        prompt = self.build_prompt(query, sources)
        print(f"📝 Generating answer from {len(sources)} sources...")
        return self.answer_provider.generate_response(prompt)

    def research(self, query: str) -> Dict[str, Any]:
        """
        Conduct research.
        LOGIC: Old Code Flow (Search -> Batch Scrape -> Serial Grade -> Filter).
        KEY CHANGE: Does NOT call generate_answer() internally. Returns results for Streamlit.
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
            'sources': []
        }
        
        # 1. Search (Old Code: Just call search)
        print(f"\n📋 STEP 1: Searching NCSU (Top-K={self.config.get('top_k')})...")
        search_results = self.scraper.search(query, max_results=self.config.get('top_k', 30))
        results['search_results'] = [{'title': r.title, 'url': str(r.url)} for r in search_results]
        
        if not search_results:
            print("❌ No search results found.")
            return results
            
        # 2. Extract (Old Code: Batch extract using scraper.scrape_pages list)
        max_pages = self.config.get('max_pages', 20)
        pages_to_extract = search_results[:max_pages]
        print(f"\n📋 STEP 2: Batch Extracting {len(pages_to_extract)} pages...")
        
        # Using the scraper's built-in batch method (Old Code Style)
        scraped_pages = self.scraper.scrape_pages(pages_to_extract)
        
        results['extracted_pages'] = [
            {'title': p.title, 'url': str(p.url), 'content': p.content, 'word_count': len(p.content.split()), 'extraction_success': p.extraction_success}
            for p in scraped_pages
        ]
        successful_pages = [p for p in results['extracted_pages'] if p['extraction_success']]
        print(f"✅ Extracted {len(successful_pages)} pages.")

        # 3. Grade (Old Code: Serial Loop)
        if self.config.get('enable_grading', True):
            print(f"\n📋 STEP 3: Grading content (Serial)...")
            graded_pages = []
            for i, page in enumerate(successful_pages, 1):
                score = self.grade_content_relevance(page['content'], query)
                print(f"  [{i}] {page['title'][:30]}... Score: {score:.2f}")
                graded_pages.append({**page, 'relevance_score': score})
            results['graded_pages'] = graded_pages
        else:
            results['graded_pages'] = [{**p, 'relevance_score': 1.0} for p in successful_pages]

        # 4. Filter (Old Code Logic)
        print(f"\n📋 STEP 4: Filtering...")
        threshold = self.config.get('relevance_threshold', 0.1)
        filtered_pages = [p for p in results['graded_pages'] if p['relevance_score'] >= threshold]
        
        # Fallback if everything filtered out
        if not filtered_pages and results['graded_pages']:
            print("⚠️ Threshold too high, using top result.")
            filtered_pages = [max(results['graded_pages'], key=lambda x: x['relevance_score'])]

        # Sort by relevance
        filtered_pages.sort(key=lambda x: x['relevance_score'], reverse=True)
        results['filtered_pages'] = filtered_pages
        
        # Prepare sources list for UI
        results['sources'] = [
            {'title': p['title'], 'url': p['url'], 'relevance_score': p['relevance_score'], 'word_count': p['word_count'], 'content': p['content']}
            for p in filtered_pages
        ]
        
        print(f"✅ Ready with {len(filtered_pages)} relevant pages.")
        
        # STOP HERE. Do not generate answer. Return results so UI can stream.
        return results

    def save_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}
        
        # Answer File
        ans_file = self.output_dir / f"answer_{timestamp}.txt"
        with open(ans_file, 'w', encoding='utf-8') as f:
            f.write(f"Query: {results['query']}\n\nANSWER:\n{results['final_answer']}\n\nSOURCES:\n")
            for s in results['sources']: f.write(f"- {s['title']} ({s['url']})\n")
        files['answer'] = str(ans_file)
        
        # JSON Data
        json_file = self.output_dir / f"data_{timestamp}.json"
        # Filter non-serializable config
        clean_config = {k:v for k,v in results['config'].items() if not callable(v)}
        results['config'] = clean_config
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        files['data'] = str(json_file)
        
        return files
    
    def display_results(self, results: Dict[str, Any]):
        print(f"\n🤖 ANSWER:\n{results['final_answer']}")
        print(f"\n📚 SOURCES:")
        for i, s in enumerate(results['sources'], 1):
            print(f"[{i}] {s['title']} ({s['relevance_score']:.2f})")


def main():
    """Main function with Embedded Configuration (Old Code Style)"""
    
    # Load env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except: pass

    # ==========================================
    # ⚙️ CONFIGURATION (High Quality Defaults)
    # ==========================================
    config = {
        'query': 'Who is doing research on yarn?',
        
        # 1. Answer Model: [CRITICAL] Use GPT-4o for best quality (Old Code Standard)
        'llm_provider': 'openai',
        'llm_model': 'gpt-4o',
        'llm_temperature': 0.1,
        'llm_max_tokens': 8000,
        
        # 2. Grading Model: Cheap but effective
        'grading_provider': 'openai',
        'grading_model': 'gpt-4o-mini',
        'grading_temperature': 0.0,
        
        # 3. Search Settings: "Generous" (High count to ensure coverage)
        'top_k': 30,          # Search 30 results
        'max_pages': 20,      # Scrape top 20
        'relevance_threshold': 0.1, # Keep almost everything
        
        # 4. Features
        'enable_grading': True,
        'selenium_enabled': True,
        'enhanced_extraction': True,
        'output_dir': 'results',
        
        # API Keys
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
    }
    
    if config.get('openai_api_key'): os.environ['OPENAI_API_KEY'] = config['openai_api_key']

    print("🚀 Running NCSU Research Assistant (Hybrid Serial Mode)")
    print("="*50)

    try:
        researcher = NCSUAdvancedResearcher(config)
        
        # 1. Run Research (Steps 1-4)
        results = researcher.research(config['query'])
        
        # 2. Generate Answer (Step 5 - Terminal Mode Only)
        # Note: UI will use researcher.build_prompt() and st.write_stream instead.
        print(f"\n📋 STEP 5: Generating Answer (Terminal Mode)...")
        final_answer = researcher.generate_answer('', config['query'], results['filtered_pages'])
        results['final_answer'] = final_answer
        
        # 3. Display & Save
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