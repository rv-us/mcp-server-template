"""
Full Documentation Pipeline
Simple pipeline: Scrape → Markdown (Groq) → Embed (ChromaDB) → Query
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from typing import List
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import our modules
from manual_url_crawler import crawl_manual_urls
from vector_db import ChromaVectorDB

# Load environment variables
load_dotenv()


def chunk_html_content(html_content: str, max_chunk_size: int = 8000) -> List[str]:
    """Simple chunking function for HTML content."""
    import re
    if len(html_content) <= max_chunk_size:
        return [html_content]

    # Detect if HTML
    is_html = bool(re.search(r'<[^>]+>', html_content))
    pattern = r'(<h[1-3][^>]*>.*?</h[1-3]>)' if is_html else r'(^#{1,3}\s+.*$)'
    parts = re.split(pattern, html_content, flags=re.IGNORECASE|re.MULTILINE|re.DOTALL)

    chunks, current = [], ""
    for part in parts:
        if len(current + part) <= max_chunk_size:
            current += part
        else:
            if current:
                chunks.append(current.strip())
            current = part
    if current:
        chunks.append(current.strip())
    return chunks


def combine_chunks(chunks: List[str]) -> str:
    """Combine processed chunks into single markdown."""
    return "\n\n".join(chunks)


def process_single_document(page, groq_client, vector_db, output_dir, doc_index, enable_vector_db):
    """Process a single document with Groq and optionally embed it."""
    url = page.get("url", f"doc_{doc_index}")
    status = page.get("status")
    raw_result = page.get("raw_result", "")
    
    print(f"📄 Processing {doc_index}: {url}")
    
    if status != "success" or not raw_result:
        print(f"⚠️ Skipping {doc_index} (no content or error)")
        page["markdown_content"] = None
        page["markdown_file"] = None
        return page, False
    
    try:
        # Chunk the HTML content
        html_content = str(raw_result)
        chunks = chunk_html_content(html_content)
        print(f"   Split into {len(chunks)} chunks")
        
        # Process each chunk with Groq
        processed_chunks = []
        for j, chunk in enumerate(chunks, 1):
            print(f"   Processing chunk {j}/{len(chunks)}...", end=" ")
            try:
                response = groq_client.chat.completions.create(
                    messages=[{
                        "role": "user", 
                        "content": f"Convert this HTML documentation to clean, well-structured markdown. Preserve code blocks, headings, and technical details:\n\n{chunk}"
                    }],
                    model="llama-3.3-70b-versatile"
                )
                processed_chunks.append(response.choices[0].message.content)
                print("✅")
            except Exception as chunk_error:
                print(f"❌ Chunk {j} failed: {str(chunk_error)}")
                processed_chunks.append(f"# Error Processing Chunk {j}\n\n{str(chunk_error)}")
        
        # Combine chunks
        markdown_content = combine_chunks(processed_chunks)
        
        # Save markdown file with safe filename
        # Remove protocol and sanitize URL for filename
        url_safe = url.replace('https://', '').replace('http://', '')
        # Remove or replace invalid Windows filename characters
        url_safe = url_safe.replace('/', '_').replace('?', '_').replace('#', '_').replace(':', '_')
        # Remove multiple consecutive underscores
        while '__' in url_safe:
            url_safe = url_safe.replace('__', '_')
        # Trim trailing underscores
        url_safe = url_safe.rstrip('_')
        filename = f"{doc_index:03d}_{url_safe}.md"
        filepath = output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        page["markdown_content"] = markdown_content
        page["markdown_file"] = str(filepath)
        
        print(f"   💾 Saved: {filepath}")
        
        # Embed into vector database (if enabled)
        if enable_vector_db and vector_db:
            print(f"   🔄 Embedding into vector database...", end=" ")
            try:
                doc_id = f"doc_{doc_index}_{url.split('/')[-1]}"
                vector_db.insert(
                    documents=[markdown_content],
                    ids=[doc_id]
                )
                print("✅")
            except Exception as e:
                print(f"❌ Embedding failed: {str(e)}")
        
        return page, True
        
    except Exception as e:
        print(f"   ❌ Error processing {doc_index}: {str(e)}")
        import traceback
        traceback.print_exc()
        page["markdown_content"] = None
        page["markdown_file"] = None
        page["error"] = str(e)
        return page, False


def run_full_pipeline(documentation_url: str, max_urls: int = 20, 
                      crawler_workers: int = 50,
                      enable_vector_db: bool = True, 
                      collection_name: str = None):
    """
    Run the complete documentation pipeline.
    
    Args:
        documentation_url: The documentation site to crawl
        max_urls: Maximum number of URLs to process
        crawler_workers: Number of concurrent workers for web scraping (default: 50)
        enable_vector_db: Enable ChromaDB integration for embeddings
        collection_name: Name for the ChromaDB collection (auto-generated if None)
    """
    print("🚀 Full Documentation Pipeline (Groq + ChromaDB)")
    print("=" * 60)
    print(f"📚 Target: {documentation_url}")
    print(f"📊 Max URLs: {max_urls}")
    print(f"🕷️ Crawler Workers: {crawler_workers}")
    print(f"🔍 Vector DB: {'Enabled' if enable_vector_db else 'Disabled'}")
    print()
    
    # Auto-generate collection name from URL if not provided
    if collection_name is None and enable_vector_db:
        parsed = urlparse(documentation_url)
        domain = parsed.netloc.replace('www.', '').replace('.', '_')
        collection_name = f"docs_{domain}"
        print(f"📚 Collection name: {collection_name}")
        print()
    
    # Initialize Groq client
    groq_api_key = os.getenv("Grok")
    if not groq_api_key:
        print("❌ Grok API key not found in environment variables.")
        return
    
    groq_client = Groq(api_key=groq_api_key)
    print("Groq client initialized")
    
    # Initialize Vector DB
    vector_db = None
    if enable_vector_db:
        try:
            vector_db = ChromaVectorDB(
                path="./chroma_db",
                collection_name=collection_name,
                chunk_size=500,  # Words per chunk
                chunk_overlap=50  # Overlapping words
            )
            print(f"Vector database initialized (Qwen 0.6B embeddings)")
            print(f"   Collection: {collection_name}")
            print(f"   Chunk size: 500 words with 50 word overlap")
            
            # Test model loading explicitly
            print(f"   Testing model loading...")
            test_embedding = vector_db.get_text_embedding(["test"])
            print(f"   Model loaded successfully - embedding shape: {test_embedding.shape}")
        except Exception as e:
            print(f"⚠️ Could not initialize vector database: {str(e)}")
            enable_vector_db = False
    print()
    
    # Step 1: Crawl the documentation site
    print("🕷️ STEP 1: Crawling documentation site...")
    print("-" * 60)
    
    scraped_data = crawl_manual_urls(
        documentation_url=documentation_url,
        max_urls=max_urls,
        max_workers=crawler_workers
    )
    
    if not scraped_data:
        print("❌ No data scraped. Exiting.")
        return
    
    print(f"✅ Crawled {len(scraped_data)} pages")
    print()
    
    # Step 2: Process with Groq to create markdown (CONCURRENT)
    print("🤖 STEP 2: Converting to Markdown with Groq (CONCURRENT)...")
    print("-" * 60)
    
    output_dir = Path("documentation_markdown")
    output_dir.mkdir(exist_ok=True)
    
    processed_data = []
    successful = 0
    failed = 0
    
    print(f"📊 Processing {len(scraped_data)} documents concurrently...")
    
    # Use ThreadPoolExecutor for concurrent processing
    max_workers = min(5, len(scraped_data))  # Limit concurrent Groq requests
    print(f"🔧 Using {max_workers} concurrent workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {}
        for i, page in enumerate(scraped_data, 1):
            future = executor.submit(
                process_single_document, 
                page, 
                groq_client, 
                vector_db, 
                output_dir, 
                i, 
                enable_vector_db
            )
            future_to_index[future] = i
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_index):
            doc_index = future_to_index[future]
            try:
                page, success = future.result()
                processed_data.append(page)
                if success:
                    successful += 1
                    print(f"✅ Document {doc_index} completed successfully")
                else:
                    failed += 1
                    print(f"❌ Document {doc_index} failed")
            except Exception as e:
                print(f"❌ Document {doc_index} exception: {str(e)}")
                failed += 1
                # Create error page
                error_page = scraped_data[doc_index - 1].copy()
                error_page["markdown_content"] = None
                error_page["markdown_file"] = None
                error_page["error"] = str(e)
                processed_data.append(error_page)
    
    # Sort processed_data by original order (fix the sorting logic)
    # Create a mapping of pages to their original indices
    url_to_index = {page['url']: i for i, page in enumerate(scraped_data)}
    processed_data.sort(key=lambda x: url_to_index.get(x.get('url', ''), 999))
    
    print(f"\n✅ All documents processed!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    
    # Ensure ChromaDB is persisted
    if enable_vector_db and vector_db:
        try:
            print(f"\n💾 Persisting vector database...")
            # Force ChromaDB to persist by getting count
            count = vector_db.collection.count()
            print(f"   Total chunks in DB: {count}")
        except Exception as e:
            print(f"   ⚠️ Warning: Could not verify DB persistence: {str(e)}")
    
    # Step 4: Save results
    print("\n💾 STEP 3: Saving results...")
    print("-" * 60)
    
    timestamp = int(time.time())
    results_file = f"full_pipeline_results_{timestamp}.json"
    
    pipeline_results = {
        "pipeline_info": {
            "documentation_url": documentation_url,
            "max_urls": max_urls,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_pages": len(processed_data),
            "successful_pages": successful,
            "failed_pages": failed,
            "vector_db_enabled": enable_vector_db,
            "collection_name": collection_name if enable_vector_db else None
        },
        "processed_data": processed_data
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
    
    # Final summary
    print("\n🎉 PIPELINE COMPLETED!")
    print("=" * 60)
    print(f"📁 Markdown files: {output_dir}")
    print(f"💾 Full results: {results_file}")
    
    print(f"\n📊 Final Summary:")
    print(f"  - Total pages: {len(processed_data)}")
    print(f"  - Successfully processed: {successful}")
    print(f"  - Failed: {failed}")
    print(f"  - Success rate: {(successful/len(processed_data)*100):.1f}%")
    
    if enable_vector_db and vector_db:
        try:
            count = vector_db.collection.count()
            print(f"\n🔍 Vector Database:")
            print(f"  - Total chunks embedded: {count}")
            print(f"  - Collection: {collection_name}")
            print(f"  - Ready for semantic search!")
        except Exception as e:
            print(f"\n⚠️ Could not retrieve vector DB stats: {str(e)}")
    
    print(f"\n📖 Your documentation is ready in the '{output_dir}' folder!")
    if enable_vector_db:
        print(f"🔎 Use search_docs.py to query the documentation!")


def main():
    """Main function to run the pipeline."""
    try:
        # Configuration
        documentation_url = "https://docs.streamlit.io/develop/api-reference"
        max_urls = 5  # Adjust as needed
        crawler_workers = 200  # Concurrent web scraping
        
        # Check if API keys are available
        groq_key = os.getenv('Grok')
        brightdata_key = os.getenv('BRIGHT_DATA_API_TOKEN')
        
        if not groq_key:
            print("❌ Grok API key not found in environment variables.")
            print("Please add it to your .env file:")
            print("Grok=your_groq_api_key_here")
            return
        
        if not brightdata_key:
            print("❌ BRIGHT_DATA_API_TOKEN not found in environment variables.")
            print("Please add it to your .env file:")
            print("BRIGHT_DATA_API_TOKEN=your_brightdata_api_key_here")
            return
        
        print("API keys found. Starting pipeline...")
        print()
        
        # Run the pipeline
        run_full_pipeline(
            documentation_url=documentation_url,
            max_urls=max_urls,
            crawler_workers=crawler_workers,
            enable_vector_db=True
        )
        
        print("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        exit(exit_code)
