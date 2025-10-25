"""
Manual URL Crawler
Extracts all URLs from a documentation page, then scrapes each URL individually.
"""

from brightdata import bdclient
import json
import re
import time
import os
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_all_urls_from_page(page_url: str) -> List[str]:
    """
    Extract all URLs from a single page.
    
    Args:
        page_url: The documentation page to extract URLs from
        
    Returns:
        List of all URLs found on the page
    """
    print(f"🔍 Extracting URLs from: {page_url}")
    
    # Initialize Bright Data client
    api_token = os.getenv('BRIGHT_DATA_API_TOKEN')
    if not api_token:
        raise ValueError("BRIGHT_DATA_API_TOKEN not found in environment variables. Please check your .env file.")
    
    client = bdclient(api_token=api_token)
    
    try:
        # Scrape the page to get its content
        result = client.scrape(
            url=page_url,
            data_format="raw",  # Get raw HTML for better link extraction
            country="US",
            timeout=30
        )
        
        if not result:
            print("❌ No content received from the page")
            return []
        
        # Extract URLs from the content
        urls = extract_urls_from_html(str(result), page_url)
        
        print(f"✅ Found {len(urls)} URLs on the page")
        return urls
        
    except Exception as e:
        print(f"❌ Error extracting URLs: {str(e)}")
        return []

def extract_urls_from_html(html_content: str, base_url: str) -> List[str]:
    """Extract all URLs from HTML content, focusing on documentation links."""
    if not html_content:
        return []
    
    # Parse the base URL to get the domain
    base_parsed = urlparse(base_url)
    base_domain = base_parsed.netloc
    
    # Find all href attributes (these are the main navigation links)
    href_pattern = r'href=["\']([^"\']+)["\']'
    href_links = re.findall(href_pattern, html_content, re.IGNORECASE)
    
    # Convert relative URLs to absolute URLs
    absolute_urls = []
    for link in href_links:
        try:
            absolute_link = urljoin(base_url, link)
            absolute_urls.append(absolute_link)
        except:
            continue
    
    # Filter URLs to only include documentation pages
    documentation_urls = []
    for url in absolute_urls:
        try:
            parsed = urlparse(url)
            # Only include URLs from the same domain
            if parsed.netloc == base_domain:
                # Filter out non-documentation URLs
                if not is_non_documentation_url(url):
                    documentation_urls.append(url)
        except:
            continue
    
    # Remove duplicates
    unique_urls = list(set(documentation_urls))
    
    return unique_urls

def is_non_documentation_url(url: str) -> bool:
    """Check if URL is not a documentation page (CSS, JS, images, etc.)."""
    # Common non-documentation URL patterns
    non_doc_patterns = [
        r'\.css$',
        r'\.js$',
        r'\.svg$',
        r'\.png$',
        r'\.jpg$',
        r'\.jpeg$',
        r'\.gif$',
        r'\.ico$',
        r'\.woff2?$',
        r'\.ttf$',
        r'\.eot$',
        r'fonts\.googleapis\.com',
        r'fonts\.gstatic\.com',
        r'fontawesome\.com',
        r'github\.com',
        r'#__codelineno-',  # Code line references
        r'#__codelineno-',  # Code line references
    ]
    
    for pattern in non_doc_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    
    return False

def is_valid_url(url: str) -> bool:
    """Check if URL is valid."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ['http', 'https'] and parsed.netloc
    except:
        return False

def filter_documentation_urls(urls: List[str], base_url: str) -> List[str]:
    """Filter and prioritize documentation URLs."""
    base_parsed = urlparse(base_url)
    base_domain = base_parsed.netloc
    
    # Categorize URLs
    documentation_pages = []
    other_pages = []
    
    for url in urls:
        try:
            parsed = urlparse(url)
            
            # Only include URLs from the same domain
            if parsed.netloc != base_domain:
                continue
            
            # Skip non-documentation URLs
            if is_non_documentation_url(url):
                continue
            
            # Categorize based on path
            path = parsed.path.lower()
            
            # Prioritize main documentation pages
            if (path == '/' or 
                path.endswith('/') or 
                not path.endswith(('.html', '.htm', '.php', '.asp', '.jsp')) or
                'ref/' in path or
                'docs/' in path or
                'guide/' in path or
                'tutorial/' in path or
                'api/' in path):
                documentation_pages.append(url)
            else:
                other_pages.append(url)
                
        except:
            continue
    
    # Sort documentation pages by priority
    def url_priority(url):
        path = urlparse(url).path.lower()
        if path == '/':
            return 0  # Homepage first
        elif 'quickstart' in path:
            return 1  # Quickstart guides
        elif 'guide' in path or 'tutorial' in path:
            return 2  # Guides and tutorials
        elif 'ref/' in path or 'api/' in path:
            return 3  # API reference
        else:
            return 4  # Other pages
    
    documentation_pages.sort(key=url_priority)
    
    # Combine prioritized lists
    return documentation_pages + other_pages

def scrape_single_url(url: str) -> Dict[str, Any]:
    """
    Scrape a single URL using Bright Data API.
    
    Args:
        url: The URL to scrape
        
    Returns:
        Dictionary with scraped data
    """
    print(f"🔍 Scraping: {url}")
    
    # Initialize Bright Data client
    api_token = os.getenv('BRIGHT_DATA_API_TOKEN')
    if not api_token:
        raise ValueError("BRIGHT_DATA_API_TOKEN not found in environment variables. Please check your .env file.")
    
    client = bdclient(api_token=api_token)
    
    try:
        # Scrape the URL
        result = client.scrape(
            url=url,
            data_format="markdown",
            country="US",
            timeout=30
        )
        
        # Parse content
        parsed_content = client.parse_content(result) if result else ""
        
        return {
            "url": url,
            "content": parsed_content,
            "raw_result": result,
            "status": "success",
            "content_length": len(parsed_content),
            "timestamp": time.time()
        }
        
    except Exception as e:
        print(f"❌ Error scraping {url}: {str(e)}")
        return {
            "url": url,
            "content": "",
            "raw_result": None,
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

def crawl_manual_urls(documentation_url: str, max_urls: int = 50, max_workers: int = 5) -> List[Dict[str, Any]]:
    """
    Extract URLs from documentation page and scrape each one concurrently.
    
    Args:
        documentation_url: The documentation page to extract URLs from
        max_urls: Maximum number of URLs to scrape
        max_workers: Maximum number of concurrent workers (default: 5)
        
    Returns:
        List of scraped data for each URL
    """
    print(f"📚 Starting manual URL extraction from: {documentation_url}")
    print(f"📊 Max URLs to scrape: {max_urls}")
    
    # Step 1: Extract all URLs from the documentation page
    all_urls = extract_all_urls_from_page(documentation_url)
    
    if not all_urls:
        print("❌ No URLs found on the documentation page")
        return []
    
    # Filter and prioritize documentation URLs
    documentation_urls = filter_documentation_urls(all_urls, documentation_url)
    
    # Limit the number of URLs to scrape
    urls_to_scrape = documentation_urls[:max_urls]
    
    print(f"\n📊 URL Analysis:")
    print(f"  - Total URLs found: {len(all_urls)}")
    print(f"  - Documentation URLs: {len(documentation_urls)}")
    print(f"  - URLs to scrape: {len(urls_to_scrape)}")
    
    print(f"\n📋 URLs to scrape ({len(urls_to_scrape)}):")
    for i, url in enumerate(urls_to_scrape, 1):
        print(f"  {i}. {url}")
    
    print(f"\n🚀 Starting concurrent scraping of {len(urls_to_scrape)} URLs...")
    
    # Step 2: Scrape URLs concurrently
    scraped_data = []
    print(f"🔧 Using {max_workers} concurrent workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scraping tasks
        future_to_url = {
            executor.submit(scrape_single_url, url): url 
            for url in urls_to_scrape
        }
        
        # Process completed tasks
        completed_count = 0
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            completed_count += 1
            
            try:
                result = future.result()
                scraped_data.append(result)
                
                status_icon = "✅" if result["status"] == "success" else "❌"
                print(f"{status_icon} [{completed_count}/{len(urls_to_scrape)}] {url}")
                
            except Exception as e:
                print(f"❌ [{completed_count}/{len(urls_to_scrape)}] {url} - Exception: {str(e)}")
                scraped_data.append({
                    "url": url,
                    "content": "",
                    "raw_result": None,
                    "status": "error",
                    "error": f"Future exception: {str(e)}",
                    "timestamp": time.time()
                })
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.5)
    
    # Step 3: Save results
    timestamp = int(time.time())
    filename = f"manual_crawl_results_{timestamp}.json"
    
    summary = {
        "crawl_info": {
            "documentation_url": documentation_url,
            "total_urls_found": len(all_urls),
            "urls_scraped": len(scraped_data),
            "successful_scrapes": len([r for r in scraped_data if r["status"] == "success"]),
            "failed_scrapes": len([r for r in scraped_data if r["status"] == "error"]),
            "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "all_urls_found": all_urls,
        "scraped_data": scraped_data
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Manual crawling completed!")
    print(f"📊 Results:")
    print(f"  - URLs found: {len(all_urls)}")
    print(f"  - URLs scraped: {len(scraped_data)}")
    print(f"  - Successful: {len([r for r in scraped_data if r['status'] == 'success'])}")
    print(f"  - Failed: {len([r for r in scraped_data if r['status'] == 'error'])}")
    print(f"💾 Results saved to: {filename}")
    
    return scraped_data

def main():
    """Example usage."""
    print("🕷️ Manual URL Crawler")
    print("=" * 50)
    
    # Example: Extract URLs from Bright Data documentation
    documentation_url = "https://docs.streamlit.io/develop/api-reference"
    
    # Or use any other documentation page
    # documentation_url = "https://your-documentation-site.com"
    
    results = crawl_manual_urls(
        documentation_url=documentation_url,
        max_urls=200,  # Adjust as needed
        max_workers=200  # Concurrent workers (adjust based on API limits)
    )
    
    print(f"\n🎉 Manual crawling complete!")
    print(f"Check the generated JSON file for all scraped content.")

if __name__ == "__main__":
    main()
