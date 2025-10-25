import os
import json
from dotenv import load_dotenv
from groq import Groq
from typing import List

load_dotenv()

# ------------------------------
# Simple chunking function
# ------------------------------
def chunk_html_content(html_content: str, max_chunk_size: int = 8000) -> List[str]:
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

# ------------------------------
# Combine processed chunks
# ------------------------------
def combine_chunks(chunks: List[str]) -> str:
    return "\n\n".join(chunks)

# ------------------------------
# Initialize Groq client
# ------------------------------
client = Groq(api_key=os.getenv("Grok"))

# ------------------------------
# Load test.json
# ------------------------------
with open("manual_crawl_results_1761367644.json", "r", encoding="utf-8") as f:
    data = json.load(f)

scraped_data = data.get("scraped_data", [])
if not scraped_data:
    print("❌ No scraped data found in the file.")
    exit(1)

print(f"📊 Found {len(scraped_data)} pages to process")

# ------------------------------
# Process all pages
# ------------------------------
processed_data = []

for i, page in enumerate(scraped_data, 1):
    url = page.get("url", f"doc_{i}")
    status = page.get("status")
    raw_result = page.get("raw_result", "")

    if status != "success" or not raw_result:
        print(f"⚠️ Skipping {url} (no content or error)")
        page["markdown_content"] = None
        processed_data.append(page)
        continue

    html_content = str(raw_result)
    chunks = chunk_html_content(html_content)
    processed_chunks = []

    for j, chunk in enumerate(chunks, 1):
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": f"Convert this HTML to clean markdown:\n\n{chunk}"}],
            model="llama-3.3-70b-versatile"
        )
        processed_chunks.append(response.choices[0].message.content)
        print(f"✅ {url}: chunk {j}/{len(chunks)} processed")

    markdown_content = combine_chunks(processed_chunks)

    # Save markdown
    filename = f"{i:03d}_{url.replace('https://','').replace('http://','').replace('/','_')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    page["markdown_content"] = markdown_content
    page["markdown_file"] = filename
    processed_data.append(page)

    print(f"💾 {url} saved -> {filename}")

# ------------------------------
# Save final processed JSON
# ------------------------------
with open("test_processed.json", "w", encoding="utf-8") as f:
    json.dump({"scraped_data": processed_data}, f, indent=2, ensure_ascii=False)

print("✅ Processing complete, results saved to test_processed.json")
