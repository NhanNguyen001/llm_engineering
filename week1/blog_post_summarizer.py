import os
import requests
import json
from typing import List, Dict
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import datetime
from urllib.parse import urljoin

# Define the model to use
MODEL = "gpt-3.5-turbo"

class BlogPost:
    def __init__(self, url: str):
        self.url = url
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an error for bad status codes
            self.soup = BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch the blog post: {str(e)}")
        
        # Clean the content
        self._clean_html()
        
        # Extract content
        self.title = self._get_title()
        self.author = self._get_author()
        self.date = self._get_date()
        self.content = self._get_content()
        self.metadata = self._get_metadata()

    def _clean_html(self):
        """Remove unwanted HTML elements."""
        # Remove scripts, styles, and other irrelevant elements
        for element in self.soup.find_all(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()

    def _get_title(self) -> str:
        """Extract the blog post title."""
        # Try different methods to get the title
        title = self.soup.find('h1')
        if not title:
            title = self.soup.find('title')
        if not title:
            return "No title found"
        return title.get_text().strip()

    def _get_author(self) -> str:
        """Extract the author name."""
        # Try common author selectors
        author_elements = self.soup.find_all(['a', 'span', 'div', 'p'], 
            class_=lambda x: x and any(word in str(x).lower() for word in ['author', 'by', 'writer']))
        for element in author_elements:
            if element.get_text().strip():
                return element.get_text().strip()
        return "Unknown Author"

    def _get_date(self) -> str:
        """Extract the publication date."""
        # Try to find date in meta tags first
        meta_date = self.soup.find('meta', {'property': ['article:published_time', 'og:published_time']})
        if meta_date:
            return meta_date.get('content', "No date found")
        
        # Try common date selectors
        date_elements = self.soup.find_all(['time', 'span', 'div', 'p'],
            class_=lambda x: x and any(word in str(x).lower() for word in ['date', 'published', 'time']))
        for element in date_elements:
            if element.get_text().strip():
                return element.get_text().strip()
        return "No date found"

    def _get_content(self) -> str:
        """Extract the main content of the blog post."""
        # Try to find the main content container
        main_content = self.soup.find(['article', 'main', 'div'], 
            class_=lambda x: x and any(word in str(x).lower() for word in ['content', 'post', 'article', 'entry']))
        
        if main_content:
            # Get all paragraphs from the main content
            paragraphs = main_content.find_all(['p', 'h2', 'h3', 'h4', 'li'])
            return "\n\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
        
        # Fallback: get all paragraphs from the body
        paragraphs = self.soup.find_all(['p', 'h2', 'h3', 'h4', 'li'])
        return "\n\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())

    def _get_metadata(self) -> Dict[str, str]:
        """Extract metadata from meta tags."""
        metadata = {}
        
        # Define meta tag mappings
        meta_mappings = {
            'description': ['description', 'og:description'],
            'keywords': ['keywords'],
            'type': ['og:type'],
            'section': ['article:section']
        }
        
        # Search for meta tags using both name and property attributes
        for key, possible_values in meta_mappings.items():
            for value in possible_values:
                # Try property attribute
                meta = self.soup.find('meta', property=value)
                if not meta:
                    # Try name attribute
                    meta = self.soup.find('meta', attrs={'name': value})
                if meta:
                    metadata[key] = meta.get('content', '')
                    break
        
        return metadata

def summarize_blog_post(url: str) -> str:
    """Summarize a blog post using OpenAI."""
    # Initialize OpenAI
    load_dotenv()
    client = OpenAI()
    
    # Get blog post content
    blog = BlogPost(url)
    
    # Create an optimized system prompt for blog summarization
    system_prompt = """You are an expert content analyst and summarizer. Your task is to create a comprehensive yet concise summary of the blog post.
    Follow these guidelines:

    1. Structure your summary with these sections:
        - Key Takeaways (3-5 bullet points)
        - Main Arguments/Points (clear and concise)
        - Supporting Evidence/Examples
        - Practical Applications/Implications
        - Target Audience

    2. Focus on:
        - The core message and main arguments
        - Unique insights or perspectives
        - Practical implications or actionable takeaways
        - Evidence and examples used to support the arguments
        - Any notable quotes or statistics

    3. Format Guidelines:
        - Use markdown formatting
        - Keep paragraphs short and focused
        - Use bullet points for lists
        - Include relevant quotes in blockquotes
        - Highlight key terms in bold

    4. Maintain the original author's tone while being objective in the summary.
    """

    # Create an optimized user prompt
    user_prompt = f"""Please analyze and summarize this blog post:

    Title: {blog.title}
    Author: {blog.author}
    Date: {blog.date}

    Content:
    {blog.content[:4000]}  # Limit content to avoid token limits

    Focus on extracting the most valuable insights and practical takeaways for readers.
    """

    # Generate summary using OpenAI
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    summary = response.choices[0].message.content

    # Create markdown content with metadata
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    markdown_content = f"""---
title: "{blog.title}"
original_url: {url}
author: {blog.author}
date_published: {blog.date}
date_summarized: {current_date}
summarizer_model: {MODEL}
---

# Blog Post Summary: {blog.title}

{summary}

---
*This summary was automatically generated using AI analysis of the original blog post.*
"""

    return markdown_content

def main():
    # Get blog post URL from user
    blog_url = input("Enter the blog post URL: ")
    
    try:
        # Generate summary
        summary = summarize_blog_post(blog_url)
        
        # Create filename from blog title or URL
        filename = f"blog_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # Save to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(summary)
        
        print(f"\nBlog post summary has been saved to: {filename}")
        print("\nSummary Preview:\n")
        print(summary[:500] + "..." if len(summary) > 500 else summary)
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")

if __name__ == "__main__":
    main() 