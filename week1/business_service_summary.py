import os
import requests
import json
from typing import List
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import datetime

# Define the model to use
MODEL = "gpt-3.5-turbo"  # or another specific model you want to use


class Website:
    url: str
    title: str
    body: str
    links: List[str]

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            # Remove irrelevant elements
            for irrelevant in soup.body(
                ["script", "style", "img", "input", "iframe", "noscript"]
            ):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        # Get all links
        links = [link.get("href") for link in soup.find_all("a")]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"


def get_relevant_links(website: Website) -> dict:
    """Get relevant links for business services from the website."""

    # Initialize OpenAI
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv(
        "OPENAI_API_KEY", "your-key-if-not-using-env"
    )
    client = OpenAI()

    # Prepare the prompt for finding relevant links
    system_prompt = """You are provided with a list of links found on a webpage.
    You are able to decide which of the links would be most relevant to understand the company's business services,
    such as links to Services page, thought-leadership page, home page, clients page, blog post page.
    You should respond in JSON as in this example:
    {
        "links": [
            {"type": "services page", "url": "https://www.ibp2.com/our-services"},
            {"type": "thought-leadership page", "url": "https://www.ibp2.com/thought-leadership"},
            {"type": "home page", "url": "https://www.ibp2.com/home"},
            {"type": "clients page", "url": "https://www.ibp2.com/clients"},
            {"type": "blog post page", "url": "https://www.ibp2.com/post/forecast-accuracy-for-free"}
        ]
    }
    Only respond with the JSON, nothing else.
    """

    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for understanding the company's business services. "
    user_prompt += "Respond with the full https URL in JSON format.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)

    # Get relevant links using OpenAI
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    try:
        return json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {completion.choices[0].message.content}")
        return {"links": []}


def get_service_details(url: str) -> str:
    """Get details from all relevant pages."""
    result = "Landing page:\n"
    website = Website(url)
    result += website.get_contents()

    # Get relevant links
    links = get_relevant_links(website)
    print("Found relevant service pages:", links)

    # Get content from each relevant page
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        try:
            result += Website(link["url"]).get_contents()
        except Exception as e:
            print(f"Error accessing {link['url']}: {str(e)}")
            result += f"Could not access page: {str(e)}\n"

    return result


def create_service_summary(company_name: str, url: str) -> str:
    """Create a summary of the company's business services."""

    # Initialize OpenAI
    load_dotenv()
    client = OpenAI()

    # System prompt for service summary
    system_prompt = """You are a business analyst that analyzes the contents of several relevant pages 
    from a company website and creates a comprehensive summary of their business services and offerings.
    Focus on:
    1. Main services and products
    2. Key industries served
    3. Unique value propositions
    4. Service delivery approach
    5. Client benefits
    
    Format the response in markdown with clear sections and bullet points."""

    # Get website content and create user prompt
    user_prompt = f"You are analyzing a company called: {company_name}\n"
    user_prompt += (
        "Here are the contents of its landing page and other relevant service pages. "
    )
    user_prompt += "Please create a comprehensive summary of their business services.\n"
    user_prompt += get_service_details(url)

    # Truncate if more than 40,000 characters
    user_prompt = user_prompt[:40_000]

    # Generate summary using OpenAI
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content


def main():
    # Example usage
    company_name = input("Enter company name: ")
    company_url = input("Enter company website URL: ")

    try:
        summary = create_service_summary(company_name, company_url)

        # Create markdown content with metadata
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        markdown_content = f"""---
            title: Business Service Summary - {company_name}
            company_url: {company_url}
            date_generated: {current_date}
            model_used: {MODEL}
            ---

            {summary}

            ---
            *This summary was automatically generated using AI analysis of the company website.*
        """

        # Save to markdown file
        filename = f"{company_name.lower().replace(' ', '_')}_service_summary.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"\nBusiness Service Summary has been saved to: {filename}")
        print("\nSummary Preview:\n")
        print(summary[:500] + "..." if len(summary) > 500 else summary)

    except Exception as e:
        print(f"Error generating summary: {str(e)}")


if __name__ == "__main__":
    main()
