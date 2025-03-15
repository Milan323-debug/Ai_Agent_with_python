import os
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel
import google.generativeai as genai
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from time import sleep
from duckduckgo_search import DDGS

# Load environment variables from .env file
load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: List[str]
    tools_used: List[str]

# Get the Gemini API key from the environment variables
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Configure Gemini
genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel('models/gemini-1.5-pro')

def search_web_with_retry(query: str, max_retries: int = 3) -> Optional[str]:
    """
    Search the web using DuckDuckGo with retry mechanism
    """
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                if results:
                    return "\n".join(result['body'] for result in results)
        except Exception as e:
            print(f"Search attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                sleep(2)  # Wait before retrying
            continue
    return None

def generate_base_content(topic: str) -> str:
    """
    Generate base content when search fails
    """
    base_prompt = f"""As an expert researcher, create a comprehensive analysis of the following topic:

    Topic: {topic}

    Please include:
    1. Introduction and background
    2. Key components and technologies involved
    3. Current state of research
    4. Potential applications and benefits
    5. Challenges and future directions

    Focus on providing accurate, technical information."""

    response = model.generate_content(base_prompt)
    return response.text

def research_topic(query: str) -> ResearchResponse:
    """
    Conduct research on a given topic using Gemini and web search.
    """
    try:
        # Try to search the web first
        search_results = search_web_with_retry(query)
        tools_used = ["web_search", "gemini-pro"]
        
        if search_results:
            # If search was successful, use both search results and AI
            prompt = f"""As a research assistant, analyze this topic and the search results to create a comprehensive summary.
            
            Topic: {query}
            
            Search Results:
            {search_results}
            
            Please provide a well-structured summary of the topic based on these search results."""
        else:
            # If search failed, use AI's knowledge only
            print("Web search failed. Using AI knowledge base only.")
            search_results = generate_base_content(query)
            tools_used = ["gemini-pro"]
            prompt = f"""As a research assistant, create a comprehensive summary of this topic:
            
            Topic: {query}
            
            Please provide a well-structured analysis focusing on technical details and current research."""
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        
        # Process the response into the required format
        research_data = ResearchResponse(
            topic=query,
            summary=response.text,
            sources=["AI Knowledge Base"] if "web_search" not in tools_used else ["Search results from DuckDuckGo"],
            tools_used=tools_used
        )
        
        return research_data
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    query = input("Please enter the topic you want to research: ")
    result = research_topic(query)
    
    # Save the results to a text file
    with open("research_results.txt", "w") as file:
        file.write("Research Results:\n")
        file.write(f"Topic: {result.topic}\n\n")
        file.write("Summary:\n")
        summary_lines = result.summary.split('. ')
        for line in summary_lines:
            file.write(f"{line.strip()}.\n")
        file.write("\nSources: {result.sources}\n")
        file.write(f"Tools Used: {result.tools_used}\n")
    
    print("\nResearch results have been saved to research_results.txt")