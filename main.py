import os
from dotenv import load_dotenv
from typing import List, Optional, Tuple
from pydantic import BaseModel
import google.generativeai as genai
from datetime import datetime
import pytz
import textwrap
import requests
from duckduckgo_search import DDGS
from time import sleep

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

def search_web(query: str, max_retries: int = 3) -> Optional[List[dict]]:
    """
    Search the web using DuckDuckGo with retry mechanism
    """
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                if results:
                    return results
        except Exception as e:
            print(f"Search attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                sleep(2)  # Wait before retrying
            continue
    return None

def get_timezone_info(city: str) -> Tuple[Optional[str], Optional[str]]:
    """Get timezone for a given city"""
    city_timezone_mapping = {
        'new york': 'America/New_York',
        'london': 'Europe/London',
        'tokyo': 'Asia/Tokyo',
        'paris': 'Europe/Paris',
        'sydney': 'Australia/Sydney',
        'dubai': 'Asia/Dubai',
        'singapore': 'Asia/Singapore',
        'hong kong': 'Asia/Hong_Kong',
        'moscow': 'Europe/Moscow',
        'berlin': 'Europe/Berlin',
        'india': 'Asia/Kolkata',
        'mumbai': 'Asia/Kolkata',
        'delhi': 'Asia/Kolkata',
        'bangalore': 'Asia/Kolkata',
        'los angeles': 'America/Los_Angeles',
        'chicago': 'America/Chicago',
        'toronto': 'America/Toronto',
        'vancouver': 'America/Vancouver',
        'beijing': 'Asia/Shanghai',
        'shanghai': 'Asia/Shanghai'
    }
    
    # Try to find timezone from mapping
    city_lower = city.lower().strip()
    timezone_str = city_timezone_mapping.get(city_lower)
    
    if timezone_str:
        try:
            tz = pytz.timezone(timezone_str)
            return timezone_str, tz
        except pytz.exceptions.PyTzError:
            return None, None
    return None, None

def get_city_time(city: str) -> Tuple[str, List[str], List[str]]:
    """Get current time for a given city with additional information"""
    timezone_str, tz = get_timezone_info(city)
    sources = []
    tools_used = ["datetime"]
    
    if not timezone_str:
        return f"Sorry, I don't have timezone information for {city}. Please try a major city.", sources, tools_used
    
    try:
        current_time = datetime.now(tz)
        time_str = current_time.strftime("%d %B %Y, %I:%M:%S %p %Z")
        
        # Try to get additional time zone information from web
        search_results = search_web(f"current time zone {city} UTC offset DST")
        if search_results:
            tools_used.append("web_search")
            sources.extend([result.get('link', 'Web Search') for result in search_results[:2]])
            
            # Create a concise response with current time and key information
            response = f"Current time in {city.title()}: {time_str}\n\n"
            response += "Additional Information:\n"
            response += f"• Timezone: {timezone_str}\n"
            response += f"• UTC Offset: {current_time.strftime('%z')}\n"
            response += f"• DST Status: {'Observing DST' if current_time.dst() else 'Not observing DST'}"
            
            return response, sources, tools_used
        else:
            # Basic response with just the time if web search fails
            return f"Current time in {city.title()}: {time_str}", ["System Clock"], tools_used
            
    except Exception as e:
        return f"Error getting time for {city}: {str(e)}", ["Error Log"], ["error_handler"]

def is_time_query(query: str) -> Tuple[bool, Optional[str]]:
    """Check if the query is asking for current time and extract city name"""
    time_keywords = ['time', 'current time', 'what time']
    query_lower = query.lower()
    
    # Check if it's a time query
    if any(keyword in query_lower for keyword in time_keywords):
        # Try to extract city name
        words = query_lower.split()
        for i, word in enumerate(words):
            if word in ['in', 'at', 'for']:
                if i + 1 < len(words):
                    # Handle multi-word city names
                    city = ' '.join(words[i+1:]).strip('?.,!')
                    return True, city
        return True, None
    return False, None

def wrap_text(text: str, width: int = 80) -> str:
    """Wrap text to specified width"""
    wrapped_lines = []
    for line in text.split('\n'):
        if line.strip().startswith('•'):
            indentation = len(line) - len(line.lstrip())
            wrapped = textwrap.fill(line.strip(), 
                                  width=width-indentation, 
                                  initial_indent=' '*indentation,
                                  subsequent_indent=' '*(indentation+2))
        else:
            wrapped = textwrap.fill(line.strip(), width=width)
        wrapped_lines.append(wrapped)
    return '\n'.join(wrapped_lines)

def format_summary(summary: str) -> str:
    """Format the summary text for better readability"""
    # Remove asterisks while preserving the text
    summary = summary.replace('*', '')
    
    paragraphs = summary.split('\n\n')
    formatted_paragraphs = []
    
    for para in paragraphs:
        if para.strip().startswith('•'):
            lines = para.split('\n')
            formatted_lines = []
            for line in lines:
                if line.strip():
                    # Clean up any remaining asterisks and extra spaces
                    cleaned_line = line.strip().replace('*', '').strip()
                    if cleaned_line.startswith('•'):
                        formatted_lines.append('  ' + cleaned_line)
                    else:
                        formatted_lines.append('  • ' + cleaned_line)
            formatted_paragraphs.append('\n'.join(formatted_lines))
        else:
            formatted_paragraphs.append(wrap_text(para.strip()))
    
    return '\n\n'.join(formatted_paragraphs)

def save_to_file(result: ResearchResponse):
    """Save research results to a file with proper formatting"""
    with open("research_results.txt", "w", encoding='utf-8') as file:
        header_line = "=" * 80
        file.write(f"{header_line}\n")
        file.write(f"{'RESEARCH RESULTS':^80}\n")
        file.write(f"{header_line}\n\n")
        
        file.write("TOPIC:\n")
        file.write("-" * 80 + "\n")
        file.write(wrap_text(result.topic, width=80) + "\n\n")
        
        file.write("SUMMARY:\n")
        file.write("-" * 80 + "\n")
        formatted_summary = format_summary(result.summary)
        file.write(formatted_summary + "\n\n")
        
        if result.sources:
            file.write("SOURCES:\n")
            file.write("-" * 80 + "\n")
            for source in result.sources:
                file.write(f"• {source}\n")
            file.write("\n")
        
        if result.tools_used:
            file.write("TOOLS USED:\n")
            file.write("-" * 80 + "\n")
            for tool in result.tools_used:
                file.write(f"• {tool}\n")
        
        file.write("\n" + "=" * 80 + "\n")
        timestamp = datetime.now().strftime('%d %B %Y, %I:%M:%S %p')
        file.write(f"Generated on: {timestamp}\n")
        file.write("=" * 80 + "\n")

def research_topic(query: str) -> ResearchResponse:
    """
    Conduct research on a given topic using both web search and Gemini.
    """
    try:
        # Check if it's a time query
        is_time, city = is_time_query(query)
        if is_time and city:
            time_info, sources, tools_used = get_city_time(city)
            return ResearchResponse(
                topic=query,
                summary=time_info,
                sources=sources,
                tools_used=tools_used
            )

        # For non-time queries, try web search first
        print("Searching the web for information...")
        search_results = search_web(query)
        tools_used = ["gemini-pro"]
        sources = ["AI Knowledge Base"]
        
        if search_results:
            tools_used.append("web_search")
            sources.extend([result.get('link', 'Web Search') for result in search_results])
            web_content = "\n\n".join(result.get('body', '') for result in search_results)
            
            prompt = f"""Analyze this topic and create a concise, factual summary using the search results:

            Topic: {query}

            Web Search Results:
            {web_content}

            Please provide:
            1. Brief introduction
            2. Key facts and current information
            3. Important developments
            4. Practical implications
            5. Brief conclusion

            Format with:
            - Short, focused paragraphs
            - Use bullet points (•) for listing facts and key points
            - Important terms should be mentioned naturally without special formatting
            - Use clear section headings
            
            Focus on factual, up-to-date information without speculation."""
        else:
            print("Web search unsuccessful. Using AI knowledge base only...")
            prompt = f"""Create a concise, factual analysis of this topic:

            Topic: {query}

            Include:
            1. Brief introduction
            2. Key facts and analysis
            3. Current state
            4. Practical applications
            5. Brief conclusion

            Format with:
            - Short, focused paragraphs
            - Use bullet points (•) for listing facts and key points
            - Important terms should be mentioned naturally without special formatting
            - Use clear section headings

            Focus on factual information."""

        response = model.generate_content(prompt)
        
        return ResearchResponse(
            topic=query,
            summary=response.text,
            sources=sources,
            tools_used=tools_used
        )

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return ResearchResponse(
            topic=query,
            summary=f"Sorry, an error occurred while processing your request: {str(e)}",
            sources=["Error Log"],
            tools_used=["error_handler"]
        )

if __name__ == "__main__":
    try:
        query = input("Please enter the topic you want to research: ")
        print("\nResearching your topic. This may take a moment...")
        result = research_topic(query)
        
        save_to_file(result)
        
        print("\nResearch Results:")
        print(f"Topic: {result.topic}")
        print("\nSummary:")
        print(result.summary[:500] + "..." if len(result.summary) > 500 else result.summary)
        print(f"\nSources: {', '.join(result.sources)}")
        print(f"Tools Used: {', '.join(result.tools_used)}")
        print("\nFull results have been saved to research_results.txt")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")