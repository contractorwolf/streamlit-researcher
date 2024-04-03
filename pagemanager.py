import asyncio
import logging
import re
import urllib.parse
from playwright.async_api import async_playwright
from searchmanager import SearchManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
BATCH_SIZE = 8
MAX_ITERATIONS = 5

combined_page_contents = ""
        
class PageManager:       
    @staticmethod
    async def get_search_content_async(query, st):
        global combined_page_contents
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                links = await SearchManager.search_async(browser, query)
                
                st.session_state['search_logs'].append(f"found: {links}")
                
                print("Researching using the following links:")
                
                st.session_state['search_logs'].append("Reasearching using the following links:")
                
                combined_page_contents = ""
                
                iterations = min(MAX_ITERATIONS, len(links))  # Use 5 or the number of links, whichever is smaller

                
                for i in range(0, iterations, BATCH_SIZE):
                    print(f"Processing batch {i // BATCH_SIZE + 1} of {len(links) // BATCH_SIZE + 1}")
                    batch = links[i:i + BATCH_SIZE]

                    # Create a list of tasks, transforming each URL into a filename
                    tasks = []
                    for link in batch:
                        st.session_state['search_logs'].append(f"consuming: {link}")
                        tasks.append(PageManager.store_content_async(link, browser))

                    # Gather and run tasks concurrently
                    await asyncio.gather(*tasks)
                        
            return combined_page_contents
        
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None
    
    @staticmethod
    async def store_content_async(link, browser):
        global combined_page_contents
        content = await PageManager.get_content_async(link, browser)
        
        if (content):
            combined_page_contents += content
        print(f"Content retrieved for: {link}")

    @staticmethod
    async def get_content_async(url, browser):
        try:
            # Check if the URL is valid
            if not urllib.parse.urlparse(url).scheme:
                print(f"Invalid URL: {url}")
                return None

            # Create a new browser context with the specified User-Agent
            context = await browser.new_context(user_agent=USER_AGENT)

            # Use the context to create a new page
            page = await context.new_page()

            # Set a generous timeout for navigation and loading of content
            navigation_timeout = 60000  # 60 seconds
            page.set_default_timeout(navigation_timeout)
            # page.set_default_navigation_timeout(navigation_timeout)
            logging.info(f"Navigating to {url}")

            # Navigate to the URL with a specific timeout
            await page.goto(url, wait_until='domcontentloaded')

            all_text = await page.evaluate("document.body.innerText")

            # Clean up the text by removing extra whitespace
            cleaned_text = re.sub(r'\s+', ' ', all_text).strip()
            await page.close()
            return cleaned_text

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None
        
        finally:
            if page: 
                await page.close()
            if context:
                await context.close()

