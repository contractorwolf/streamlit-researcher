import logging

PAGE_TIMEOUT = 60000  # 60 seconds
SEARCH_URL = "https://www.google.com/search?q="
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
   
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')   
   
class SearchManager:
    @staticmethod
    async def get_links_with_jsname_async(page):
        # Select all <a> elements that have a 'jsname' attribute
        links = await page.query_selector_all('a[jsname]')

        # Extract the href attribute from each link and apply the filter
        hrefs = []
        for link in links:
            href = await link.get_attribute('href')
            if href: hrefs.append(href)
        return hrefs
    
    @staticmethod
    def clean_links(links):
        clean_links = [link.split('#')[0].split('?')[0] for link in links]
        
        links = []
        for link in clean_links:
            if link and link.startswith('https') and 'google' not in link and 'youtube' not in link:
                links.append(link)
        return links
    
    @staticmethod
    async def search_async(browser, topic):
        # Create a new browser context with the specified User-Agent
        context = await browser.new_context(user_agent=USER_AGENT)

        # Use the context to create a new page
        page = await context.new_page()

        # Set a generous timeout for navigation and loading of content
        page.set_default_timeout(PAGE_TIMEOUT)  # Apply to general actions

        await page.goto(SEARCH_URL + topic, wait_until="domcontentloaded")
        
        links = await SearchManager.get_links_with_jsname_async(page)
        final_links = SearchManager.clean_links(links)

        return final_links
        
