# Real-time Google Research Assistant

This Streamlit application acts as a real-time research assistant, leveraging the power of Google search and OpenAI's GPT-4 model to generate responses to user queries based on content scraped from the first 10 pages of Google search results.

## Features

- **Search and Scrape:** Automates the process of searching Google with a user's query, scraping the first 10 pages of results.
- **In-Memory Vector Database:** Uses the scraped content to create an in-memory vector database for efficient information retrieval.
- **OpenAI GPT-4 Integration:** Utilizes OpenAI's GPT-4 model to generate a detailed response to the research question based on the most relevant snippets from the scraped content.

## Installation

### Dependencies

Before running the application, you need to install the necessary dependencies. Please follow the steps below:

1. **Create a Conda Environment (Optional but Recommended):**
2. **Install Required Python Packages:**
3. **Install Playwright:**

### Setting Up Your API Key

- An OpenAI API key is necessary for running the application. You can get your API key from [OpenAI Platform](https://platform.openai.com/api-keys) if you have a ChatGPT Plus account.

## Usage

1. **Start the Streamlit Application:**
Run the application by navigating to the application's directory and using the command:

2. **Enter Your OpenAI API Key:**
Upon launching the application, you'll be prompted to enter your OpenAI API key in the sidebar.

3. **Using the Application:**
- Enter your research question in the provided text input.
- Click the "Search" button to initiate the research process.
- Wait for the application to scrape Google search results, analyze the content, and generate a response based on the most relevant information.

## Limitations and Notes

- The application requires a stable internet connection for scraping and accessing the OpenAI API.
- Response times may vary based on the complexity of the query and the volume of content to be analyzed.
- The application does not store user queries or responses, ensuring privacy.

## Contributions

Contributions to improve the Real-time Google Research Assistant are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
