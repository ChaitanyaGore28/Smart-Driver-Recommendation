# ✅ test_ddg.py
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize the DuckDuckGo retriever
search = DuckDuckGoSearchRun()

# Run a sample query
result = search.run("safe braking techniques while driving a car")

# Print only first 500 characters of the search result
print(result[:500])