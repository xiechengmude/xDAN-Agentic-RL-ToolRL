import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

config = {
    "finance": {
        # make sure you start your weather server on port 8000
        "url": "http://34.87.170.99:7223/mcp",
        "transport": "sse",
    }
}


async def main():
    async with MultiServerMCPClient(config) as client:
        return client.get_tools()


if __name__ == "__main__":
    tool_list = asyncio.run(main())
    print(tool_list[0])
