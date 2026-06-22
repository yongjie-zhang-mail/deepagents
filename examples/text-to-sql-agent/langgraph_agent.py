import argparse
import os
import sys
from urllib.parse import quote_plus

from langchain_deepseek import ChatDeepSeek
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from rich.console import Console
from rich.panel import Panel

# Load environment variables
load_dotenv()

console = Console()


def _build_pg_uri() -> str:
    """Build a PostgreSQL SQLAlchemy URI from env vars, with clear errors."""
    user = os.getenv("DB_USER", "")
    pwd = os.getenv("DB_PWD", "")
    host = os.getenv("DB_HOST", "")
    dbname = os.getenv("DB_NAME", "")
    port = os.getenv("DB_PORT", "")

    # URL-encode user/password so special chars (~ @ : / # etc.) are safe.
    return f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(pwd)}@{host}:{port}/{dbname}"


def create_sql_deep_agent():
    """Create and return a text-to-SQL Deep Agent"""

    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Connect to Chinook database
    # db_path = os.path.join(base_dir, "chinook.db")
    # db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)
    db = SQLDatabase.from_uri(_build_pg_uri(), sample_rows_in_table_info=3)

    # model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    # Initialize deepseek-v4-flash model in non-thinking mode
    model = ChatDeepSeek(
        model="deepseek-v4-flash",
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )

    # Create SQL toolkit and get tools
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools = toolkit.get_tools()

    # Create the Deep Agent with all parameters
    agent = create_deep_agent(
        model=model,  # DeepSeek Chat with temperature=0
        memory=["./AGENTS.md"],  # Agent identity and general instructions
        skills=[
            "./skills/"
        ],  # Specialized workflows (query-writing, schema-exploration)
        tools=sql_tools,  # SQL database tools
        interrupt_on={
            "sql_db_schema": True,  # Default: approve, edit, reject
        },
        subagents=[],  # No subagents needed
        backend=FilesystemBackend(root_dir=base_dir),  # Persistent file storage
    )

    return agent


def main():
    """Main entry point for the SQL Deep Agent CLI"""
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Deep Agent powered by LangChain Deep Agents and Claude Sonnet 4.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py "What are the top 5 best-selling artists?"
  python agent.py "Which employee generated the most revenue by country?"
  python agent.py "How many customers are from Canada?"
        """,
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question to answer using the Chinook database",
    )

    args = parser.parse_args()

    # Display the question
    console.print(
        Panel(f"[bold cyan]Question:[/bold cyan] {args.question}", border_style="cyan")
    )
    console.print()

    # Create the agent
    console.print("[dim]Creating SQL Deep Agent...[/dim]")
    agent = create_sql_deep_agent()

    # Invoke the agent
    console.print("[dim]Processing query...[/dim]\n")

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": args.question}]}
        )

        # Extract and display the final answer
        final_message = result["messages"][-1]
        answer = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )

        console.print(
            Panel(f"[bold green]Answer:[/bold green]\n\n{answer}", border_style="green")
        )

    except Exception as e:
        console.print(
            Panel(f"[bold red]Error:[/bold red]\n\n{str(e)}", border_style="red")
        )
        sys.exit(1)


# Expose agent at module level for LangGraph API server (`langgraph dev`)
agent = create_sql_deep_agent()

# if __name__ == "__main__":
#     main()
