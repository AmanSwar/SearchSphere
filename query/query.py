# query.py
import faiss
import numpy as np
import json
import argparse # Keep for standalone use, but run.py won't use it
import time
import warnings
import os
import logging

# Rich for beautiful CLI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track, Progress, SpinnerColumn, TextColumn # Use rich progress

# --- Configuration ---
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Make TF/CUDA less verbose - adjust GPU visibility as needed for query phase
# Usually, embedding lookup is CPU-bound or uses FAISS's GPU index if configured
# Setting to -1 ensures CPU for TF operations here if any were accidentally used.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Local imports
try:
    from encoder.embedding import text_extract # Assuming this works standalone
    from encoder.faiss_base import FAISSManagerHNSW
    from query import utils # Assuming utils exists in query module/submodule
except ImportError as e:
    print(f"Error importing local modules in query.py: {e}")
    # Handle case where run.py runs this vs running query.py standalone

# --- Globals ---
# Use a default console, but allow passing one (from run.py)
default_console = Console()
faiss_manager = FAISSManagerHNSW(verbose=False) # Verbosity controlled by prints now
faiss_init_flag = 0

# --- Initialization ---
def faiss_init(console=default_console):
    """Initializes FAISS index, prints status using Rich."""
    global faiss_init_flag
    if faiss_init_flag == 0:
        console.print("[yellow]💾 Loading FAISS index and metadata...[/yellow]")
        try:
            start_time = time.time()
            faiss_manager.load_state()
            load_time = time.time() - start_time
            console.print(f"[green]✅ FAISS index loaded successfully in {load_time:.2f}s.[/green]")
            current_size = faiss_manager.current_size()
            console.print(f"   📊 Text Index: [cyan]{current_size[0]}[/cyan] items")
            console.print(f"   🖼️ Image Index: [cyan]{current_size[1]}[/cyan] items")
            faiss_init_flag = 1
        except Exception as e:
            console.print(f"[bold red]❌ Error loading FAISS index:[/bold red] {e}")
            console.print_exception(show_locals=False)
            raise # Re-raise to signal failure

# --- Query Processing ---
def query_extractor(query: str, console=default_console):
    """
    Converts the query to embedding. Uses Rich status.
    """
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        transient=True, # Spinner disappears after completion
        console=console
    ) as progress:
        task = progress.add_task("Analyzing query type...", total=1)
        # Assuming index_token is fast, no real progress needed here
        type_token = utils.index_token(query=query) 
        progress.update(task, completed=1, description=f"Query type: [bold]{type_token}[/bold]")
        
        task = progress.add_task("Generating query embedding...", total=1)
        # Assuming text_extract handles model loading internally if needed
        query_embed = text_extract(query) 
        progress.update(task, completed=1, description="Query embedding generated.")

    return type_token, query_embed

# --- Search Execution ---
def search(query: str, console=default_console, verbose=False, k: int = 5):
    """
    Main function for search, using Rich for status and results.
    Args:
        query (str): The search query.
        console (Console): Rich console instance for printing.
        verbose (bool): Whether to show similarity scores.
        k (int): Number of results to retrieve.
    """
    global faiss_init_flag
    if faiss_init_flag == 0:
        # Should have been called by run.py, but call defensively
        try:
            faiss_init(console=console)
        except Exception:
            console.print("[bold red]Search cannot proceed without a loaded index.[/bold red]")
            return # Exit search if init failed

    console.print(f"\n[cyan]Searching for:[/cyan] [italic]'{query}'[/italic]")
    
    search_start_time = time.time()
    dist = []
    indice = []
    metadata = {}
    
    try:
        # --- Get Query Embedding ---
        type_token, query_embed = query_extractor(query=query, console=console)

        # --- Perform Search ---
        description = f"Searching {type_token.lower()} index..."
        search_func = None
        if type_token == "TEXT":
            search_func = faiss_manager.search_text
        elif type_token == "IMAGE":
            search_func = faiss_manager.search_image
        else:
             console.print(f"[bold red]Error:[/bold red] Invalid query type token '{type_token}' generated.")
             return

        # Using rich.progress.track for the search itself if it's potentially long
        # Note: FAISS search is often very fast, so track might flash briefly.
        # If search is always instant, a simple status message might be better.
        # Let's use track assuming it *could* take a moment on huge indices.
        
        # We can't easily use track() directly if the function doesn't yield progress.
        # Let's use a status spinner instead.
        with console.status(f"[bold yellow]{description}[/bold yellow]", spinner="earth"):
            dist, indice, metadata = search_func(query_embed=query_embed)
            
    except Exception as e:
        console.print(f"[bold red]❌ Error during search execution:[/bold red]")
        console.print_exception(show_locals=False)
        return

    search_end_time = time.time() - search_start_time

    # --- Display Results ---
    if not indice.size > 0 or not metadata: # Check if indice is non-empty numpy array
        console.print("[bold orange_red1]😔 No relevant results found.[/bold orange_red1]")
        console.print(f"⏱️ Search took {search_end_time:.3f} seconds.")
    else:
        console.print(f"[bold green]✅ Found {len(indice)} results in {search_end_time:.3f} seconds:[/bold green]")

        table = Table(title="Search Results", show_header=True, header_style="bold magenta", border_style="blue")
        table.add_column("Rank", style="dim", width=4)
        table.add_column("File Name", style="bold green", no_wrap=True)
        table.add_column("File Path", style="cyan")
        if verbose:
            table.add_column("Similarity", style="yellow", justify="right")

        for i, faiss_id in enumerate(indice):
            rank = i + 1
            try:
                result_meta = metadata[str(faiss_id)] # FAISS IDs might be int64, ensure consistency
                file_name = result_meta.get("file_name", "N/A")
                file_path = result_meta.get("file_path", "N/A")
                
                row_data = [str(rank), file_name, file_path]
                if verbose:
                    similarity_score = dist[i]
                    row_data.append(f"{similarity_score:.4f}")
                    
                table.add_row(*row_data)
                
            except KeyError:
                 console.print(f"[yellow]Warning: Metadata not found for FAISS ID {faiss_id}[/yellow]")
                 row_data = [str(rank), f"ID: {faiss_id}", "[Metadata Missing]"]
                 if verbose:
                     similarity_score = dist[i]
                     row_data.append(f"{similarity_score:.4f}")
                 table.add_row(*row_data)
            except Exception as e:
                 console.print(f"[red]Error processing result {faiss_id}: {e}[/red]")
                 row_data = [str(rank), f"ID: {faiss_id}", "[Error Processing]"]
                 if verbose:
                     row_data.append("N/A")
                 table.add_row(*row_data)


        console.print(table)


def search_logic(query: str, k: int = 10) -> list: # Note: RETURN type
    """
    Performs search and returns structured results. NO PRINTING.
    """
    # ... (faiss_init check if needed) ...
    start_time = time.time()

    # ... query_extractor logic ...
    type_token, query_embed = query_extractor(query) # Assume this works

    results_list = []
    if type_token == "TEXT":
        distances, indices, metadata = faiss_manager.search_text(query_embed=query_embed, k=k)
    elif type_token == "IMAGE":
        distances, indices, metadata = faiss_manager.search_image(query_embed=query_embed, k=k)
    else:
        raise ValueError(f"Invalid token type: {type_token}")

    # Check if indices is not None and has elements
    if indices is not None and indices.size > 0:
        for i in range(len(indices)):
            faiss_id = indices[i]
            dist = distances[i]
            try:
                # Ensure metadata keys are consistent (using str(faiss_id))
                result_meta = metadata.get(str(faiss_id), {}) 
                results_list.append({
                    "rank": i + 1,
                    "score": float(dist), # Ensure score is float
                    "name": result_meta.get("file_name", "N/A"),
                    "path": result_meta.get("file_path", "N/A"),
                    "id": int(faiss_id) # Store original ID if needed
                })
            except Exception as e:
                 # Log this error if possible, maybe return partial results
                 print(f"Error processing metadata for ID {faiss_id}: {e}") # Temp print for debug
                 results_list.append({
                     "rank": i + 1,
                     "score": float(dist),
                     "name": f"[Error Processing ID {faiss_id}]",
                     "path": "N/A",
                     "id": int(faiss_id)
                 })

    duration = time.time() - start_time
    # Return the list and maybe duration/other info if needed by caller
    # The TUI worker will handle adding the duration message
    return results_list

# --- Standalone Execution (for CLI use of query.py directly) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Search Tool")
    parser.add_argument("--search", type=str, required=True, help="Your search query")
    parser.add_argument("--verbose", action="store_true", help="Show similarity scores")
    parser.add_argument("-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    standalone_console = Console() # Use a separate console for standalone mode
    
    try:
        faiss_init(console=standalone_console) # Initialize FAISS
        search(args.search, console=standalone_console, verbose=args.verbose, k=args.k) # Perform search
    except Exception as e:
        standalone_console.print("[bold red]An error occurred during standalone execution:[/bold red]")
        standalone_console.print_exception(show_locals=False)