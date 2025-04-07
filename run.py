# run.py
import os
import time
import warnings

# Rich for beautiful CLI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt # Using rich's prompt for consistency

# Local imports (assuming they are structured correctly)
try:
    import encoder.main_seq
    import encoder.utils
    from query import query
    import encoder
    from encoder.main_seq import dir_traversal, faiss_manager
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure your project structure and PYTHONPATH are correct.")
    exit(1)

# --- Configuration ---
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set GPU - adjust if needed

# --- Initialize Rich Console ---
console = Console()

# --- Welcome Message ---
console.print(Panel("[bold magenta]🚀 Welcome to Semantic Search Engine 🚀[/bold magenta]", title="Initialization", border_style="blue"))

# --- Get Search Directory ---
search_dir = ""
while True:
    # Use rich.prompt for better input handling
    temp_search_dir = Prompt.ask("[bold cyan]Enter the directory path to index[/bold cyan]")
    
    # Attempt to prepare the directory path
    try:
        search_dir = encoder.utils.prep_dir(temp_search_dir)
    except Exception as e:
        console.print(f"[bold red]Error preparing directory path:[/bold red] {e}")
        search_dir = "" # Reset on error
        continue # Ask again

    if not os.path.exists(search_dir):
        console.print(f"[bold red]Error:[/bold red] Directory not found: '[italic]{search_dir}[/italic]'")
        console.print("Please enter a valid directory path.")
        search_dir = "" # Reset search_dir so the loop continues
    else:
        console.print(f"[bold green]✅ Directory set:[/bold green] [italic]{search_dir}[/italic]")
        break

# --- Embedding Generation ---
console.print("\n" + "="*50)
console.print("[bold yellow]✨ Starting Embedding Generation ✨[/bold yellow]")
console.print(f"Indexing files in: [italic]{search_dir}[/italic]")
console.print("="*50 + "\n")

start_time = time.time()

# Assuming dir_traversal now uses rich.progress internally (see main_seq.py changes)
try:
    # Pass the console object if main_seq needs it for printing
    dir_traversal(search_dir=search_dir, console=console) 
    faiss_manager.save_state() # Save state after traversal and adding
    end_time = time.time() - start_time
    final_size = faiss_manager.current_size()
    console.print("\n" + "="*50)
    console.print(f"[bold green]✅ Embedding generation complete![/bold green]")
    console.print(f"⏱️ Time taken: [cyan]{end_time:.2f}[/cyan] seconds")
    console.print(f"📊 Text Index Size: [cyan]{final_size[0]}[/cyan] items")
    console.print(f"🖼️ Image Index Size: [cyan]{final_size[1]}[/cyan] items")
    console.print(f"💾 Index state saved.")
    console.print("="*50 + "\n")

except Exception as e:
    console.print(f"\n[bold red]❌ An error occurred during embedding generation:[/bold red]")
    console.print_exception(show_locals=False) # Prints traceback nicely
    exit(1)


# --- Query Loop ---
console.print(Panel("[bold magenta]🔍 Ready for Queries 🔍[/bold magenta]", title="Search Phase", border_style="blue"))

# Initialize query module (ensure FAISS loads index here)
try:
    query.faiss_init(console=console) # Pass console if needed for init messages
except Exception as e:
    console.print(f"\n[bold red]❌ Failed to initialize search index:[/bold red]")
    console.print_exception(show_locals=False)
    exit(1)

while True:
    console.print("\n" + "-"*50)
    # Use rich Prompt again
    q = Prompt.ask("[bold cyan]Enter your search query (or type 'exit' to quit)[/bold cyan]")
    console.print("-"*50)


    if q.lower() == 'exit':
        console.print("[bold yellow]👋 Exiting Semantic Search Engine. Goodbye![/bold yellow]")
        break
    
    if not q.strip():
        console.print("[bold yellow]⚠️ Please enter a query.[/bold yellow]")
        continue

    # Perform search using the query module
    try:
        # Pass console to search function for rich output
        query.search(q, console=console) 
    except Exception as e:
        console.print(f"\n[bold red]❌ An error occurred during search:[/bold red]")
        console.print_exception(show_locals=False) # Nicely formatted traceback