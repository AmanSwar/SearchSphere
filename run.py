import os
import time
import warnings
import platform
import random
from typing import Tuple, List, Optional
from datetime import datetime

# Rich for beautiful CLI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.text import Text
from rich.align import Align
from rich.style import Style
from rich.columns import Columns
from rich.rule import Rule
from rich.table import Table
from rich.box import ROUNDED
from rich import box

# Local imports
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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# --- Initialize Rich Console ---
console = Console(highlight=False)

# --- Color Configuration ---
# Gradient color scheme
GRADIENT_START = "#00FFFF"  # Bright cyan
GRADIENT_MID = "#00BFFF"    # Deep sky blue
GRADIENT_END = "#8A2BE2"    # Blue violet
ACCENT_COLOR = "#FF1493"    # Deep pink for pop accents
SUCCESS_COLOR = "#00FF7F"   # Spring green
WARNING_COLOR = "#FFD700"   # Gold
ERROR_COLOR = "#FF4500"     # Orange red

# --- ASCII Art for Logo ---
def get_logo() -> List[str]:
    logo = [
        "   ░██████╗███████╗░█████╗░██████╗░░█████╗░██╗░░██╗",
        "   ██╔════╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██║░░██║",
        "   ╚█████╗░█████╗░░███████║██████╔╝██║░░╚═╝███████║",
        "   ░╚═══██╗██╔══╝░░██╔══██║██╔══██╗██║░░██╗██╔══██║",
        "   ██████╔╝███████╗██║░░██║██║░░██║╚█████╔╝██║░░██║",
        "   ╚═════╝░╚══════╝╚═╝░░╚═╝╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝",
        "   ░██████╗██████╗░██╗░░██╗███████╗██████╗░███████╗",
        "   ██╔════╝██╔══██╗██║░░██║██╔════╝██╔══██╗██╔════╝",
        "   ╚█████╗░██████╔╝███████║█████╗░░██████╔╝█████╗░░",
        "   ░╚═══██╗██╔═══╝░██╔══██║██╔══╝░░██╔══██╗██╔══╝░░",
        "   ██████╔╝██║░░░░░██║░░██║███████╗██║░░██║███████╗",
        "   ╚═════╝░╚═╝░░░░░╚═╝░░╚═╝╚══════╝╚═╝░░╚═╝╚══════╝"
    ]
    return logo

def gradient_color(percent: float) -> str:
    """Generate a color in the gradient based on percentage (0-1)"""
    if percent < 0.5:
        # Interpolate between start and mid
        t = percent * 2
        r1, g1, b1 = int(GRADIENT_START[1:3], 16), int(GRADIENT_START[3:5], 16), int(GRADIENT_START[5:7], 16)
        r2, g2, b2 = int(GRADIENT_MID[1:3], 16), int(GRADIENT_MID[3:5], 16), int(GRADIENT_MID[5:7], 16)
    else:
        # Interpolate between mid and end
        t = (percent - 0.5) * 2
        r1, g1, b1 = int(GRADIENT_MID[1:3], 16), int(GRADIENT_MID[3:5], 16), int(GRADIENT_MID[5:7], 16)
        r2, g2, b2 = int(GRADIENT_END[1:3], 16), int(GRADIENT_END[3:5], 16), int(GRADIENT_END[5:7], 16)
    
    # Linear interpolation
    r = int(r1 * (1 - t) + r2 * t)
    g = int(g1 * (1 - t) + g2 * t)
    b = int(b1 * (1 - t) + b2 * t)
    
    return f"#{r:02x}{g:02x}{b:02x}"

def print_logo() -> None:
    """Print the stylized ASCII art logo with gradient color"""
    logo = get_logo()
    total_lines = len(logo)
    
    # Create aligned text for each line with gradient color
    styled_lines = []
    for i, line in enumerate(logo):
        percent = i / (total_lines - 1) if total_lines > 1 else 0
        color = gradient_color(percent)
        styled_line = Text(line, style=Style(color=color))
        styled_lines.append(Align.center(styled_line))
    
    # Print each line
    for line in styled_lines:
        console.print(line)

def create_animated_spinner():
    """Create a custom spinner animation for progress indicators"""
    frames = ["◢", "◣", "◤", "◥"]
    return frames

def print_welcome_message() -> None:
    """Print a stylish welcome message with decorative elements"""
    # Create a table for structured layout
    table = Table(show_header=False, box=box.ROUNDED, border_style=GRADIENT_MID)
    table.add_column("Content")
    
    # Welcome text with gradient effect
    current_time = datetime.now().strftime('%H:%M:%S')
    welcome_text = Text(f"[{current_time}] ", style=Style(color=GRADIENT_START, bold=True))
    welcome_text.append("Welcome to ", style="white")
    
    # Add each letter of "Search Sphere" with gradient color
    name = "Search Sphere"
    for i, char in enumerate(name):
        percent = i / (len(name) - 1) if len(name) > 1 else 0
        color = gradient_color(percent)
        welcome_text.append(char, style=Style(color=color, bold=True))
    
    welcome_text.append(" research preview", style="white")
    
    # Add inspirational subtitle
    subtitle = Text("Discover the universe of knowledge at lightspeed", style=Style(color=ACCENT_COLOR, italic=True))
    
    # Version info
    version_info = Text("v2.5.0-experimental | ", style="dim")
    version_info.append("AI-Powered Semantic Search", style=Style(color=GRADIENT_END, italic=True))
    
    # Add rows to table
    table.add_row(Align.center(welcome_text))
    table.add_row(Align.center(subtitle))
    table.add_row(Align.center(version_info))
    
    # Create outer panel
    panel = Panel(
        table,
        border_style=gradient_color(0.3),
        box=box.ROUNDED,
        expand=False,
        padding=(1, 3)
    )
    console.print(Align.center(panel))

def print_status_message(message: str, style: str = "white") -> None:
    """Print a status message with sparkles emoji"""
    text = Text("✨ ", style="white")
    text.append(message, style=style)
    console.print(text)

def clear_screen() -> None:
    """Clear the terminal screen based on OS"""
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def get_search_directory() -> str:
    """Prompt for and validate the search directory"""
    search_dir = ""
    
    # Create fancy panel for input prompt
    input_panel = Panel(
        Text("Please enter the directory path you want to index.", style="white"),
        title="[bold]Directory Selection[/bold]",
        title_align="left",
        border_style=gradient_color(0.2),
        box=box.ROUNDED
    )
    console.print(input_panel)
    
    while True:
        console.print("")
        temp_search_dir = Prompt.ask(
            Text("🔍 Directory path", style=Style(color=GRADIENT_MID, bold=True))
        )
        
        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold blue]Processing..."),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Validating directory...", total=None)
            
            try:
                search_dir = encoder.utils.prep_dir(temp_search_dir)
                # Add small delay for visual effect
                time.sleep(0.5)
            except Exception as e:
                progress.stop()
                console.print(f"[bold {ERROR_COLOR}]Error preparing directory path:[/bold {ERROR_COLOR}] {e}")
                search_dir = ""
                continue

            # Add small delay for visual effect
            time.sleep(0.3)
            
            if not os.path.exists(search_dir):
                progress.stop()
                error_panel = Panel(
                    Text(f"Directory not found: '{search_dir}'", style=Style(color=ERROR_COLOR)),
                    title="[bold red]Error[/bold red]",
                    border_style=ERROR_COLOR,
                    box=box.ROUNDED
                )
                console.print(error_panel)
                search_dir = ""
            else:
                progress.stop()
                success_panel = Panel(
                    Text(f"Directory successfully set: {search_dir}", style=Style(color=SUCCESS_COLOR)),
                    title="[bold green]Success[/bold green]",
                    border_style=SUCCESS_COLOR,
                    box=box.ROUNDED
                )
                console.print(success_panel)
                break
            
    return search_dir
def generate_embeddings(search_dir: str) -> Tuple[float, Tuple[int, int]]:
    """Generate embeddings for files in the search directory with fancy progress displays"""
    console.print("")
    
    # Create header panel
    header_panel = Panel(
        Align.center(Text("✨ Starting Embedding Generation ✨", style=Style(color=GRADIENT_END, bold=True))),
        border_style=gradient_color(0.5),
        box=box.ROUNDED
    )
    console.print(header_panel)
    
    # Create info table
    info_table = Table(box=box.SIMPLE, show_header=False, border_style=GRADIENT_MID)
    info_table.add_column("Property")
    info_table.add_column("Value")
    
    info_table.add_row(
        Text("Source Directory:", style=Style(color=GRADIENT_START, bold=True)),
        Text(search_dir, style="italic")
    )
    info_table.add_row(
        Text("Process Type:", style=Style(color=GRADIENT_START, bold=True)),
        Text("Semantic Vector Encoding", style=Style(color=ACCENT_COLOR))
    )
    info_table.add_row(
        Text("Start Time:", style=Style(color=GRADIENT_START, bold=True)),
        Text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), style="white")
    )
    
    console.print(info_table)
    console.print("")
    
    start_time = time.time()
    
    # Custom progress display
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold blue]Processing..."),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        embedding_task = progress.add_task("Generating embeddings...", total=None)
        
        try:
            # Pass the progress object to dir_traversal
            dir_traversal(search_dir=search_dir, console=console, external_progress=progress)
            progress.update(embedding_task, completed=80)
            time.sleep(0.3)  # Small delay for visual effect
            
            # Save state with visual indicator
            progress.update(embedding_task, description="Saving index state...")
            faiss_manager.save_state()
            progress.update(embedding_task, completed=100)
            time.sleep(0.3)  # Small delay for visual effect
            
            end_time = time.time() - start_time
            final_size = faiss_manager.current_size()
            return end_time, final_size
        except Exception as e:
            console.print(f"\n[bold {ERROR_COLOR}]❌ An error occurred during embedding generation:[/bold {ERROR_COLOR}]")
            console.print_exception(show_locals=False)
            exit(1)

def print_completion_stats(time_taken: float, index_size: Tuple[int, int]) -> None:
    """Print completion statistics with enhanced visuals"""
    console.print("")
    
    # Create a table for the statistics
    stats_table = Table(
        title="Embedding Generation Results",
        title_style=Style(color=SUCCESS_COLOR, bold=True),
        box=box.ROUNDED,
        border_style=gradient_color(0.7)
    )
    
    # Add columns
    stats_table.add_column("Metric", style=Style(color=GRADIENT_START, bold=True))
    stats_table.add_column("Value", style=Style(color=GRADIENT_END))
    
    # Add rows with stats
    stats_table.add_row("Time Taken", f"{time_taken:.2f} seconds")
    stats_table.add_row("Text Index Size", f"{index_size[0]} items")
    stats_table.add_row("Image Index Size", f"{index_size[1]} items")
    stats_table.add_row("Status", "Index state saved successfully")
    stats_table.add_row("Ready for", "Query operations")
    
    # Print the table
    console.print(stats_table)
    
    # Create success message
    success_panel = Panel(
        Text("✅ Embedding generation complete and ready for searching!", style=Style(color=SUCCESS_COLOR, bold=True)),
        border_style=SUCCESS_COLOR,
        box=box.ROUNDED
    )
    console.print(success_panel)
def run_query_loop() -> None:
    """Run the query loop for searching with enhanced UI"""
    # Create animated transition
    with console.status("[bold blue]Initializing search engine...", spinner="dots"):
        time.sleep(1.0)  # For visual effect
        
        try:
            query.faiss_init(console=console)
        except Exception as e:
            console.print(f"\n[bold {ERROR_COLOR}]❌ Failed to initialize search index:[/bold {ERROR_COLOR}]")
            console.print_exception(show_locals=False)
            exit(1)
    
    # Query interface panel
    query_panel = Panel(
        Align.center(Text("🔮 Ready for Queries 🔮", style=Style(color=GRADIENT_END, bold=True))),
        title="[bold]Search Sphere Terminal[/bold]",
        subtitle="Type 'exit' to quit | Type 'help' for commands",
        border_style=gradient_color(0.5),
        box=box.ROUNDED
    )
    console.print(query_panel)
    
    search_count = 0
    while True:
        console.print("")
        
        # Custom query prompt
        prompt_style = gradient_color(random.random())  # Random gradient position for variety
        q = Prompt.ask(
            Text(f"[{search_count}] query", style=Style(color=prompt_style, bold=True))
        )
        console.print(Rule(style=gradient_color(0.3), characters="·"))
        
        if q.lower() == 'exit':
            exit_panel = Panel(
                Text("Thank you for using Search Sphere!", style=Style(color=GRADIENT_MID)),
                title="[bold]Session Complete[/bold]",
                border_style=gradient_color(0.8),
                box=box.ROUNDED
            )
            console.print(exit_panel)
            
            login_text = Text("🌐 Login successful. Press ", style="blue")
            login_text.append("Enter", style=Style(color=GRADIENT_END, bold=True))
            login_text.append(" to continue", style="blue")
            console.print(login_text)
            break
        
        elif q.lower() == 'help':
            help_table = Table(box=box.ROUNDED, title="Available Commands", border_style=GRADIENT_MID)
            help_table.add_column("Command", style=Style(color=GRADIENT_START, bold=True))
            help_table.add_column("Description")
            
            help_table.add_row("exit", "Exit the Search Sphere application")
            help_table.add_row("help", "Display this help message")
            help_table.add_row("clear", "Clear the terminal screen")
            help_table.add_row("<query>", "Any other text will perform a semantic search")
            
            console.print(help_table)
            continue
            
        elif q.lower() == 'clear':
            clear_screen()
            print_logo()
            console.print(query_panel)
            continue
        
        if not q.strip():
            console.print(Text("⚠️ Please enter a query.", style=Style(color=WARNING_COLOR, bold=True)))
            continue
        
        search_count += 1
        
        # Animate the search process
        with console.status(f"[bold {GRADIENT_MID}]Searching for: {q}...", spinner="dots"):
            try:
                time.sleep(0.3)  # Small delay for visual effect
                # Pass is_nested=True to avoid creating nested live displays
                query.search(q, console=console, is_nested=True)
            except Exception as e:
                console.print(f"\n[bold {ERROR_COLOR}]❌ An error occurred during search:[/bold {ERROR_COLOR}]")
                console.print_exception(show_locals=False)

                
def startup_animation():
    """Display a startup animation"""
    clear_screen()
    
    # Spinner animation
    with console.status("[bold blue]Initializing Search Sphere...", spinner="dots") as status:
        time.sleep(0.7)
        status.update("[bold cyan]Loading modules...")
        time.sleep(0.5)
        status.update("[bold cyan]Configuring environment...")
        time.sleep(0.5)
        status.update("[bold cyan]Preparing interface...")
        time.sleep(0.7)
    
    # Clear again before showing the logo
    clear_screen()

def main():
    """Main function to run the Search Sphere Engine"""
    startup_animation()
    print_logo()
    print_welcome_message()
    
    search_dir = get_search_directory()
    time_taken, index_size = generate_embeddings(search_dir)
    print_completion_stats(time_taken, index_size)
    
    run_query_loop()

if __name__ == "__main__":
    main()