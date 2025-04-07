# main_seq.py
import os
import psutil
# Multiprocessing imports removed as they weren't used in the provided snippet's main path
# from multiprocessing import Process , Queue , Manager 
import numpy as np
import traceback 
import argparse
import torch
# from tqdm import tqdm # Replacing tqdm with rich.progress
import warnings
import json
import time
# Rich for beautiful CLI
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn
)
from rich.panel import Panel

# Local imports
try:
    import encoder.config as config
    import encoder.utils as utils
    import encoder.embedding as embedding
    from encoder.faiss_base import FAISSManagerHNSW
except ImportError as e:
    print(f"Error importing local modules in main_seq.py: {e}")
    exit(1)


warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Globals & Setup ---
# Use a default console, but allow passing one (from run.py)
default_console = Console() 
# Initialize FAISS Manager globally - verbosity controlled here
# Let's make verbose=False default, rely on rich progress/prints
faiss_manager = FAISSManagerHNSW(verbose=False) 

content_extractor_func = {
    "pdf" : utils.pdf_extractor,
    "txt" : utils.text_extractor,
    "docx" : utils.docs_extractor,
    "ppt" : utils.ppt_extractor, # Assuming these exist in utils
    "xlsx" : utils.excel_extractor, # Corrected common excel extension
    "xls" : utils.excel_extractor,
    "md" : utils.markdown_extractor
    # Add other mappings as needed based on utils implementation
}


def get_files_to_process(search_dir: str) -> list[str]:
    """Scans directory and returns a list of file paths to process."""
    file_list = []
    for dirpath, dirnames, filenames in os.walk(search_dir):
        # Simple exclusion example (can be made more robust)
        # Skip hidden directories (like .git, .vscode etc.)
        dirnames[:] = [d for d in dirnames if not d.startswith('.')] 
        # Skip specific restricted paths if needed (example from original code)
        # if any(restricted in dirpath.split(os.sep) for restricted in config.RESTRICTED_DIRS_INITIAL):
        #    continue 
        
        for filename in filenames:
             # Skip hidden files
            if filename.startswith('.'):
                continue
                
            file_path = os.path.join(dirpath, filename)
            file_ext = file_path.split('.')[-1].lower() # Use lower case for consistency
            if file_ext in config.SUPPORTED_EXT_IMG or file_ext in content_extractor_func:
                 file_list.append(file_path)
    return file_list

def process_directory_with_progress(search_dir: str, progress_callback: callable):
    """
    Processes directory and calls callback for progress updates.
    NO PRINTING.
    """
    faiss_manager.reset_index()
    files = get_files_to_process(search_dir)
    total_files = len(files)
    processed_count = 0
    errors = 0

    for file_path in files:
        current_file_name = os.path.basename(file_path)
        try:
            # --- Call your existing extraction/embedding logic ---
            # Replace prints within these functions or capture their output.
            # meta = utils.get_meta(file_path)
            # content = content_extract_logic(file_path) # Must not print
            # embedding = generate_embedding_logic(content, meta) # Must not print
            # store_embedding_logic(embedding, meta) # Must not print
            # --- Example: simulate work ---
            time.sleep(0.01)
            # ------------------------------
            processed_count += 1
            progress_callback(processed_count, total_files, current_file_name)
        except Exception as e:
            errors += 1
            # Report error via callback
            progress_callback(processed_count, total_files, current_file_name, message=f"[red]Error: {e}[/red]")

    # After loop: train, add, save
    faiss_manager.train_add()
    faiss_manager.save_state()
    final_counts = faiss_manager.current_size()

    # Maybe return final status (could also be part of last callback)
    return {"success": True, "processed": processed_count, "errors": errors, "final_counts": final_counts}

def dir_traversal(search_dir, console=default_console):       
    """
    Traverses directory, extracts content, generates embeddings, and adds to FAISS.
    Uses Rich progress bar.
    """
    console.print("[yellow]Resetting FAISS index before traversal...[/yellow]")
    faiss_manager.reset_index()
    current_size = faiss_manager.current_size()
    console.print(f"   📊 Initial Text Index: [cyan]{current_size[0]}[/cyan] items")
    console.print(f"   🖼️ Initial Image Index: [cyan]{current_size[1]}[/cyan] items")
    
    console.print(f"[blue]🔍 Starting traversal of:[/blue] [italic]{search_dir}[/italic]")

    # Collect all files first to get a total for the progress bar
    file_list = []
    for dirpath, dirnames, filenames in os.walk(search_dir):
        # Simple exclusion example (can be made more robust)
        # Skip hidden directories (like .git, .vscode etc.)
        dirnames[:] = [d for d in dirnames if not d.startswith('.')] 
        # Skip specific restricted paths if needed (example from original code)
        # if any(restricted in dirpath.split(os.sep) for restricted in config.RESTRICTED_DIRS_INITIAL):
        #    continue 
        
        for filename in filenames:
             # Skip hidden files
            if filename.startswith('.'):
                continue
                
            file_path = os.path.join(dirpath, filename)
            file_ext = file_path.split('.')[-1].lower() # Use lower case for consistency
            if file_ext in config.SUPPORTED_EXT_IMG or file_ext in content_extractor_func:
                 file_list.append(file_path)

    if not file_list:
        console.print("[yellow]⚠️ No supported files found in the specified directory.[/yellow]")
        return # Nothing to process

    # Setup Rich Progress Bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console
    )

    processed_files = 0
    errors = 0

    try:
        with progress:
            task_id = progress.add_task("[yellow]Processing files...", total=len(file_list))
            
            for file_path in file_list:
                progress.update(task_id, description=f"Processing: [cyan]{os.path.basename(file_path)}[/cyan]")
                try:
                    content_extract(file_path=file_path, console=console) # Pass console
                    processed_files += 1
                except Exception as e:
                    errors += 1
                    # Log error but continue processing other files
                    console.print(f"\n[bold red]Error processing file:[/bold red] [italic]{file_path}[/italic]")
                    console.print(f"[red]   {e}[/red]") 
                    # Optionally print traceback for debugging, but keep progress clean
                    # traceback.print_exc() 
                finally:
                    # Advance progress bar regardless of success/failure for this file
                    progress.advance(task_id)

        console.print(f"\n[blue]Traversal complete.[/blue]")
        console.print(f"  Processed: [green]{processed_files}[/green] files")
        if errors > 0:
            console.print(f"  Skipped due to errors: [red]{errors}[/red] files")

        # --- Training and Adding ---
        console.print("[yellow]Storing embeddings in FAISS index...[/yellow]")
        # This might take time, add a status indicator
        with console.status("[bold yellow]Adding vectors to FAISS...[/bold yellow]", spinner="dots"):
            faiss_manager.train_add() 
        console.print("[green]✅ Embeddings added to FAISS index.[/green]")

    except Exception as e:
        console.print(f"\n[bold red]❌ An critical error occurred during directory traversal or indexing:[/bold red]")
        console.print_exception(show_locals=False) # Rich traceback
        raise # Re-raise after logging


def content_extract(file_path, console=default_console):
    """
    Extracts content from a single file based on its extension.
    (Minor logging changes if needed, primarily relies on generate_embedding)
    """
    try:
        file_ext = file_path.split('.')[-1].lower()
        file_meta_data = utils.get_meta(file_path=file_path) # Assuming get_meta works

        content = None
        content_type = "unknown"

        # --- Text-based files ---
        if file_ext in content_extractor_func:
            extractor = content_extractor_func[file_ext]
            # console.print(f"  Extracting text from: {os.path.basename(file_path)}", style="dim") # Optional finer logs
            content = extractor(file_path=file_path)
            content_type = "text"
            
        # --- Image files ---
        elif file_ext in config.SUPPORTED_EXT_IMG:
            # console.print(f"  Processing image: {os.path.basename(file_path)}", style="dim") # Optional finer logs
            content = file_path + "~" # Signal for image processing in generate_embedding
            content_type = "image"
            
        # --- Generate Embedding ---
        if content is not None:
             content_dic = {"content": content, "metadata": file_meta_data, "type": content_type}
             generate_embedding(content_dic, console=console) # Pass console
        # else: # File type not supported or extractor failed silently
             # console.print(f"  [dim]Skipping unsupported or empty file: {os.path.basename(file_path)}[/dim]")
             
    except Exception as e:
        # Log error specifically for this file, but let dir_traversal handle overall flow
        console.print(f"\n[bold red]Error during content extraction for:[/bold red] [italic]{file_path}[/italic]")
        console.print(f"[red]   {e}[/red]")
        # Re-raise so the main loop catches it and increments the error count
        raise  


def generate_embedding(content_data, console=default_console):
    """
    Generates embedding for text or image content and queues it for FAISS.
    """
    try:  
        content = content_data["content"]
        metadata = content_data["metadata"]
        embed_type = content_data["type"] # "text" or "image"
        generated_embedding = None
	 	
        if embed_type == "image" and content.endswith("~"):
            img_path = content[:-1]
            if os.path.exists(img_path):
                # console.print(f"    Generating image embedding...", style="dim") # Optional
                generated_embedding = embedding.image_extract(img_path)
            else:
                console.print(f"[yellow]Warning: Image path not found after signal removal: {img_path}[/yellow]")
                return # Skip if path invalid

        elif embed_type == "text":
            if content and content.strip(): # Ensure content is not empty
                # console.print(f"    Generating text embedding...", style="dim") # Optional
                generated_embedding = embedding.text_extract(content)
            else:
                 # console.print(f"    Skipping empty text content for {metadata.get('file_name', 'file')}", style="dim") # Optional
                 return # Skip empty text

        else:
             console.print(f"[yellow]Warning: Unknown content type '{embed_type}' for {metadata.get('file_name', 'file')}[/yellow]")
             return

        # --- Store Embedding Temporarily ---
        if generated_embedding is not None:
            if isinstance(generated_embedding, torch.Tensor):
                generated_embedding = generated_embedding.cpu().numpy()
            
            # Ensure it's a 2D array for FAISS
            if generated_embedding.ndim == 1:
                 generated_embedding = np.expand_dims(generated_embedding, axis=0)

            # Basic check for NaN/Inf before storing (optional but good practice)
            if not np.isfinite(generated_embedding).all():
                 console.print(f"[red]Warning: Embedding for {metadata.get('file_name', 'file')} contains NaN or Inf. Skipping.[/red]")
                 return
                 
            # Normalize embeddings? FAISS often works better with normalized vectors for cosine sim
            # If your embedding models don't normalize, uncomment below:
            # faiss.normalize_L2(generated_embedding) 
            
            # Debugging norm print - keep if useful, remove for cleaner output
            # norm = np.linalg.norm(generated_embedding)
            # console.print(f"    Embedding norm for {metadata.get('file_name', 'file')}: {norm:.4f}", style="dim")

            data = (embed_type, generated_embedding, metadata)
            store_embedding(data, console=console) # Pass console
            
    except Exception as e:
        console.print(f"\n[bold red]Error during embedding generation for {metadata.get('file_name', 'file')}:[/bold red]")
        console.print(f"[red]   {e}[/red]")
        # Don't raise here, allow main loop to continue with other files


def store_embedding(data: tuple, console=default_console):
    """
    Temporarily stores embedding and metadata in FAISSManager's buffer.
    """
    try:
        embed_type, embedding_vec, metadata = data
        faiss_manager.store_temp(type=embed_type, embedding=embedding_vec, metadata=metadata)
        # console.print(f"      Stored {embed_type} embedding for {metadata['file_name']}", style="dim") # Very verbose

    except Exception as e:
        file_info = metadata.get('file_name', '[unknown file]')
        console.print(f"\n[bold red]Error storing embedding temporarily for {file_info}:[/bold red]")
        console.print(f"[red]   {e}[/red]")
        # Potentially log traceback here if needed for debugging storage issues
        # traceback.print_exc()


# --- Standalone Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Generation CLI")
    parser.add_argument("--dir", type=str, required=True, help="Directory to scan and create embeddings for")
    # Keep verbose flag if you want detailed file-by-file console output during processing
    # parser.add_argument("--verbose", action="store_true", help="Detailed output during processing") 
    args = parser.parse_args()

    # Use a specific console for standalone execution
    standalone_console = Console() 

    try:
        search_dir_arg = utils.prep_dir(args.dir)
        if not os.path.exists(search_dir_arg):
             standalone_console.print(f"[bold red]Error:[/bold red] Directory not found: {search_dir_arg}")
             exit(1)
             
        standalone_console.print(Panel(f"[bold green]Starting Standalone Embedding Generation[/bold green]\nDirectory: [cyan]{search_dir_arg}[/cyan]", border_style="blue"))
        
        start_time = time.time()
        
        # Call the main traversal function, passing the console
        dir_traversal(search_dir=search_dir_arg, console=standalone_console) 
        
        # Save the state after traversal completes
        standalone_console.print("[yellow]💾 Saving final FAISS index state...[/yellow]")
        faiss_manager.save_state()
        standalone_console.print("[green]✅ Index state saved.[/green]")
        
        end_time = time.time() - start_time
        final_size = faiss_manager.current_size()

        standalone_console.print("\n" + "="*50)
        standalone_console.print(f"[bold green]✨ Standalone Embedding Generation Finished ✨[/bold green]")
        standalone_console.print(f"⏱️ Total Time: [cyan]{end_time:.2f}[/cyan] seconds")
        standalone_console.print(f"📊 Final Text Index Size: [cyan]{final_size[0]}[/cyan] items")
        standalone_console.print(f"🖼️ Final Image Index Size: [cyan]{final_size[1]}[/cyan] items")
        standalone_console.print("="*50 + "\n")

    except Exception as e:
        standalone_console.print(f"\n[bold red]❌ A critical error occurred during standalone execution:[/bold red]")
        standalone_console.print_exception(show_locals=False)
        exit(1)