import concurrent.futures
from typing import Callable, List, Any, Tuple
import traceback
from tqdm import tqdm


def process_in_parallel(
    items: List[Tuple[Any, ...]],
    process_func: Callable,
    max_workers: int = 10,
    show_progress: bool = True
) -> List[dict]:
    """
    Process items in parallel using ThreadPoolExecutor with progress tracking.

    Args:
        items: List of tuples, each containing arguments for process_func
        process_func: Function to process each item
        max_workers: Maximum number of parallel threads
        show_progress: Whether to display progress using tqdm

    Returns:
        List of results ordered by the first element of each item tuple (assumed to be index)
    """
    total = len(items)
    results_dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and create a mapping from future to index
        future_to_index = {}
        for item in items:
            # First element is assumed to be the index
            index = item[0]
            future = executor.submit(process_func, *item)
            future_to_index[future] = index

        # Collect results as they complete with tqdm progress bar.
        futures_iterator = concurrent.futures.as_completed(future_to_index)
        if show_progress:
            futures_iterator = tqdm(
                futures_iterator,
                total=total,
                desc="Processing",
                unit="sample",
            )

        for future in futures_iterator:
            index = future_to_index[future]
            try:
                result = future.result()
            except Exception as exc:
                for pending_future in future_to_index:
                    if not pending_future.done():
                        pending_future.cancel()
                tb = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
                raise RuntimeError(
                    f"Parallel processing failed at item index={index}.\n"
                    f"Original exception:\n{tb}"
                ) from exc
            results_dict[index] = result

    # Sort results by index to maintain order
    results = [results_dict[i] for i in sorted(results_dict.keys())]

    return results
