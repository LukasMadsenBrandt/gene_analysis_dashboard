import itertools
import sys
import warnings
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from gene_analysis_kutsche.decorators import timing_decorator
import multiprocessing
import csv
# Suppress FutureWarnings from statsmodels
warnings.simplefilter(action='ignore', category=FutureWarning)



def process_gene_combination(combination, time_series_data, progress_queue):
    gene1, gene2 = combination
    test_data = time_series_data[[gene2, gene1]]  # gene 1 causes gene 2
    try:
        if test_data.std(axis=0).eq(0).any():
            result = combination, {'error': 'constant data'}
        else:
            result = combination, grangercausalitytests(test_data, maxlag=1, verbose=False)
    except Exception as e:  # Replace InfeasibleTestError with generic Exception
        result = combination, {'error': str(e)}
    if progress_queue is not None:
        progress_queue.put(1)
    return result

def update_progress_bar(total_combinations, progress_queue):
    processed_combinations = 0
    while processed_combinations < total_combinations:
        processed_combinations += progress_queue.get()
        percent_complete = (processed_combinations / total_combinations) * 100
        sys.stdout.write(f"\rProgress: {processed_combinations}/{total_combinations} gene pairs processed ({percent_complete:.2f}%)")
        sys.stdout.flush()
    print()

@timing_decorator
def perform_granger_causality_tests(df_filtered_wt_weighted_mean, progress=False):
    """
    Perform Granger causality tests on all pairs of genes.
    """
    time_series_data = df_filtered_wt_weighted_mean.T  # To make each column a timeseries

    genes = df_filtered_wt_weighted_mean.index.tolist()[:100]
    # n * (n-1) combinations of genes
    gene_combinations = list(itertools.permutations(genes, 2))
    total_combinations = len(gene_combinations)

    gc_results = {}

    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue() if progress else None

        with multiprocessing.Pool() as pool:
            if progress:
                progress_updater = multiprocessing.Process(target=update_progress_bar, args=(total_combinations, progress_queue))
                progress_updater.start()
            
            results = pool.starmap(
                process_gene_combination,
                [(combination, time_series_data, progress_queue) for combination in gene_combinations]
            )

            for result in results:
                if result:
                    gc_results[result[0]] = result[1]

            if progress:
                progress_queue.put(total_combinations)  # Ensure progress updater finishes
                progress_updater.join()

    return gc_results



def collect_significant_edges(gc_results, p_value_threshold=0.05, file=False, filepath=None):
    """
    Collects gene pairs with significant Granger causality into a list of edges with their corresponding lag,
    and whether the relationship is positive or negative.
    """
    significant_edges = []
    # If the results are not stored, we can directly access the results
    if not file:
        for (gene1, gene2), results in gc_results.items():
            if 'error' in results or not results:
                continue
            for lag, result in results.items():
                if 'ssr_ftest' in result[0]:
                    p_value = result[0]['ssr_ftest'][1]  # Get the p-value of the F-test at this lag
                    if p_value < p_value_threshold:
                        # Append the edge with the corresponding lag
                        significant_edges.append(((gene1, lag), (gene2, 0), p_value))
    # If the results are stored, we need to read them from the file
    else:
        if filepath is None:
            raise ValueError("file_path must be provided when stored is True.")
        with open(filepath, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                gene1 = row['gene1']
                gene2 = row['gene2']
                try:
                    lag = int(row['lag'])
                    p_value = float(row['p-value'])
                except ValueError:
                    continue  # Skip rows with invalid lag or p-value
                
                if p_value < p_value_threshold:
                    significant_edges.append(((gene1, lag), (gene2, 0), p_value))
                    
    return significant_edges


def save_results_to_csv(gc_results, output_file):
    # Save the results to a csv, #gene1, gene2, lag, p-value
    with open(output_file, 'w') as f:
        f.write('gene1,gene2,lag,p-value\n')
        for (gene1, gene2), results in gc_results.items():
            if 'error' in results or not results:
                f.write(f"{gene1},{gene2},NaN,NaN\n")
                continue
            for lag, result in results.items():
                if 'ssr_ftest' in result[0]:
                    p_value = result[0]['ssr_ftest'][1]  # Get the p-value of the F-test at this lag
                    f.write(f"{gene1},{gene2},{lag},{p_value}\n")
                else:
                    f.write(f"{gene1},{gene2},{lag},NaN\n")
