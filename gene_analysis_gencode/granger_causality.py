from itertools import permutations
import itertools
import os
import warnings
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.tsa.stattools import grangercausalitytests
from gene_analysis_gencode.decorators import timing_decorator
import sys
import multiprocessing
# Suppress FutureWarnings from statsmodels

# Add the parent directory to sys.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.simplefilter(action='ignore', category=FutureWarning)


def process_gene_ZEB2_caused_by(gene, fixed_gene, time_series_data, progress_queue):
    if gene == fixed_gene:
        return None
    test_data = time_series_data[[fixed_gene, gene]]
    if test_data.std(axis=0).eq(0).any():
        result = (fixed_gene, gene), 'constant data'
    else:
        try:
            result = (fixed_gene, gene), grangercausalitytests(test_data, maxlag=1, verbose=False)
        except Exception as e:
            result = (fixed_gene, gene), str(e)
    progress_queue.put(1)
    return result

def process_gene_caused_by_ZEB2(gene, fixed_gene, time_series_data, progress_queue):
    if gene == fixed_gene:
        return None
    test_data = time_series_data[[gene, fixed_gene]]
    if test_data.std(axis=0).eq(0).any():
        result = (gene, fixed_gene), 'constant data'
    else:
        try:
            result = (gene, fixed_gene), grangercausalitytests(test_data, maxlag=1, verbose=False)
        except Exception as e:
            result = (gene, fixed_gene), str(e)
    progress_queue.put(1)
    return result

def update_progress_bar(total_genes, progress_queue):
    processed_genes = 0
    while processed_genes < total_genes:
        processed_genes += progress_queue.get()
        percent_complete = (processed_genes / total_genes) * 100
        sys.stdout.write(f"\rProgress: {processed_genes}/{total_genes} genes processed ({percent_complete:.2f}%)")
        sys.stdout.flush()
    print()

@timing_decorator
def perform_granger_causality_tests_ZEB2_caused_by_gene(data_weighted_mean, fixed_gene='ZEB2', progress=False):
    time_series_data = data_weighted_mean.transpose()
    genes = time_series_data.columns.tolist()

    if fixed_gene not in genes:
        return f"Gene {fixed_gene} not found in the dataset."

    gc_results = {}
    total_genes = len(genes) - 1  # Exclude the fixed gene itself

    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()

        with multiprocessing.Pool() as pool:
            if progress:
                progress_updater = multiprocessing.Process(target=update_progress_bar, args=(total_genes, progress_queue))
                progress_updater.start()
            results = pool.starmap(
                process_gene_ZEB2_caused_by, 
                [(gene, fixed_gene, time_series_data, progress_queue) for gene in genes if gene != fixed_gene]
            )
            for result in results:
                if result:
                    gc_results[result[0]] = result[1]
            if progress:
                progress_queue.put(total_genes)  # Ensure progress updater finishes
                progress_updater.join()

    return gc_results

@timing_decorator
def perform_granger_causality_tests_gene_caused_by_ZEB2(data_weighted_mean, fixed_gene='ZEB2', progress=False):
    time_series_data = data_weighted_mean.transpose()
    genes = time_series_data.columns.tolist()

    if fixed_gene not in genes:
        return f"Gene {fixed_gene} not found in the dataset."

    gc_results = {}
    total_genes = len(genes) - 1  # Exclude the fixed gene itself

    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()

        with multiprocessing.Pool() as pool:
            if progress:
                progress_updater = multiprocessing.Process(target=update_progress_bar, args=(total_genes, progress_queue))
                progress_updater.start()
            results = pool.starmap(
                process_gene_caused_by_ZEB2, 
                [(gene, fixed_gene, time_series_data, progress_queue) for gene in genes if gene != fixed_gene]
            )
            for result in results:
                if result:
                    gc_results[result[0]] = result[1]
            if progress:
                progress_queue.put(total_genes)  # Ensure progress updater finishes
                progress_updater.join()

    return gc_results

def process_gene_pair(gene_pair, time_series_data, progress_queue):
    gene1, gene2 = gene_pair
    try:
        # Gene 1 causing gene 2
        test_data = time_series_data[[gene2, gene1]]
        if test_data.std(axis=0).eq(0).any():
            result = (gene1, gene2), {'error': 'constant data'}
        else:
            result = (gene1, gene2), grangercausalitytests(test_data, maxlag=1, verbose=False)
    except Exception as e:
        result = (gene1, gene2), {'error': str(e)}
    progress_queue.put(1)
    return result

@timing_decorator
def perform_granger_causality_tests_tf(data_weighted_mean, genes_file='gene_names.txt', progress=False):
    """
    Perform Granger causality tests on all permutations of a list of genes,
    excluding pairs where either series is constant, with optional progress output.
    """
    time_series_data = data_weighted_mean.transpose()
    # Remove all columns with only zeros
    time_series_data = time_series_data.loc[:, (time_series_data != 0).any(axis=0)]

    # Read the list of specific genes from the file
    with open(genes_file, 'r') as file:
        genes = [line.strip() for line in file.readlines()]

    # Filter columns in time_series_data to include only the genes listed in the file
    genes = [gene for gene in genes if gene in time_series_data.columns]

    gc_results = {}
    gene_combinations = list(itertools.permutations(genes, 2))
    total_combinations = len(gene_combinations)

    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()

        with multiprocessing.Pool() as pool:
            if progress:
                progress_updater = multiprocessing.Process(target=update_progress_bar, args=(total_combinations, progress_queue))
                progress_updater.start()
            results = pool.starmap(
                process_gene_pair,
                [(gene_pair, time_series_data, progress_queue) for gene_pair in gene_combinations]
            )
            for result in results:
                if result:
                    gc_results[result[0]] = result[1]
            if progress:
                progress_queue.put(total_combinations)  # Ensure progress updater finishes
                progress_updater.join()

    return gc_results


def collect_significant_edges(gc_results, p_value_threshold=0.05):
    """
    Collects gene pairs with significant Granger causality into a list of edges with their corresponding lag,
    and whether the relationship is positive or negative.
    """
    significant_edges = []
    for (gene1, gene2), results in gc_results.items():
        for lag, result in results.items():
            
            p_value = result[0]['ssr_ftest'][1]  # Get the p-value of the F-test at this lag
            if p_value < p_value_threshold:
                # Append the edge with the corresponding lag
                significant_edges.append(((gene1, lag), (gene2, 0)))
    return significant_edges

def collect_significant_edges_tf(tf_genes, p_value_threshold=0.05):
    """
    Collects gene pairs with significant Granger causality into a list of edges with their corresponding lag,
    and whether the relationship is positive or negative.
    """
    significant_edges = []
    for (gene1, gene2), results in tf_genes.items():
        if 'error' in results or not results:
            #print(f"Error processing {gene1}, {gene2}: {results['error']}")
            continue
        for lag, result in results.items():
            if 'ssr_ftest' in result[0]:
                p_value = result[0]['ssr_ftest'][1]  # Get the p-value of the F-test at this lag
                if p_value < p_value_threshold:
                    # Append the edge with the corresponding lag
                    # Gene 2 is causing gene 1
                    significant_edges.append(((gene1, lag), (gene2, 0), p_value))
            else:
                pass
                #print(f"No 'ssr_ftest' found for {gene1}, {gene2} at lag {lag}")
    return significant_edges
