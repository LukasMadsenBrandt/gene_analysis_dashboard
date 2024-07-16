import os
from gene_analysis_kutsche.granger_causality import perform_granger_causality_tests as perform_gc_kutsche
from gene_analysis_kutsche.granger_causality import collect_significant_edges as collect_significant_edges_kutsche
from gene_analysis_kutsche.granger_causality import save_results_to_csv as save_results_to_csv_kutsche
from gene_analysis_kutsche.data_preprocessing import load_and_preprocess_data as load_and_preprocess_kutsche
from gene_analysis_kutsche.data_filtering import filter_data_proximity_based_weights as filter_proximity_kutsche
from gene_analysis_kutsche.data_filtering import filter_data_arithmetic_mean as filter_mean_kutsche
from gene_analysis_kutsche.data_filtering import filter_data_median as filter_median_kutsche
from gene_analysis_kutsche.data_filtering import filter_data_wt as filter_wt_kutsche


def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_kutsche(os.path.join('Data', 'Kutsche', 'genes_all.txt'))
    print("Data loaded and preprocessed.")
    df_filtered, raw_data, day_map = filter_proximity_kutsche(df)
    print("Data filtered.")
    gc_results = perform_gc_kutsche(df_filtered, progress=True)
    print("Granger causality tests performed.")

    # Save results
    save_results_to_csv_kutsche(gc_results, "granger_causality_results.csv")
    print("Results saved to granger_causality_results.csv")
    

if __name__ == '__main__':
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    main()
