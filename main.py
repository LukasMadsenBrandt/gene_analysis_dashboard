import os
#Kutsche
from gene_analysis_kutsche.granger_causality import perform_granger_causality_tests as perform_gc_kutsche
from gene_analysis_kutsche.granger_causality import collect_significant_edges as collect_significant_edges_kutsche
from gene_analysis_kutsche.granger_causality import save_results_to_csv as save_results_to_csv_kutsche
from gene_analysis_kutsche.data_preprocessing import load_and_preprocess_data as load_and_preprocess_kutsche
from gene_analysis_kutsche.data_filtering import filter_data_proximity_based_weights as filter_proximity_kutsche
from gene_analysis_kutsche.data_filtering import filter_data_arithmetic_mean as filter_mean_kutsche
from gene_analysis_kutsche.data_filtering import filter_data_median as filter_median_kutsche
from gene_analysis_kutsche.data_filtering import filter_data_wt as filter_wt_kutsche

#Benito
from gene_analysis_benito.granger_causality import perform_granger_causality_tests as perform_gc_benito
from gene_analysis_benito.granger_causality import collect_significant_edges as collect_significant_edges_benito
from gene_analysis_benito.data_preprocessing import filter_data_proximity_based_weights as filter_proximity_benito
from gene_analysis_benito.data_preprocessing import filter_data_arithmetic_mean as filter_mean_benito
from gene_analysis_benito.data_preprocessing import filter_data_median as filter_median_benito
from gene_analysis_benito.data_filtering import filter_data as mapper_benito

def main():
    kutsche = True
    if kutsche:            
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df = load_and_preprocess_kutsche(os.path.join('Data', 'Kutsche', 'genes_all.txt'))
        print("Data loaded and preprocessed.")
        df_filtered, raw_data, day_map = filter_proximity_kutsche(df)
        print("Data filtered.")
        #print 5 lines of the data
        print(df_filtered.head())
        gc_results = perform_gc_kutsche(df_filtered, progress=True)
        print("Granger causality tests performed.")

        # Save results
        save_results_to_csv_kutsche(gc_results, "granger_causality_results_2.csv")
        print("Results saved to granger_causality_results_2.csv")
    else:
        df_human = mapper_benito(
            datafile=os.path.join('Data', 'Benito', 'Benito_Human'),
            mappingfile=os.path.join('Data', 'Benito', 'gene_id_to_gene_name.txt'),
            map_speciment_to_gene_file=os.path.join('Data', 'Benito', 'map_speciment_to_gene.csv')
            )
        print("Data loaded and preprocessed.")
        filter_function, _, _ = filter_proximity_benito(df_human)
        print("Data filtered.")
        data_human = filter_function.loc[(filter_function != 0).any(axis=1)]
        print(data_human.head())
        print("Performing Granger causality tests...")
        gc_results = perform_gc_benito(data_human, genes_file=os.path.join('Data', 'Benito', 'gene_names_all.txt'), progress=True)
        print("Granger causality tests performed.")

        save_results_to_csv_kutsche(gc_results, "granger_causality_results_benito.csv")
        print("granger_causality_results_benito.csv")

if __name__ == '__main__':
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    main()
