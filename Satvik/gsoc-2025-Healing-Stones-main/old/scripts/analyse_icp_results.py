import pandas as pd

def analyze_icp_results(csv_file, top_n=5):
    
    df = pd.read_csv(csv_file)

    # Sroting by fitness and rmse
    df_sorted = df.sort_values(by=['Fitness', 'RMSE'], ascending=[False, True])

    # top N pairs
    print(f"\nTop {top_n} Fragment Pairs Based on Fitness Score:\n")
    print(df_sorted.head(top_n).to_string(index=False))

    return df_sorted

if __name__ == "__main__":
    csv_path = "icp_results.csv"  
    top_n = 5  

    sorted_results = analyze_icp_results(csv_path, top_n=top_n)

    # save updated csv
    sorted_results.to_csv("sorted_icp_results.csv", index=False)
    print("\nSorted results saved to 'sorted_icp_results.csv'")
