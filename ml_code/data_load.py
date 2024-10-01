import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path: str):
    try:
        df = pd.read_csv(file_path)
        sns.countplot(x='Attrition',data=df)
        plt.show()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
