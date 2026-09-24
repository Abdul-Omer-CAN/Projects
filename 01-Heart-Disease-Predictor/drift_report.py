# Import section #
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Load reference data #
url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
reference_data = pd.read_csv(url)
reference_data = reference_data.drop('target', axis=1)

# Simulate new data #
current_data = reference_data.copy()
current_data['age'] = current_data['age'] + 15
current_data['chol'] = current_data['chol'] * 1.3

# Build and run the drift report #

report = Report([DataDriftPreset()]) # report object that runs the drift detection stat.
my_eval = report.run(current_data, reference_data) # compares current_data against reference_data.

# Save the report #

my_eval.save_html("drift_report.html")
print("Drift report generated: drift_report.html")