Cross-Domain Adversarial Attacks: Network Intrusion to CAN Bus Evasion
This repository implements the attack framework described in the research paper "Cross-Domain Adversarial Attacks: Translating Network Intrusion to CAN Bus Evasion in V2X Environments." The framework enables cross-domain adversarial attacks by translating network-based intrusions into Controller Area Network (CAN) bus evasion in vehicle-to-everything (V2X) environments.
Overview
The repository contains a comprehensive implementation of:

Data Preprocessing Pipeline: Handles missing values, normalizes features, and prepares datasets (KDDCup99, UNSW-NB15, CICIoV2024) for the attack framework.
Data Balancing: Applies Synthetic Minority Oversampling Technique (SMOTE) to handle class imbalance in the datasets.
Ensemble Feature Selection (EFS): Implements Algorithm 1 from the paper, combining multiple machine learning models to identify the most important features for cross-domain attacks.
V2X-Aware Projected Gradient Descent (VX-PGD): Implements Algorithm 2 from the paper, generating adversarial examples while respecting domain-specific protocol constraints.
CAN-Deep PackGen: Implements Algorithm 3 from the paper, transforming adversarial network features into protocol-compliant CAN frames.
Evaluation Framework: Comprehensive metrics to assess attack performance, including evasion rate, distortion measures, translation success, and more.

Installation

# Clone the repository
git clone https://github.com/yourusername/cross-domain-adversarial-attacks.git
cd cross-domain-adversarial-attacks

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Requirements

Python 3.8+
NumPy
Pandas
Scikit-learn
TensorFlow 2.x
Matplotlib
Seaborn
SciPy
tqdm

Usage
Data Preprocessing

from data_preprocessing import DataPreprocessor

# Initialize preprocessor for your dataset type
preprocessor = DataPreprocessor('kddcup99')  # or 'unsw-nb15' or 'ciciov2024'

# Load the data
data = preprocessor.load_data('path_to_your_dataset.csv')

# Apply full preprocessing pipeline
X_train, X_test, y_train, y_test = preprocessor.prepare_data(
    data, 
    target_column='label'  # Adjust target column name as needed
)

# Save preprocessed data to CSV
preprocessor.save_to_csv(X_train, X_test, y_train, y_test, output_dir='./preprocessed_data')

Ensemble Feature Selection (EFS)
from ensemble_feature_selection import EnsembleFeatureScoring, process_dataset_with_efs

# Use the convenience function for a complete pipeline
results = process_dataset_with_efs(
    input_file="your_balanced_dataset.csv",
    output_dir="./efs_output",
    top_k=10,  # Number of top features to select
    target_column="label"
)

# Or use the class directly
efs = EnsembleFeatureScoring(top_k=10)
efs.fit(X, y)
best_features = efs.get_best_features()

V2X-Aware Projected Gradient Descent (VX-PGD)

from vx_pgd import VXProjectedGradientDescent, generate_adversarial_examples

# Generate adversarial examples using the convenience function
results = generate_adversarial_examples(
    input_file="selected_features_dataset.csv",
    output_dir="./vxpgd_results",
    target_column="label",
    dataset_type="network",  # 'network', 'can', or 'v2x'
    epsilon=0.1,            # Maximum perturbation
    alpha=0.01,             # Step size
    iterations=10           # Number of iterations
)

# Or use the class directly
vx_pgd = VXProjectedGradientDescent(
    model=classifier,
    feature_names=feature_names,
    dataset_type='network',
    epsilon=0.1,
    alpha=0.01,
    iterations=10
)
X_adv, success_rate = vx_pgd.generate(X_test, y_test)

CAN-Deep PackGen

from can_deep_packgen import CANDeepPackGen, process_csv_with_can_deep_packgen

# Process network adversarial examples and translate to CAN frames
results = process_csv_with_can_deep_packgen(
    network_csv="network_adversarial_examples.csv",
    output_dir="./can_deep_packgen_results",
    train_mode=False
)

# With CAN training data for better translation
results = process_csv_with_can_deep_packgen(
    network_csv="network_adversarial_examples.csv",
    output_dir="./can_deep_packgen_results",
    train_mode=True,
    can_csv="can_samples.csv"
)

Datasets
The code supports the following datasets:

KDDCup99: Network intrusion detection dataset with 41 features.
UNSW-NB15: Modern network intrusion detection dataset with 49 features.
CICIoV2024: CAN bus dataset for monitoring vehicular communication with 15 features.

You need to download these datasets separately from their respective sources.
Research Paper Implementation
This repository implements the methods described in the paper "Cross-Domain Adversarial Attacks: Translating Network Intrusion to CAN Bus Evasion in V2X Environments." The key implemented algorithms are:

Algorithm 1: Ensemble Feature Scoring (EFS)
Algorithm 2: V2X-Aware Projected Gradient Descent (VX-PGD)
Algorithm 3: CAN-Deep PackGen


Acknowledgments
This implementation is based on the research paper "Cross-Domain Adversarial Attacks: Translating Network Intrusion to CAN Bus Evasion in V2X Environments."








