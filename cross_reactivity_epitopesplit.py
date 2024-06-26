import pandas as pd
from itertools import combinations
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#import data
with open('C:/Users/loekd/OneDrive/Documenten//SearchTable-2024-02-06 12_24_08.532.tsv', 'r') as file:
    VDJDB_data = pd.read_csv(file, sep='\t')


# functions used in this file
def encode_protseq(new_seq, values):
    encoded_sequence = [values[amino] for amino in new_seq]
    return encoded_sequence


def encode_df_column(frame, column, values):
    frame[column] = frame[column].apply(lambda seq: encode_protseq(seq, values))


def split_list_column_inplace(frame, column_name):
    # Extract the column containing lists
    list_column = frame[column_name]

    # Create new columns for each element in the list
    max_list_length = list_column.apply(len).max()
    new_columns = [f"{column_name}_{i + 1}" for i in range(max_list_length)]

    # Split the list into separate columns and add them to the same DataFrame
    frame[new_columns] = pd.DataFrame(list_column.tolist(), index=frame.index)

    # Drop the original list column inplace
    frame.drop(columns=[column_name], inplace=True)


def calculate_hydrophobicity(sequence):
    kd_hydrophobicity = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    total_hydrophobicity = 0.0
    for amino_acid in sequence:
        if amino_acid in kd_hydrophobicity:
            total_hydrophobicity += kd_hydrophobicity[amino_acid]
    return total_hydrophobicity / len(sequence)


def add_hydrophobicity_difference_column(df, sequence_column_1, sequence_column_2, new_column):
    hydrophobicity_differences = []
    for seq1, seq2 in zip(df[sequence_column_1], df[sequence_column_2]):
        hydrophobicity1 = calculate_hydrophobicity(seq1)
        hydrophobicity2 = calculate_hydrophobicity(seq2)
        hydrophobicity_difference = hydrophobicity1 - hydrophobicity2
        hydrophobicity_differences.append(hydrophobicity_difference)
    df[new_column] = hydrophobicity_differences
    return df


def encode_df_column_binary(frame, column, value):
    frame[column] = frame[column].apply(lambda x: 1 if x == value else 0)

# Define the full BLOSUM62 matrix
blosum62 = {
    ('A', 'A'): 4, ('A', 'R'): -1, ('A', 'N'): -2, ('A', 'D'): -2, ('A', 'C'): 0, ('A', 'Q'): -1, ('A', 'E'): -1,
    ('A', 'G'): 0, ('A', 'H'): -2, ('A', 'I'): -1, ('A', 'L'): -1, ('A', 'K'): -1, ('A', 'M'): -1, ('A', 'F'): -2,
    ('A', 'P'): -1, ('A', 'S'): 1, ('A', 'T'): 0, ('A', 'W'): -3, ('A', 'Y'): -2, ('A', 'V'): 0, ('R', 'R'): 5,
    ('R', 'N'): 0, ('R', 'D'): -2, ('R', 'C'): -3, ('R', 'Q'): 1, ('R', 'E'): 0, ('R', 'G'): -2, ('R', 'H'): 0,
    ('R', 'I'): -3, ('R', 'L'): -2, ('R', 'K'): 2, ('R', 'M'): -1, ('R', 'F'): -3, ('R', 'P'): -2, ('R', 'S'): -1,
    ('R', 'T'): -1, ('R', 'W'): -3, ('R', 'Y'): -2, ('R', 'V'): -3, ('N', 'N'): 6, ('N', 'D'): 1, ('N', 'C'): -3,
    ('N', 'Q'): 0, ('N', 'E'): 0, ('N', 'G'): 0, ('N', 'H'): 1, ('N', 'I'): -3, ('N', 'L'): -3, ('N', 'K'): 0,
    ('N', 'M'): -2, ('N', 'F'): -3, ('N', 'P'): -2, ('N', 'S'): 1, ('N', 'T'): 0, ('N', 'W'): -4, ('N', 'Y'): -2,
    ('N', 'V'): -3, ('D', 'D'): 6, ('D', 'C'): -3, ('D', 'Q'): 2, ('D', 'E'): 2, ('D', 'G'): -1, ('D', 'H'): -1,
    ('D', 'I'): -3, ('D', 'L'): -4, ('D', 'K'): -1, ('D', 'M'): -3, ('D', 'F'): -3, ('D', 'P'): -1, ('D', 'S'): 0,
    ('D', 'T'): -1, ('D', 'W'): -4, ('D', 'Y'): -3, ('D', 'V'): -3, ('C', 'C'): 9, ('C', 'Q'): -3, ('C', 'E'): -4,
    ('C', 'G'): -3, ('C', 'H'): -3, ('C', 'I'): -1, ('C', 'L'): -1, ('C', 'K'): -3, ('C', 'M'): -1, ('C', 'F'): -2,
    ('C', 'P'): -3, ('C', 'S'): -1, ('C', 'T'): -1, ('C', 'W'): -2, ('C', 'Y'): -2, ('C', 'V'): -1, ('Q', 'Q'): 5,
    ('Q', 'E'): 2, ('Q', 'G'): -2, ('Q', 'H'): 0, ('Q', 'I'): -3, ('Q', 'L'): -2, ('Q', 'K'): 1, ('Q', 'M'): 0,
    ('Q', 'F'): -3, ('Q', 'P'): -1, ('Q', 'S'): 0, ('Q', 'T'): -1, ('Q', 'W'): -2, ('Q', 'Y'): -1, ('Q', 'V'): -2,
    ('E', 'E'): 5, ('E', 'G'): -2, ('E', 'H'): 0, ('E', 'I'): -3, ('E', 'L'): -3, ('E', 'K'): 1, ('E', 'M'): -2,
    ('E', 'F'): -3, ('E', 'P'): -1, ('E', 'S'): 0, ('E', 'T'): -1, ('E', 'W'): -3, ('E', 'Y'): -2, ('E', 'V'): -2,
    ('G', 'G'): 6, ('G', 'H'): -2, ('G', 'I'): -4, ('G', 'L'): -4, ('G', 'K'): -2, ('G', 'M'): -3, ('G', 'F'): -3,
    ('G', 'P'): -2, ('G', 'S'): 0, ('G', 'T'): -2, ('G', 'W'): -2, ('G', 'Y'): -3, ('G', 'V'): -3, ('H', 'H'): 8,
    ('H', 'I'): -3, ('H', 'L'): -3, ('H', 'K'): -1, ('H', 'M'): -2, ('H', 'F'): -1, ('H', 'P'): -2, ('H', 'S'): -1,
    ('H', 'T'): -2, ('H', 'W'): -2, ('H', 'Y'): 2, ('H', 'V'): -3, ('I', 'I'): 4, ('I', 'L'): 2, ('I', 'K'): -3,
    ('I', 'M'): 1, ('I', 'F'): 0, ('I', 'P'): -3, ('I', 'S'): -2, ('I', 'T'): -1, ('I', 'W'): -3, ('I', 'Y'): -1,
    ('I', 'V'): 3, ('L', 'L'): 4, ('L', 'K'): -2, ('L', 'M'): 2, ('L', 'F'): 0, ('L', 'P'): -3, ('L', 'S'): -2,
    ('L', 'T'): -1, ('L', 'W'): -2, ('L', 'Y'): -1, ('L', 'V'): 1, ('K', 'K'): 5, ('K', 'M'): -1, ('K', 'F'): -3,
    ('K', 'P'): -1, ('K', 'S'): 0, ('K', 'T'): -1, ('K', 'W'): -3, ('K', 'Y'): -2, ('K', 'V'): -2, ('M', 'M'): 5,
    ('M', 'F'): 0, ('M', 'P'): -2, ('M', 'S'): -1, ('M', 'T'): -1, ('M', 'W'): -1, ('M', 'Y'): -1, ('M', 'V'): 1,
    ('F', 'F'): 6, ('F', 'P'): -4, ('F', 'S'): -2, ('F', 'T'): -2, ('F', 'W'): 1, ('F', 'Y'): 3, ('F', 'V'): -1,
    ('P', 'P'): 7, ('P', 'S'): -1, ('P', 'T'): -1, ('P', 'W'): -4, ('P', 'Y'): -3, ('P', 'V'): -2, ('S', 'S'): 4,
    ('S', 'T'): 1, ('S', 'W'): -3, ('S', 'Y'): -2, ('S', 'V'): -2, ('T', 'T'): 5, ('T', 'W'): -2, ('T', 'Y'): -2,
    ('T', 'V'): 0, ('W', 'W'): 11, ('W', 'Y'): 2, ('W', 'V'): -3, ('Y', 'Y'): 7, ('Y', 'V'): -1, ('V', 'V'): 4
}


# get score from BLOSUM62 matrix
def get_blosum62_score(aa1, aa2):
    if (aa1, aa2) in blosum62:
        return blosum62[(aa1, aa2)]
    elif (aa2, aa1) in blosum62:
        return blosum62[(aa2, aa1)]
    else:
        return 0  # If pair not found, return 0


# Function to calculate the total BLOSUM62 score for two sequences
def calculate_blosum62_score(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")
    total_score = 0
    for aa1, aa2 in zip(seq1, seq2):
        total_score += get_blosum62_score(aa1, aa2)
    return total_score


# Function to add BLOSUM62 score to dataframe
def add_blosum62_score(df, seq_col1, seq_col2, new_col):
    df[new_col] = df.apply(lambda row: calculate_blosum62_score(row[seq_col1], row[seq_col2]), axis=1)


# generating training data function
def get_epitope_df(epitopes, df, points=None):
    intermediate_df = df.loc[df['Epitope'].isin(epitopes)]
    # getting all combinations / using combinatios to avoid duplicate checking
    all_combinations = list(combinations(intermediate_df.iterrows(), 2))
    # selecting some amount of combinations
    if points is not None:
        if points <= len(all_combinations):
            selected_combinations = random.sample(all_combinations, points)
        else:
            selected_combinations = all_combinations
    else:
        selected_combinations = all_combinations

    # coupling sequences for positives
    # Initialize empty DataFrames for positive and negative results
    positive_rows = []
    negative_rows = []

    # checking if epitope is same or different
    for (i, row1), (j, row2) in selected_combinations:
        # Create a dictionary with columns from both row1 and row2
        combined_dict = {
            'CDR3_1': row1['CDR3'],
            'Epitope_1': row1['Epitope'],
            'Epitope gene_1': row1['Epitope gene'],
            'Epitope species_1': row1['Epitope species'],
            'Gene_1': row1['Gene'],
            'CDR3_2': row2['CDR3'],
            'Epitope_2': row2['Epitope'],
            'Epitope gene_2': row2['Epitope gene'],
            'Epitope species_2': row2['Epitope species'],
            'Gene_2': row2['Gene']
        }

        if row1['Epitope'] == row2['Epitope']:
            positive_rows.append(combined_dict)
        else:
            negative_rows.append(combined_dict)

    # Convert lists of dictionaries to DataFrames
    positive_df = pd.DataFrame(positive_rows)
    negative_df = pd.DataFrame(negative_rows)

    # Ensure columns are in the correct order
    columns = ['CDR3_1', 'Epitope_1', 'Epitope gene_1', 'Epitope species_1', 'Gene_1',
               'CDR3_2', 'Epitope_2', 'Epitope gene_2', 'Epitope species_2', 'Gene_2']
    positive_df = positive_df.reindex(columns=columns)
    negative_df = negative_df.reindex(columns=columns)

    del all_combinations, selected_combinations

    # Add a column indicating whether the row is positive or negative
    positive_df['Is_Positive'] = True
    negative_df['Is_Positive'] = False

    # equalizing amount of positive and negative data
    if len(positive_df) > len(negative_df):
        positive_df = positive_df.sample(n=len(negative_df), random_state=0)
    elif len(positive_df) < len(negative_df):
        negative_df = negative_df.sample(n=len(positive_df), random_state=0)

    # Combine the positive and negative DataFrames
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True)

    combined_df = combined_df[['CDR3_1', 'CDR3_2', 'Gene_1', 'Gene_2', 'Is_Positive', 'Epitope_1', 'Epitope_2']]
    del positive_df, negative_df

    # # calculate blossum score
    # add_blosum62_score(combined_df, 'CDR3_1', 'CDR3_2', 'Blosum62')

    add_hydrophobicity_difference_column(combined_df, 'CDR3_1', 'CDR3_2', 'hydrophobicity_difference')

    aa = ["L", "I", "N", "G", "V", "E", "P", "H", "K", "A", "Y", "W", "Q", "M", "S", "C", "T", "F", "R", "D"]

    aa_encode = [
        "00000", "00001", "00010", "00011", "00100", "00101", "00110", "00111", "01000", "01001",
        "01010", "01011", "01100", "01101", "01110", "01111", "10000", "10001", "10010", "10011"
    ]

    aa_encoding_series = pd.Series(aa_encode, index=aa)

    encode_df_column(combined_df, 'CDR3_1', aa_encoding_series)
    encode_df_column(combined_df, 'CDR3_2', aa_encoding_series)
    # encode TRA and TRB and HLA A and HLA B
    # encode_df_column_binary(combined_df, 'Gene', 'TRA') # possibly include this
    encode_df_column_binary(combined_df, 'Is_Positive', True)
    encode_df_column_binary(combined_df, 'Gene_1', 'TRA')
    encode_df_column_binary(combined_df, 'Gene_2', 'TRA')
    # split the list
    split_list_column_inplace(combined_df, 'CDR3_1')
    split_list_column_inplace(combined_df, 'CDR3_2')

    # encodining epitope (optional)
    # for epitope 1
    encode_df_column(combined_df, 'Epitope_1', aa_encoding_series)
    split_list_column_inplace(combined_df, 'Epitope_1')
    # # for epitope 2
    # encode_df_column(combined_df, 'Epitop_2', aa_encoding_series)
    # split_list_column_inplace(combined_df, 'Epitope_2')

    return combined_df


# removing duplicates
VDJDB_data_unique = VDJDB_data.drop_duplicates(subset=['CDR3'])
VDJDB_data_unique = VDJDB_data_unique.loc[(VDJDB_data_unique['CDR3'].str.len() == 15)]
# removing all unnecisairy columns
VDJDB_data2 = VDJDB_data_unique[['CDR3', 'Epitope', 'Epitope gene', 'Epitope species', 'Gene']]
del VDJDB_data_unique

# making a density plot for amount of TCR's per epitope
epitopes = []

# extracting all epitopes from data
for index, row in VDJDB_data2.iterrows():
    epitopes.append(row['Epitope'])

unique_epitopes = []
# counting amount of the same epitope
for epitope in epitopes:
    if epitope not in unique_epitopes:
        unique_epitopes.append(epitope)

print(f'amount unique epitopes: {len(unique_epitopes)}')
# Randomly sample 60% of the elements for training_epitopes
training_epitopes = random.sample(unique_epitopes, int(len(unique_epitopes) * 0.7))

# Create a set of the training_epitopes for efficient membership checking
training_set = set(training_epitopes)

# Create a list of the remaining elements
test_epitopes = [epitope for epitope in unique_epitopes if epitope not in training_set]



# create dataframe for epitopes
train_df = get_epitope_df(training_epitopes, VDJDB_data2, 10**7)
del training_epitopes
test_df = get_epitope_df(test_epitopes, VDJDB_data2, 10**6)
del test_epitopes

# make sure all the dataframes have the same columns
# Identify all unique column names from all dataframes
all_columns = set(train_df.columns).union(set(test_df.columns))

# Add any missing columns to each dataframe filling with None
for col in all_columns:
    if col not in train_df.columns:
        train_df[col] = None
    if col not in test_df.columns:
        test_df[col] = None

train_df = train_df[sorted(all_columns)]
test_df = test_df[sorted(all_columns)]

# make graph of blosum62 scores and statisticd
# # calculating average and stdev of hydrophobicity
# Blosum62_ident = []
# Blosum62_diffe = []
# for index, row in train_df.iterrows():
#     if row['Is_Positive'] == True:
#         Blosum62_ident.append(row['Blosum62'])
#     elif row['Is_Positive'] == False:
#         Blosum62_diffe.append(row['Blosum62'])
# for index, row in test_df.iterrows():
#     if row['Is_Positive'] == True:
#         Blosum62_ident.append(row['Blosum62'])
#     elif row['Is_Positive'] == False:
#         Blosum62_diffe.append(row['Blosum62'])
# print(
#     f'average blosum identical: {sum(Blosum62_ident) / len(Blosum62_ident)}, stdev: {np.std(Blosum62_ident)}')
# print(
#     f'average blosum different: {sum(Blosum62_diffe) / len(Blosum62_diffe)}, stdev: {np.std(Blosum62_diffe)}')
#
# Blosum62_df = pd.DataFrame({'Blosum62 score': Blosum62_diffe + Blosum62_ident,
#                                   'Is same': ['Different epitope'] * len(Blosum62_diffe) +
#                                              ['Same epitope'] * len(Blosum62_ident)})

# # Create a boxplot to compare the distributions of lengths between the two groups
# sns.displot(data=Blosum62_df, kind='kde', fill=True, x='Blosum62 score', hue='Is_same') # if smoothing not ok add bw_method= 0.3
# plt.savefig('plots/Blosum62_difference.png', bbox_inches='tight')
# plt.close()

# calculating average and stdev of hydrophobicity
hydrophobicity_identical = []
hydrophobicity_different = []
for index, row in train_df.iterrows():
    if row['Is_Positive'] == True:
        hydrophobicity_identical.append(row['hydrophobicity_difference'])
    elif row['Is_Positive'] == False:
        hydrophobicity_different.append(row['hydrophobicity_difference'])
for index, row in test_df.iterrows():
    if row['Is_Positive'] == True:
        hydrophobicity_identical.append(row['hydrophobicity_difference'])
    elif row['Is_Positive'] == False:
        hydrophobicity_different.append(row['hydrophobicity_difference'])

print(
    f'average hydro identical: {sum(hydrophobicity_identical) / len(hydrophobicity_identical)}'
    f', stdev: {np.std(hydrophobicity_identical)}')
print(
    f'average hydro different: {sum(hydrophobicity_different) / len(hydrophobicity_different)}, '
    f'stdev: {np.std(hydrophobicity_different)}')

hydrophobicity_df = pd.DataFrame({'Hydrophobicity difference': hydrophobicity_different + hydrophobicity_identical,
                                  'Is same': ['Different epitope'] * len(hydrophobicity_different) +
                                             ['Same epitope'] * len(hydrophobicity_identical)})

# Create a boxplot to compare the distributions of lengths between the two groups
sns.displot(data=hydrophobicity_df, kind='kde', fill=True, x='Hydrophobicity difference', hue='Is same', bw_method=0.3)
plt.savefig('plots/Hydrophobicity_epitope.png', bbox_inches='tight')
plt.close()

train_df.drop(['Epitope_2'], axis=1, inplace=True)
test_df.drop(['Epitope_2'], axis=1, inplace=True)



# splitting dataset for model
y_train = train_df['Is_Positive']
X_train = train_df.drop('Is_Positive', axis=1)
y_test = test_df['Is_Positive']
X_test = test_df.drop('Is_Positive', axis=1)

del train_df, test_df

# # hyperparameter tuning
best_params = None
# Define the parameter grid for GridSearchCV
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(10, 30),
    'criterion': ['gini', 'entropy'],
    'max_leaf_nodes': randint(10, 30)
}

# Initialize RandomForestClassifier
clf = RandomForestClassifier(random_state=42)

# # Initialize GridSearchCV
# random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_dist,
#                                    n_iter=20, cv=3, n_jobs=2, verbose=2, random_state=42)
#
# # Fit GridSearchCV
# random_search.fit(X_train, y_train)
#
# # Get the best parameters from GridSearchCV
# best_params = random_search.best_params_
print("Best parameters found: ", best_params)

if best_params == None:
    best_params = {'n_estimators': 300, 'max_depth': 25, 'criterion': 'gini', 'max_leaf_nodes': None }

# Initialize RandomForestClassifier with the best parameters
best_clf = RandomForestClassifier(**best_params, random_state=42)

# Train the RandomForestClassifier with the best parameters
best_clf.fit(X_train, y_train)

# Evaluate the model
y_pred = best_clf.predict(X_test)

# Predict probabilities
y_prob = best_clf.predict_proba(X_test)[:, 1]  # Probability of being in the positive class

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Compute AUC
roc_auc = auc(fpr, tpr)


print(roc_auc)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('plots/epitope_prediction_hydro.png', bbox_inches='tight')

