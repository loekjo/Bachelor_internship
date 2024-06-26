import pandas as pd
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import matplotlib.pyplot as plt

# importing data and exporting it to fasta format

with open('C:/Users/loekd/OneDrive/Documenten/SearchTable-2024-02-06 12_24_08.532.tsv', 'r') as file:
    data = pd.read_csv(file, sep='\t')
data = data.drop_duplicates(subset=['CDR3'])


def create_alignment_from_dataframe(df, column,  min_length=None, max_length=None):
    sequences = []
    for seq_str in df[column]:
        seq = Seq(seq_str)
        if (min_length is None or len(seq) >= min_length) and (max_length is None or len(seq) <= max_length):
            seq_record = SeqRecord(seq)
            sequences.append(seq_record)
    alignment = MultipleSeqAlignment(sequences)
    return alignment


def calculate_mutual_information(alignment):
    num_sequences = len(alignment)
    alignment_length = alignment.get_alignment_length()
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')

    mutual_information_matrix = np.zeros((alignment_length, alignment_length))

    for i in range(alignment_length):
        for j in range(i + 1, alignment_length):
            mutual_information = 0
            for aa1 in amino_acids:
                for aa2 in amino_acids:
                    count_aa1_aa2 = sum(seq[i] == aa1 and seq[j] == aa2 for seq in alignment)
                    if count_aa1_aa2 > 0:
                        p_aa1_aa2 = count_aa1_aa2 / num_sequences
                        p_aa1 = sum(seq[i] == aa1 for seq in alignment) / num_sequences
                        p_aa2 = sum(seq[j] == aa2 for seq in alignment) / num_sequences
                        mutual_information += p_aa1_aa2 * np.log2(p_aa1_aa2 / (p_aa1 * p_aa2))
            mutual_information_matrix[i][j] = mutual_information
            mutual_information_matrix[j][i] = mutual_information
    return mutual_information_matrix


def remove_after_character_in_df(frame, column, char):
    frame.loc[:, column] = frame[column].apply(lambda text: text.split(char)[0])


remove_after_character_in_df(data, 'MHC A', '*')

TRA = data[data['Gene'] == 'TRA']
TRB = data[data['Gene'] == 'TRB']

data_TRA = create_alignment_from_dataframe(df= TRA, column= 'CDR3', max_length=15, min_length=15 )
data_TRB = create_alignment_from_dataframe(df= TRB, column= 'CDR3', max_length=15, min_length=15 )

mi_matrix_TRA = calculate_mutual_information(data_TRA)
mi_matrix_TRB = calculate_mutual_information(data_TRB)


# Example matrices for demonstration
mi_matrix_TRA = np.random.rand(15, 15)
mi_matrix_TRB = np.random.rand(15, 15)

fig, (ax1, ax2) = plt.subplots(1, 2)

cax1 = ax1.imshow(mi_matrix_TRA, cmap='viridis', interpolation='nearest')
ax1.set_title('TRA')
ax1.set_xlabel('amino acid position')
ax1.set_ylabel('amino acid position')
ax1.set_xticks(range(1, 16, 5))
ax1.set_yticks(range(1, 16, 5))

cax2 = ax2.imshow(mi_matrix_TRB, cmap='viridis', interpolation='nearest')
ax2.set_title('TRB')
ax2.set_xlabel('amino acid position')
ax2.set_ylabel('amino acid position')
ax2.set_xticks(range(0, 16, 5))
ax2.set_yticks(range(0, 16, 5))

# Create a single color bar
cbar = fig.colorbar(cax2, ax=[ax1, ax2], orientation='vertical')
cbar.set_label('Mutual Information')


plt.savefig('plots/mutual_information_matrixv2.png')
plt.close()

TRA_HLAA = data[(data['Gene'] == 'TRA') & (data['MHC A'] == 'HLA-A')]
TRA_HLAB = data[(data['Gene'] == 'TRA') & (data['MHC A'] == 'HLA-B')]
TRB_HLAA = data[(data['Gene'] == 'TRB') & (data['MHC A'] == 'HLA-A')]
TRB_HLAB = data[(data['Gene'] == 'TRB') & (data['MHC A'] == 'HLA-B')]


if len(TRA_HLAA) > 2000:
    TRA_HLAA = TRA_HLAA.sample(n=2000, random_state=0)
if len(TRA_HLAB) > 2000:
    TRA_HLAB = TRA_HLAB.sample(n=2000, random_state=0)
if len(TRB_HLAA) > 2000:
    TRB_HLAA = TRB_HLAA.sample(n=2000, random_state=0)
if len(TRB_HLAB) > 2000:
    TRB_HLAB = TRB_HLAB.sample(n=2000, random_state=0)

data_TRA_HLAA = create_alignment_from_dataframe(df=TRA_HLAA, column='CDR3', min_length=15, max_length=15)
data_TRA_HLAB = create_alignment_from_dataframe(df=TRA_HLAB, column='CDR3', min_length=15, max_length=15)
data_TRB_HLAA = create_alignment_from_dataframe(df=TRB_HLAA, column='CDR3', min_length=15, max_length=15)
data_TRB_HLAB = create_alignment_from_dataframe(df=TRB_HLAB, column='CDR3', min_length=15, max_length=15)

mi_matrix_TRA_HLAA = calculate_mutual_information(data_TRA_HLAA)
mi_matrix_TRA_HLAB = calculate_mutual_information(data_TRA_HLAB)
mi_matrix_TRB_HLAA = calculate_mutual_information(data_TRB_HLAA)
mi_matrix_TRB_HLAB = calculate_mutual_information(data_TRB_HLAB)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 11))

cax1 = ax[0, 0].imshow(mi_matrix_TRA_HLAA, cmap='viridis', interpolation='nearest')
ax[0, 0].set_title('TRA HLA-A')
ax[0, 0].set_xlabel('amino acid position')
ax[0, 0].set_ylabel('amino acid position')
ax[0, 0].set_xticks(range(0, 15, 5))
ax[0, 0].set_yticks(range(0, 15, 5))

cax2 = ax[0, 1].imshow(mi_matrix_TRB_HLAA, cmap='viridis', interpolation='nearest')
ax[0, 1].set_title('TRB HLA-A')
ax[0, 1].set_xlabel('amino acid position')
ax[0, 1].set_ylabel('amino acid position')
ax[0, 1].set_xticks(range(0, 15, 5))
ax[0, 1].set_yticks(range(0, 15, 5))

cax3 = ax[1, 0].imshow(mi_matrix_TRA_HLAB, cmap='viridis', interpolation='nearest')
ax[1, 0].set_title('TRA HLA-B')
ax[1, 0].set_xlabel('amino acid position')
ax[1, 0].set_ylabel('amino acid position')
ax[1, 0].set_xticks(range(0, 15, 5))
ax[1, 0].set_yticks(range(0, 15, 5))

cax4 = ax[1, 1].imshow(mi_matrix_TRB_HLAB, cmap='viridis', interpolation='nearest')
ax[1, 1].set_title('TRB HLA-B')
ax[1, 1].set_xlabel('amino acid position')
ax[1, 1].set_ylabel('amino acid position')
ax[1, 1].set_xticks(range(0, 15, 5))
ax[1, 1].set_yticks(range(0, 15, 5))

# Create a single color bar
cbar = fig.colorbar(cax4, ax=ax.ravel().tolist(), orientation='vertical')
cbar.set_label('Mutual Information')

fig.tight_layout()
plt.savefig('plots/mutual_information_MHCA.png', bbox_inches='tight')
plt.close()