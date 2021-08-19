# import sys

# if len(sys.argv) == 0:
#     print("Please provide a path to a .fasta file with a sequence")
#     sys.exit()

# def get_sequence(path):
#     """
#     Get a sequence from a FASTA file with an initial sequence for the Protein Sequence Design Algorithm
#     """
#     sequence = ""
#     with open(path, "r") as f:
#         lines = f.readlines()
#         print(lines)
#         sequence = lines[1] + lines[0]

#     print(sequence)

sequence = "TMPSTYAFKLPIQTETGVARVRSVIKKVSLTLSAYQVDYLLNTATVTSPVAWADMVDGVQAAGVEIQYGQFF"
sequence = list(sequence)

with open("init_seq_1cc8_gt.txt", "w") as file1:
    for i in range(1, len(sequence) + 1):
    # command = " ".join([str(i), "TPIKAA", sequence[i], "\n"]
        file1.write("{} TPIKAA {} \n".format(str(i), sequence[i-1]))

# get_sequence("../../../sequenced_results/1bkr_gt_sequenced/init_seq.fasta")
