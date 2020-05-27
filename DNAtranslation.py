input_file = 'DNA.txt'
with open(input_file, 'r') as f:
    seq = f.read()
    seq = seq.replace('\n', '')
    seq = seq.replace('\r','')

def translate(seq):
    table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }

    protein = ''
    if len(seq)%3==0:
        for i in range(0,len(seq),3):
            characters = seq[i:i+3]
            protein += table[characters]

    return protein

#creating function to read file
def read_seq(input_file):
    '''Reads and returns the input sequence with special characters removed.'''
    with open(input_file, 'r') as f:
        seq = f.read()
    seq = seq.replace('\n', '')
    seq = seq.replace('\r','')
    return seq

#creating read objects
prt = read_seq('proteins.txt')
dna = read_seq('DNA.txt')

#translating into proteins
result = translate(dna[20:938])

#False because have extra 3 characters called stop codons, so need to slice off end
result==prt
result = translate(dna[20:935])
result==prt #now True