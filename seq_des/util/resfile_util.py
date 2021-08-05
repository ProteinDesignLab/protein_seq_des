# developed by Damir Temir | github.com/dtemir | as a part of the RosettaCommons Summer Internship

import common
import re

def read_resfile(filename):
    """ 
    read a resfile and return a dictionary of constraints for each residue id
    
    the constraints is a dictionary where the keys are residue ids and values are the amino acids to restrict
    (passed residue ids in the resfile are subtracted 1 because the count in PDBs starts from 1,
    while in the logits the count is from 0)

    example:
    65 ALLAA # allow all amino acids at residue id 65 (64 in the tensor)
    54 ALLAAxc # allow all amino acids except cysteine at residue id 54 (53 in the tensor)
    30 POLAR # allow only polar amino acids at residue id 30 (29 in the tensor)
    31 - 33 NOTAA CFYG # disallow the specified amino acids at residue ids 31 to 33 (30 to 32 in the tensor) 
    43 TPIKAA C # allow only cysteine when initializing the sequence (same logic for TNOTAA)

    results into a dictionary: 
    {64: {}, 53: {'C'}, 29: {'T', 'R', 'K', 'Q', 'D', 'E', 'S', 'N', 'H'},
    30: {'C', 'F', 'Y', 'G'}, 31: {'C', 'F', 'Y', 'G'}, 32: {'C', 'F', 'Y', 'G'}}

    plus it returns a header from check_for_header():
    {"DEFAULT": {}}

    plus it returns a dictionary with the amino acids for initial sequence (NOTE: amino acids listed will NOT be used to initialize the sequence)
    {42: 'C'}
    """
    def place_constraints(constraint, init_seq):
        """
        places the constraints in the appropriate dicts 
        -initial_seq for building the initial sequence with TPIKAA and TNOTAA
        -constraints for restricting the conditional model with PIKAA, NOTAA, ALLAA, POLAR, etc.
        """
        if not init_seq:
            constraints[res_id] = constraint
        else:
            initial_seq[res_id] = constraint

    constraints = dict() # amino acids to restrict in the design
    header, start_id = check_for_header(filename) # amino acids to use as default for those not specified in constraints
    initial_seq = dict() # amino acids to use when initializing the sequence

    with open(filename, "r") as f:
        # iterate over the lines and extract arguments (residue id, command)
        lines = f.readlines()
        for line in lines[start_id + 1:]:
            args = [arg.strip() for arg in line.split(" ")]
            is_integer(args[0]) # the res id needs to be an integer
            assert isinstance(args[1], str), "the resfile command needs to be a string"
            
            res_id = int(args[0]) - 1
            if args[1] == "-": # if given a range of residue ids (ex. 31 - 33 NOTAA)
                is_integer(args[2]) # the res id needs to be an integer
                for res_id in range(res_id, int(args[2])):
                    constraint, init_seq  = check_for_commands(args, 3, 4)
                    place_constraints(constraint, init_seq)
            else: # if not given a range (ex. 31 NOTAA CA)
                constraint, init_seq = check_for_commands(args, 1, 2)
                place_constraints(constraint, init_seq)
    
    # update the initial seq dictionary to only have one element per residue id (at random)
    initial_seq = {res_id : (common.atoms.resfile_commands["ALLAAwc"] - restricted_aa).pop() for res_id, restricted_aa in initial_seq.items()}

    return constraints, header, initial_seq

def check_for_header(filename):
    """
    read a resfile and return the header if present

    the header is commands that should be applied by default
    to all residues that are not specified after the 'start' keyword

    example of a header:
    ALLLA # allows all amino acids for residues that are not specified in the body
    START # divides the body and header
    # ... the body starts here, see read_resfile()
    """
    header = {}
    start_id = -1
    with open(filename, "r") as f:
        start = re.compile(r"\bSTART|start\b")
        # if the file has the keyword start, extract header
        if bool(start.search(f.read())):
            f.seek(0) # set the cursor back to the beginning
            lines = f.readlines()
            for i, line in enumerate(lines):
                if start.match(line):
                    start_id = i # the line number where start is used (divides header and body)
                    break
                args = line.split()
                args.insert(0, "") # check_for_commands only handles the second argument (first is usually res_id)
                header['DEFAULT'] = check_for_commands(args, 1, 2)

    return header, start_id


def check_for_commands(args, command_id, list_id):
    """
    converts given commands into sets of amino acids to restrict in the logits
    
    so far, it handles these commands: ALLAA, ALLAAxc, POLAR, APOLAR, NOTAA, PIKAA

    command_id - the index where the command is within the args
    list_id - the index where the possible list of AA is within the args (only for NOTAA and PIKAA) 
    """
    constraint = set()
    command = args[command_id].upper()
    init_seq = False # reflect if it's TPIKAA or TNOTAA
    if command in common.atoms.resfile_commands.keys():
        constraint = common.atoms.resfile_commands["ALLAAwc"] - common.atoms.resfile_commands[command]
    elif "PIKAA" in command: # allow only the specified amino acids
        constraint = common.atoms.resfile_commands["ALLAAwc"] - set(args[list_id].strip())
    elif "NOTAA" in command: # disallow only the specified amino acids
        constraint = set(args[list_id].strip())

    if command == "TPIKAA" or command == "TNOTAA":
        init_seq = True

    return constraint, init_seq

def get_natro(filename):
    """
    provides a list of indecies whose input rotamers and identities need to be presevered (Native Rotamer - NATRO)

    overrides the sampler.py's self.fixed_idx attribute with a list of the NATRO residues to be skipped in the 
    self.get_blocks() function that picks sampling blocks

    if ALL residues in the resfile are NATRO, the sampler.py's self.step() skips running the neural network for
    amino acid prediction AND rotamer prediction
    """
    fixed_idx = set()
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            args = [arg.strip().upper() for arg in line.split(" ")]
            if "NATRO" in args:
                is_integer(args[0])
                if args[1] == "-": # provided a range of NATRO residues
                    is_integer(args[2])
                    fixed_idx.update(range(int(args[0]) - 1, int(args[2])))
                else: # provided a single NATRO residue
                    fixed_idx.add(int(args[0]) - 1)

    return list(fixed_idx)

def is_integer(n):
    try:
        int(n)
    except ValueError:
        raise ValueError("Incorrect residue index in the resfile ", n)
