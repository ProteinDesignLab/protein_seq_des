import re

def read_resfile(filename):
    """ 
    read a resfile and return a dictionary of constraints for each residue id
    
    the constraints is a dictionary where the keys are residue ids and values are their commands
    (passed residue ids in the resfile are subtracted 1 because the count in PDBs starts from 1,
    while in the logits the count is from 0)

    Example:
    65 ALLAA # allow all amino acids at residue id 65
    54 ALLAAxc # allow all amino acids except cysteine at residue id 54
    30 POLAR # allow only polar amino acids at residue id 30

    results into a dictionary {65: "ALLAA", 54: "ALLAAxc", 30: "POLAR"},
    plus it returns a header {"DEFAULT": "ALLAA"}
    """
    constraints = {}
    header, start_id = check_for_header(filename)
    with open(filename, "r") as f:
        # iterate over the lines and extract arguments (residue id, command)
        lines = f.readlines()
        for line in lines[start_id + 1:]:
            args = line.split(" ")
            assert isinstance(int(args[0]), int), "the resfile residue id needs to be an integer"
            assert isinstance(args[1], str), "the resfile command needs to be a string"
            constraints[int(args[0]) - 1] = args[1].strip()

    return constraints, header

def check_for_header(filename):
    """
    read a resfile and return the header if present

    the header is commands that should be applied by default
    to all residues that are not specified after the 'start' keyword

    Example of a header:
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
                header['DEFAULT'] = args[0]

    return header, start_id
