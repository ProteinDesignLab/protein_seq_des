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

    results into a dictionary: 
    {64: {}, 53: {'C'}, 29: {'T', 'R', 'K', 'Q', 'D', 'E', 'S', 'N', 'H'},
    30: {'C', 'F', 'Y', 'G'}, 31: {'C', 'F', 'Y', 'G'}, 32: {'C', 'F', 'Y', 'G'}}

    plus it returns a header from check_for_header():
    {"DEFAULT": {}}
    """
    constraints = dict()
    header, start_id = check_for_header(filename)
    with open(filename, "r") as f:
        # iterate over the lines and extract arguments (residue id, command)
        lines = f.readlines()
        for line in lines[start_id + 1:]:
            args = [arg.strip() for arg in line.split(" ")]
            assert isinstance(int(args[0]), int), "the resfile residue id needs to be an integer"
            assert isinstance(args[1], str), "the resfile command needs to be a string"
            
            res_id = int(args[0]) - 1
            if args[1] == "-": # if given a range of residue ids (ex. 31 - 33 NOTAA)
                for res_id in range(res_id, int(args[2])):
                    constraints[res_id] = check_for_commands(args, 3, 4)
            else:
                constraints[res_id] = check_for_commands(args, 1, 2)

    return constraints, header

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
    if command in common.atoms.resfile_commands.keys():
        constraint = common.atoms.resfile_commands["ALLAAwc"] - common.atoms.resfile_commands[command]
    elif command == "PIKAA": # allow only the specified amino acids
        constraint = common.atoms.resfile_commands["ALLAAwc"] - set(args[list_id].strip())
    elif command == "NOTAA": # disallow only the specified amino acids
        constraint = set(args[list_id].strip())

    return constraint
