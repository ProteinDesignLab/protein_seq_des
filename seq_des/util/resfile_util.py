import common.atoms

def read_resfile(filename):
    # read resfile and return a dictionary of constraints for each residue id
    constraints = {}
    with open(filename, "r") as f:
        # iterate over the lines and extract arguments (residue id, command)
        for line in f:
            args = line.split()
            assert isinstance(int(args[0]), int), "the resfile residue id needs to be an integer"
            assert isinstance(args[1], str), "the resfile command needs to be a string"
            constraints[int(args[0])] = args[1]

    return constraints
