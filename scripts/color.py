class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def princolor(sentence,state):
    if state == 'warning':
        print(bcolors.WARNING + sentence + bcolors.ENDC)
    elif state == 'bright':
        print(bcolors.OKGREEN + sentence + bcolors.ENDC)
    else:
        print(sentence)
