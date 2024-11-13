import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tokenfile",
    default=os.path.expanduser("~/.githubaccesstoken"),
    help="text file containing github acccess token",
)

parser.add_argument("remote", help="remote host login, e.g. ubuntu@192.168.0.0")
parser.add_argument("myname", help='for git config, e.g. "John Doe"')
parser.add_argument("myemail", help="for git config, e.g. johndoe@gmail.com")

args = parser.parse_args()

myname = f'"{args.myname}"'
myemail = f'"{args.myemail}"'

with open(args.tokenfile) as f:
    token = f.readline()

cmd = f"scp provision.sh {args.remote}:~"
subprocess.call(cmd, shell=True)
cmd = f'ssh {args.remote} "chmod 755 provision.sh; GHTOKEN={token} MYNAME={myname} MYEMAIL={myemail} sh provision.sh"'
print(cmd)
# subprocess.call(cmd, shell=True)
