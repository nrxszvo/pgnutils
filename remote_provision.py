import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("remote", help="remote host login, e.g. ubuntu@192.168.0.0")
parser.add_argument(
    "--tokenfile",
    default="~/.githubaccesstoken",
    help="text file containing github acccess token",
)
args = parser.parse_args()

with open(args.tokenfile) as f:
    token = f.readline()

cmd = f"scp provision.sh {args.remote}:~"
subprocess.call(cmd, shell=True)
cmd = f'ssh {args.remote} "chmod 755 provision.sh; GHTOKEN={token} sh provision.sh"'
subprocess.call(cmd, shell=True)
