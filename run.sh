#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_FILE=$1
PYTHON_PATH="$SCRIPT_DIR/$PYTHON_FILE"  

echo $PYTHON_PATH

Help()
{
   echo "Usage: srun -p gpu  --gres gpu:p100:1 --time=1-0 run.sh <python_file> <train_config> <model_config>"
   echo
   echo "Required Arguments:"
   echo "<python_file> -- 'run.py' or 'embed.py'"
   echo "<train_config>"
   echo "<model_config>"
   echo
   echo "See $PYTHON_PATH for more"
   echo
}

while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;;
   esac
done

DTSTR=`date +"%Y-%m-%d__%H-%M-%S"`

python "$PYTHON_PATH" "--train_config=$2" "--model_config=$3" > ./logs/stdout/"${DTSTR}.out" 2>&1
