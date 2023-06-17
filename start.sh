echo "This script is designed to run experiment iteratively."

# shellcheck disable=SC2162
read -p "Please input the start iteration(default 0):" start
start=${start:-0}

# shellcheck disable=SC2162
read -p "Please input the end iteration(default 30):" end
end=${end:-30}

# shellcheck disable=SC2162
read -p "Please input the filename(iter.py | iter_alex.py): " filename
filename=${filename:-iter.py}

echo "Got it. $filename from $start from to $end"

# shellcheck disable=SC2162
read -p "Press Enter to start..."

nohup bash iter.sh $start $end $filename > iter.out &

#bash iter.sh $start $end $filename
