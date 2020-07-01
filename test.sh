echo greedy
python run_match.py -f -r 10 -o GREEDY -p 4
echo random
python run_match.py -f -r 10 -o RANDOM -p 4
echo minimax
python run_match.py -f -r 10 -o MINIMAX -p 4
echo self
python run_match.py -f -r 10 -o SELF -p 4