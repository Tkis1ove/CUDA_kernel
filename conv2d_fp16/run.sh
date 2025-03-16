make clean
make

total_time=0

echo "-----------------------------------sample1-----------------------------"
output=$(./conv2dfp16 64 256 14 14 256 3 3 1 1 1 1)
echo "$output"
sample1_time=$(echo "$output" | grep "time:" | awk '{print $2}')
weighted_sample1_time=$(echo "$sample1_time * 0.20" | bc)
total_time=$(echo "$total_time + $weighted_sample1_time" | bc)

echo "-----------------------------------sample2-----------------------------"
output=$(./conv2dfp16 256 192 14 14 192 3 3 1 1 1 1)
echo "$output"
sample2_time=$(echo "$output" | grep "time:" | awk '{print $2}')
weighted_sample2_time=$(echo "$sample2_time * 0.20" | bc)
total_time=$(echo "$total_time + $weighted_sample2_time" | bc)

echo "-----------------------------------sample3-----------------------------"
output=$(./conv2dfp16 16 256 26 26 512 3 3 1 1 1 1)
echo "$output"
sample3_time=$(echo "$output" | grep "time:" | awk '{print $2}')
weighted_sample3_time=$(echo "$sample3_time * 0.20" | bc)
total_time=$(echo "$total_time + $weighted_sample3_time" | bc)

echo "-----------------------------------sample4-----------------------------"
output=$(./conv2dfp16 32 256 14 14 256 3 3 1 1 1 1)
echo "$output"
sample4_time=$(echo "$output" | grep "time:" | awk '{print $2}')
weighted_sample4_time=$(echo "$sample4_time * 0.20" | bc)
total_time=$(echo "$total_time + $weighted_sample4_time" | bc)

echo "-----------------------------------sample5-----------------------------"
output=$(./conv2dfp16 2 1280 16 16 1280 3 3 1 1 1 1)
echo "$output"
sample5_time=$(echo "$output" | grep "time:" | awk '{print $2}')
weighted_sample5_time=$(echo "$sample5_time * 0.10" | bc)
total_time=$(echo "$total_time + $weighted_sample5_time" | bc)

echo "-----------------------------------sample6-----------------------------"
output=$(./conv2dfp16 2 960 64 64 32 3 3 1 1 1 1)
echo "$output"
sample6_time=$(echo "$output" | grep "time:" | awk '{print $2}')
weighted_sample6_time=$(echo "$sample6_time * 0.10" | bc)
total_time=$(echo "$total_time + $weighted_sample6_time" | bc)

echo "-----------------------------------Your Grad-----------------------------"
echo "$total_time"