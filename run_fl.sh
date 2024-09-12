#!/bin/bash

# Start the server and redirect its output to a log file
echo "Starting the server..."
START=$(date +%s)
python server.py > ./logs/server.log 2>&1 &

# Give the server some time to start up
sleep 5

# Start monitoring the server log
echo "Monitoring server output..."
tail -f ./logs/server.log &

# Start clients in the background
echo "Starting client 1..."
python c1.py > ./logs/client1.log 2>&1 &

echo "Starting client 2..."
python c2.py > ./logs/client2.log 2>&1 &

echo "Starting client 3..."
python c3.py > ./logs/client3.log 2>&1 &

echo "Starting client 4..."
python c4.py > ./logs/client4.log 2>&1 &

# Wait for all background processes to finish
wait

# End time
END=$(date +%s)
DIFF=$((END - START))

# Display total time
echo "Total training time: $DIFF seconds"
