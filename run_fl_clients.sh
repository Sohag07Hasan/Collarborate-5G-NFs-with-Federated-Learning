#!/bin/bash

# Start clients in the background and capture their PIDs
echo "Starting client 1 for fold $FOLD..."
python c1.py > ./logs/client1_fold_$FOLD.log 2>&1 &
CLIENT1_PID=$!

echo "Starting client 2 for fold $FOLD..."
python c2.py > ./logs/client2_fold_$FOLD.log 2>&1 &
CLIENT2_PID=$!

echo "Starting client 3 for fold $FOLD..."
python c3.py > ./logs/client3_fold_$FOLD.log 2>&1 &
CLIENT3_PID=$!

echo "Starting client 4 for fold $FOLD..."
python c4.py > ./logs/client4_fold_$FOLD.log 2>&1 &
CLIENT4_PID=$!

  # Monitor for server completion by checking for the "done" flag in the file
echo "Monitoring server completion..."
while [ ! -f server_done.txt ]
do
  echo "Server is still running..."
  sleep 120  # Check every 120 seconds
done