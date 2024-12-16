#!/bin/bash

for FOLD in $(seq 1 5)
do
  echo "Running fold $FOLD..."

  # Calculate the port number dynamically (e.g., base port is 8088, increment by 1 for each fold)
  PORT=$((8080 + FOLD))

  # Check if any process is using the desired port and kill it if found
  echo "Checking if port $PORT is in use..."
  PID=$(lsof -t -i:$PORT)

  if [ ! -z "$PID" ]; then
    echo "Port $PORT is in use by process $PID. Killing the process..."
    kill -9 $PID
  else
    echo "Port $PORT is free."
  fi


  # Update FOLD and SERVER_ADDRESS in the config.json file
  jq --argjson fold "$FOLD" --arg port "0.0.0.0:$PORT" '.dataset_path.FOLD = $fold | .server.SERVER_ADDRESS = $port' config.json > temp.json && mv temp.json config.json

  # Remove any existing server_done.txt file before starting the server
  rm -f server_done.txt

  # Start the server and redirect its output to a log file, capture its PID
  echo "Starting the server for fold $FOLD on port $PORT..."
  START=$(date +%s)
  python server.py > ./logs/server_fold_$FOLD.log 2>&1 &
  SERVER_PID=$!

  # Give the server some time to start up
  sleep 5

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

  echo "Server process for fold $FOLD has finished."

  # Kill the server process if it's still running
  if ps -p $SERVER_PID > /dev/null; then
    echo "Killing the server process for fold $FOLD..."
    kill -9 $SERVER_PID
  fi

  # Clean up the done flag file
  rm -f server_done.txt

  # End time
  END=$(date +%s)
  DIFF=$((END - START))

  # Display total time
  echo "Total training time for fold $FOLD: $DIFF seconds"
done
