if [ ! -e "emails.txt" ] || [ ! -e "training_data" ]; then
  curl -L -o data.zip "https://www.dropbox.com/s/e1f6retimzix1xx/data.zip?dl=1" && unzip -n data.zip
fi

cd training/keras && python3 learn_and_evaluate.py ; cd ../..
