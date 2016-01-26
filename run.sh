if [ ! -e "emails.txt" ] || [ ! -e "training_data" ]; then
  curl -L -o data.zip "https://www.dropbox.com/s/e1f6retimzix1xx/data.zip?dl=1" && unzip -n data.zip
fi

# Yeah, yeah, I know... but please, keep in mind that this is do-it-fast uni project
cd training/keras && python learn_and_evaluate.py ; cd ../..
