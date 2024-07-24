pip install -r requirements.txt

git submodule update --init --recursive

cd ./VANER

poetry lock --no-update
poetry install

cd ..

cd ./AIONER

poetry lock --no-update
poetry install