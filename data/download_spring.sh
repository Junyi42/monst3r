# download spring dataset
gdown --folder https://drive.google.com/drive/folders/1oJqS7YOqtgO6l4WI_fdCZ-Jvp2RUvHZz?usp=sharing -O spring
cd spring
# unzip all
find . -name "*.zip" -exec unzip -o -q {} \;
# remove all zip files
find . -name "*.zip" -exec rm {} \;
# move data/spring/spring to data/spring
mv spring/* .
rm -rf spring