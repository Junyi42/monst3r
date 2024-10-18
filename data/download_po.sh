# Download point_odyssey
mkdir -p point_odyssey
cd point_odyssey
# train
gdown --id 1ivaHRZV6iwxxH4qk8IAIyrOF9jrppDIP
# test
gdown --id 1jn8l28BBNw9f9wYFmd5WOCERH48-GsgB
# sample
gdown --id 1dnl9XMImdwKX2KcZCTuVDhcy5h8qzQIO
# unzip all *.tar.gz
find . -name "*.tar.gz" -exec tar -zxvf {} \;
# remove all zip files
find . -name "*.tar.gz" -exec rm {} \;