home_dir=$(dirname $0)
cd $home_dir

rm -rf "GenePlexusZooManuscript.tar.gz"*

if [ -d data ]; then
	echo cleaning up old data
	rm -rf ./data/*
else
	mkdir data
fi

wget https://zenodo.org/record/7888044/files/GenePlexusZooManuscript.tar.gz
tar -xvzf "GenePlexusZooManuscript.tar.gz" -C data --strip-components=1
rm -f "GenePlexusZooManuscript.tar.gz"
